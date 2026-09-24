"""Nonlinear beam FEA of a whole compliant mechanism, driven by input rotation.

Where the PRBM replaces each flexure with a pin and a spring, this meshes the
mechanism as it is actually built: flexures as slender beams, rigid links as stout
ones, ground clamped, and the input link's rotation **prescribed**. Equilibrium is
found by Newton-Raphson at every step of the arc.

The mesh is built from the same attachment points the CAD uses, so the thing
being solved is the thing being printed rather than a separate idealisation.

What this captures that the PRBM cannot
---------------------------------------
* **Distributed compliance.** The flexure bends along its length rather than
  hinging at a point, and the shape of that bend changes with load.
* **Axial and shear compliance.** A real flexure stretches, so the effective link
  lengths are not constant. The PRBM's pin cannot do this -- and it is the main
  reason the two predict different coupler paths at all, since with a prescribed
  input and no external load the PRBM path is pure geometry.
* **Load-dependent pivot drift.** The effective pivot moves as the load ratio
  changes through the cycle; the PRBM fixes it.

Cost
----
Each input angle is a nonlinear solve, so this is seconds where the PRBM is
milliseconds. That is the trade the dataset is built around: PRBM for volume, FEA
for accuracy, and the disagreement between them recorded as a metric.

Deformed shape
--------------
:class:`SimulationResult` records where the *joints* and output points went, which
is what the metrics need. The bent shape of each flexure -- what a viewer draws --
is discarded, because keeping it for every sample of a Phase B batch would dwarf
the sample itself. :func:`solve_with_shape` keeps it, returning a
:class:`DeformedShape` beside the usual result.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from cmtool.convert.base import CompliantMechanism
from cmtool.convert.placement import attachment_points
from cmtool.core.graph import Linkage
from cmtool.core.provenance import Provenance
from cmtool.core.units import FloatArray, rotation_matrix, wrap_to_pi
from cmtool.materials.loader import Printer
from cmtool.solvers.base import SOLVERS, MechanismState, SimulationResult
from cmtool.solvers.beam import BeamModel, BeamSection, solve

#: Beam elements per flexure. Convergence is second order, and the validation
#: sweep shows 8 elements already gives sub-1e-3 L accuracy on a bent cantilever.
DEFAULT_FLEXURE_ELEMENTS = 8

#: Beam elements per rigid link. Two is plenty: these barely deform.
DEFAULT_LINK_ELEMENTS = 2

#: Largest prescribed rotation increment per load step, in degrees.
DEFAULT_STEP_DEG = 2.0


class BeamMesh:
    """Mesh of a compliant mechanism, plus the bookkeeping to read results back."""

    def __init__(
        self,
        mechanism: CompliantMechanism,
        *,
        link_width_mm: float,
        part_thickness_mm: float,
        youngs_modulus_mpa: float,
        flexure_elements: int = DEFAULT_FLEXURE_ELEMENTS,
        link_elements: int = DEFAULT_LINK_ELEMENTS,
    ) -> None:
        self.mechanism = mechanism
        self.linkage: Linkage = mechanism.base
        self.attachments, self.axes = attachment_points(mechanism, self.linkage)

        self._points: list[FloatArray] = []
        self._elements: list[tuple[int, int]] = []
        self._sections: list[BeamSection] = []
        self.flexure_elements: dict[str, list[int]] = {}
        self.link_elements: dict[str, list[int]] = {}
        self.attachment_node: dict[tuple[str, str], int] = {}

        link_section = BeamSection.rectangular(link_width_mm, part_thickness_mm, youngs_modulus_mpa)

        # Flexures first, so their end nodes exist for the links to attach to.
        for joint_name, joint in self.linkage.joints.items():
            sized = mechanism.sizing[joint_name]
            host = sized.host_body
            other = joint.other(host)
            root = self.attachments[host][joint_name]
            tip = self.attachments[other][joint_name]
            section = BeamSection.rectangular(
                sized.geometry.thickness_mm, sized.geometry.width_mm, youngs_modulus_mpa
            )
            chain = self._chain(root, tip, flexure_elements, section)
            self.flexure_elements[joint_name] = chain[2]
            self.attachment_node[(host, joint_name)] = chain[0]
            self.attachment_node[(other, joint_name)] = chain[1]

        # Rigid links between each body's two attachment nodes.
        self.body_nodes: dict[str, tuple[int, int]] = {}
        for body in self.linkage.bodies:
            joints = self.linkage.joints_of(body)
            if len(joints) != 2:
                continue
            first = self.attachment_node[(body, joints[0])]
            second = self.attachment_node[(body, joints[1])]
            self.body_nodes[body] = (first, second)
            if self.linkage.bodies[body].is_ground:
                continue  # ground is clamped, so it needs no elements
            self.link_elements[body] = self._connect(first, second, link_elements, link_section)

        self.model = BeamModel(
            np.asarray(self._points, dtype=float), self._elements, self._sections
        )

        # Output points, expressed in the frame of their body's two attachment nodes.
        self.output_local: dict[str, tuple[str, FloatArray]] = {}
        for name, output in self.linkage.outputs.items():
            if output.body not in self.body_nodes:
                continue
            origin, angle = self._body_frame(output.body, self.model.nodes)
            local = rotation_matrix(-angle) @ (output.position_mm - origin)
            self.output_local[name] = (output.body, local)

    # ------------------------------------------------------------- building

    def _node(self, point: FloatArray) -> int:
        self._points.append(np.asarray(point, dtype=float))
        return len(self._points) - 1

    def _chain(
        self, start: FloatArray, end: FloatArray, count: int, section: BeamSection
    ) -> tuple[int, int, list[int]]:
        """Create a chain of elements; return (first node, last node, element ids)."""
        nodes = [self._node(start)]
        for index in range(1, count + 1):
            fraction = index / count
            nodes.append(self._node(np.asarray(start) + fraction * (np.asarray(end) - start)))
        ids = []
        for first, second in itertools.pairwise(nodes):
            self._elements.append((first, second))
            self._sections.append(section)
            ids.append(len(self._elements) - 1)
        return nodes[0], nodes[-1], ids

    def _connect(self, first: int, second: int, count: int, section: BeamSection) -> list[int]:
        """Connect two existing nodes with a chain of intermediate nodes.

        Returns the element ids in order from ``first`` to ``second``, so a
        caller can walk the chain as a polyline.
        """
        start = self._points[first]
        end = self._points[second]
        previous = first
        ids: list[int] = []
        for index in range(1, count):
            fraction = index / count
            previous_new = self._node(np.asarray(start) + fraction * (np.asarray(end) - start))
            self._elements.append((previous, previous_new))
            self._sections.append(section)
            ids.append(len(self._elements) - 1)
            previous = previous_new
        self._elements.append((previous, second))
        self._sections.append(section)
        ids.append(len(self._elements) - 1)
        return ids

    def _body_frame(self, body: str, coords: FloatArray) -> tuple[FloatArray, float]:
        first, second = self.body_nodes[body]
        origin = coords[first]
        along = coords[second] - origin
        return origin, float(np.arctan2(along[1], along[0]))

    # --------------------------------------------------------------- results

    def ground_dofs(self) -> dict[int, float]:
        """Clamped degrees of freedom: every DOF of both ground attachment nodes."""
        ground = self.linkage.ground
        fixed: dict[int, float] = {}
        for joint in self.linkage.joints_of(ground):
            node = self.attachment_node[(ground, joint)]
            for component in range(3):
                fixed[self.model.dof(node, component)] = 0.0
        return fixed

    def input_rotation_dof(self) -> int:
        """Return the DOF whose value is the prescribed input rotation."""
        node = self.attachment_node[(self.linkage.input_body, self.linkage.input_joint)]
        return self.model.dof(node, 2)

    def body_angles(self, displacement: FloatArray) -> dict[str, float]:
        """Absolute orientation of each body, from its two attachment nodes."""
        coords = self.model.deformed_nodes(displacement)
        angles = {}
        for body in self.linkage.bodies:
            if body not in self.body_nodes:
                angles[body] = 0.0
                continue
            if self.linkage.bodies[body].is_ground:
                angles[body] = 0.0
                continue
            angles[body] = self._body_frame(body, coords)[1]
        return angles

    def joint_positions(self, displacement: FloatArray) -> dict[str, FloatArray]:
        """Characteristic-pivot position of each joint in the deformed state."""
        coords = self.model.deformed_nodes(displacement)
        out = {}
        for joint_name, joint in self.linkage.joints.items():
            sized = self.mechanism.sizing[joint_name]
            host = sized.host_body
            other = joint.other(host)
            root = coords[self.attachment_node[(host, joint_name)]]
            tip = coords[self.attachment_node[(other, joint_name)]]
            out[joint_name] = root + (tip - root) * sized.pivot_fraction
        return out

    def output_points(self, displacement: FloatArray) -> dict[str, FloatArray]:
        """World position of each tracked output point."""
        coords = self.model.deformed_nodes(displacement)
        out = {}
        for name, (body, local) in self.output_local.items():
            origin, angle = self._body_frame(body, coords)
            out[name] = origin + rotation_matrix(angle) @ local
        return out

    def element_thickness_mm(self) -> FloatArray:
        """In-plane thickness of every element: flexure ``t``, or link width."""
        return np.asarray([s.in_plane_thickness_mm for s in self._sections], dtype=float)

    def element_strains(self, displacement: FloatArray) -> FloatArray:
        """Peak surface bending strain in every element, ``curvature * t / 2``."""
        return self.model.element_curvatures(displacement) * self.element_thickness_mm() / 2.0

    def flexure_strains(self, displacement: FloatArray) -> dict[str, float]:
        """Peak bending strain in each flexure, from element curvature."""
        curvatures = self.model.element_curvatures(displacement)
        out = {}
        for joint_name, ids in self.flexure_elements.items():
            thickness = self.mechanism.sizing[joint_name].geometry.thickness_mm
            out[joint_name] = float(max(curvatures[i] for i in ids) * thickness / 2.0)
        return out


@dataclass(frozen=True)
class DeformedShape:
    """The bent geometry of a meshed mechanism across a whole sweep.

    This is the geometry a viewer draws: not the idealised centrelines of the
    rigid model, but the beam elements where the solver actually put them, with
    the strain each one carries.

    Attributes
    ----------
    nodes_mm
        ``(n_states, n_nodes, 2)`` deformed node coordinates, in mm.
    elements
        ``(n_elements, 2)`` node index pairs.
    element_strain
        ``(n_states, n_elements)`` peak surface bending strain per element,
        ``curvature * t / 2`` with ``t`` that element's in-plane thickness.
    element_thickness_mm
        ``(n_elements,)`` in-plane thickness of each element: the flexure
        thickness for a flexure, the link width for a link. This is what lets a
        drawing show true proportions rather than a stick figure.
    flexure_elements
        Element indices belonging to each joint's flexure.
    body_elements
        Element indices belonging to each moving body's rigid link. Ground is
        absent: it is clamped, so it is never meshed.
    """

    nodes_mm: FloatArray
    elements: FloatArray
    element_strain: FloatArray
    element_thickness_mm: FloatArray
    flexure_elements: dict[str, list[int]]
    body_elements: dict[str, list[int]]

    @property
    def n_states(self) -> int:
        """Number of swept states."""
        return int(self.nodes_mm.shape[0])

    def element_polyline(self, state: int, element: int) -> FloatArray:
        """Return the ``(2, 2)`` deformed end coordinates of one element."""
        first, second = self.elements[element]
        return np.asarray(
            [self.nodes_mm[state, int(first)], self.nodes_mm[state, int(second)]], dtype=float
        )

    def chain_polyline(self, state: int, element_ids: list[int]) -> FloatArray:
        """Return the deformed polyline through a connected chain of elements.

        The chain is walked in the order given, which is the order the mesh was
        built in, so consecutive elements share a node.
        """
        if not element_ids:
            return np.empty((0, 2))
        points = [self.nodes_mm[state, int(self.elements[element_ids[0]][0])]]
        points.extend(self.nodes_mm[state, int(self.elements[i][1])] for i in element_ids)
        return np.asarray(points, dtype=float)


class BeamFeaSolver:
    """Nonlinear beam FEA of a compliant mechanism over a prescribed input arc."""

    name = "beam_fea"

    def can_solve(self, linkage: Linkage) -> bool:
        """Return whether this linkage can be meshed: every body needs two joints."""
        return all(len(linkage.joints_of(body)) == 2 for body in linkage.bodies)

    def solve(
        self,
        linkage: Linkage,
        input_angles_rad: FloatArray,
        **kwargs: Any,
    ) -> SimulationResult:
        """Solve the meshed mechanism at each input angle.

        The sweep is solved in order with each step warm-started from the last
        converged state, which is what keeps Newton in its basin across a large
        total rotation.
        """
        kwargs.pop("capture_shape", None)
        result, _ = self.solve_with_shape(linkage, input_angles_rad, capture_shape=False, **kwargs)
        return result

    def solve_with_shape(
        self,
        linkage: Linkage,
        input_angles_rad: FloatArray,
        *,
        mechanism: CompliantMechanism | None = None,
        flexure_elements: int = DEFAULT_FLEXURE_ELEMENTS,
        link_elements: int = DEFAULT_LINK_ELEMENTS,
        link_width_mm: float = 8.0,
        step_deg: float = DEFAULT_STEP_DEG,
        capture_shape: bool = True,
        **_: Any,
    ) -> tuple[SimulationResult, DeformedShape | None]:
        """Solve as :meth:`solve` does, optionally keeping the deformed geometry.

        Parameters
        ----------
        capture_shape
            Keep the deformed node coordinates and per-element strain of every
            state, returned as a :class:`DeformedShape`. ``False`` returns
            ``None`` in its place, which is what :meth:`solve` does so that batch
            runs do not carry geometry they will discard.
        """
        if mechanism is None:
            raise ValueError(
                "the beam_fea solver needs the compliant mechanism (flexure geometry); "
                "call simulate(converted_mechanism, solver='beam_fea')"
            )

        provenance = Provenance(
            config_hash=mechanism.provenance.config_hash,
            notes={"solver": self.name, **mechanism.provenance.notes},
        )
        provenance.merge(mechanism.provenance)

        printer = Printer.load(mechanism.printer_name)
        modulus = float(mechanism.diagnostics.get("youngs_modulus_MPa") or 0.0)
        if modulus <= 0.0:
            from cmtool.materials.loader import Material

            modulus = Material.load(mechanism.material_name).youngs_modulus_mpa(provenance)

        mesh = BeamMesh(
            mechanism,
            link_width_mm=link_width_mm,
            part_thickness_mm=printer.part_thickness_mm(provenance),
            youngs_modulus_mpa=modulus,
            flexure_elements=flexure_elements,
            link_elements=link_elements,
        )

        fixed = mesh.ground_dofs()
        rotation_dof = mesh.input_rotation_dof()
        reference = math.radians(mechanism.reference_input_deg)

        angles = np.atleast_1d(np.asarray(input_angles_rad, dtype=float))
        displacement = np.zeros(mesh.model.n_dof)
        previous_target = 0.0

        states: list[MechanismState] = []
        torque = np.zeros(angles.size)
        strains: dict[str, list[float]] = {j: [] for j in linkage.joints}
        iterations: list[int] = []
        shape_nodes: list[FloatArray] = []
        shape_strain: list[FloatArray] = []

        for index, angle in enumerate(angles):
            target = float(wrap_to_pi(angle - reference))
            increment = abs(target - previous_target)
            steps = max(1, math.ceil(math.degrees(increment) / max(step_deg, 1e-6)))
            outcome = solve(
                mesh.model,
                fixed_dofs=fixed,
                prescribed_dofs={rotation_dof: target},
                steps=steps,
                initial=displacement,
            )
            displacement = outcome.displacement
            previous_target = target
            iterations.append(sum(outcome.iterations))

            torque[index] = float(outcome.reaction[rotation_dof])
            for joint_name, value in mesh.flexure_strains(displacement).items():
                strains[joint_name].append(value)
            if capture_shape:
                shape_nodes.append(mesh.model.deformed_nodes(displacement))
                shape_strain.append(mesh.element_strains(displacement))

            states.append(
                MechanismState(
                    input_angle_rad=float(angle),
                    joint_positions_mm=mesh.joint_positions(displacement),
                    body_angles_rad=mesh.body_angles(displacement),
                    output_points_mm=mesh.output_points(displacement),
                    branch=mechanism.base.meta.get("assembly_branch", 1),
                )
            )

        flexure_strain = {name: np.asarray(v, dtype=float) for name, v in strains.items()}
        peak = {name: float(np.max(v)) for name, v in flexure_strain.items()}
        allowable = mechanism.feasibility.allowable_strain
        overall = max(peak.values()) if peak else 0.0

        result = SimulationResult(
            solver=self.name,
            linkage=linkage,
            states=states,
            provenance=provenance,
            input_torque_nmm=torque,
            flexure_strain=flexure_strain,
            diagnostics={
                "n_nodes": mesh.model.n_nodes,
                "n_elements": len(mesh.model.elements),
                "flexure_elements": flexure_elements,
                "link_elements": link_elements,
                "newton_iterations_total": sum(iterations),
                "peak_strain": peak,
                "peak_strain_overall": overall,
                "allowable_strain": allowable,
                "strain_margin": (allowable / overall) if overall > 0.0 else float("inf"),
                "max_input_torque_nmm": float(np.max(np.abs(torque))) if torque.size else 0.0,
                "prbm_models": mechanism.feasibility.prbm_models,
            },
        )

        if not capture_shape:
            return result, None

        shape = DeformedShape(
            nodes_mm=np.asarray(shape_nodes, dtype=float),
            elements=np.asarray(mesh.model.elements, dtype=int),
            element_strain=np.asarray(shape_strain, dtype=float),
            element_thickness_mm=mesh.element_thickness_mm(),
            flexure_elements={k: list(v) for k, v in mesh.flexure_elements.items()},
            body_elements={k: list(v) for k, v in mesh.link_elements.items()},
        )
        return result, shape


SOLVERS.add("beam_fea", BeamFeaSolver())


def solve_with_shape(
    mechanism: CompliantMechanism,
    input_angles_rad: FloatArray,
    **kwargs: Any,
) -> tuple[SimulationResult, DeformedShape]:
    """Run the beam FEA over an arc and keep the deformed geometry.

    The registered ``beam_fea`` solver returns only a
    :class:`~cmtool.solvers.base.SimulationResult`, because that is what the
    metrics and the dataset consume. Drawing the mechanism needs the bent shape
    of each flexure as well, and this is the way to get it.

    Parameters
    ----------
    mechanism
        The converted mechanism. Unlike :func:`cmtool.api.simulate` this takes
        the compliant mechanism directly, since the FEA cannot run without it.
    input_angles_rad
        Absolute input-link orientations to solve at, in radians.
    **kwargs
        Forwarded to :meth:`BeamFeaSolver.solve_with_shape`.

    Returns
    -------
    tuple
        The usual result, and the :class:`DeformedShape` of the same states.
    """
    solver = BeamFeaSolver()
    result, shape = solver.solve_with_shape(
        mechanism.base,
        input_angles_rad,
        mechanism=mechanism,
        capture_shape=True,
        **kwargs,
    )
    if shape is None:  # pragma: no cover - capture_shape is forced on above
        raise RuntimeError("beam FEA returned no deformed shape despite capture_shape=True")
    return result, shape
