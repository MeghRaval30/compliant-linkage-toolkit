"""The drawable form of a simulated mechanism, independent of how it is drawn.

A :class:`ViewerScene` is everything a picture of this project needs and nothing
about pixels: where every link and flexure sits at every step of the arc, the
four coupler paths, the strain and torque at each step, and the provenance that
says whether any of it is a physical prediction.

Two renderers consume it -- the self-contained HTML viewer in
:mod:`cmtool.viz.html`, and the PyVista viewer in A6 -- so the geometry is
decided once. It serialises to plain JSON, which is what lets the HTML viewer be
a single file with the scene baked into it.

What the scene contains, and what it deliberately does not
----------------------------------------------------------
The **beam FEA** contributes the real bent geometry: flexures as the solver
actually deformed them, each element carrying its own strain. The **rigid** and
**PRBM** models contribute centreline outlines at the same input angles, drawn as
ghosts, because a pin-jointed model has no flexure shape to show -- its flexures
are points. Seeing the three at one input angle is how pivot drift becomes
visible.

The printed part's **input lever, marker pads and base plate outline are not
here**. They are rigidly attached to bodies that the scene already draws, so they
add no information about the motion, and pulling them in would drag the CAD
kernel into the viewer.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cmtool.convert.base import CompliantMechanism
from cmtool.core.provenance import Provenance
from cmtool.core.units import FloatArray
from cmtool.metrics.paths import MeasuredPath, compare_paths, resample_to_angles
from cmtool.solvers.base import SimulationResult
from cmtool.solvers.beam_fea import DeformedShape

#: Models the scene can carry, in legend order.
MODEL_KEYS: tuple[str, ...] = ("rigid", "prbm", "fea", "measured")


@dataclass(frozen=True)
class ModelFrame:
    """One model's outline at one input angle.

    Attributes
    ----------
    links
        Body name to polyline, in mm. Two points for a centreline, more for a
        meshed link.
    flexures
        Joint name to polyline along the flexure. Empty for the rigid and PRBM
        models, whose flexures are single points by construction.
    flexure_strain
        Peak surface bending strain per element along each flexure polyline, so
        it has one fewer entry than that polyline has points.
    output_mm
        Where the tracked output point is.
    """

    links: dict[str, list[list[float]]]
    output_mm: list[float]
    flexures: dict[str, list[list[float]]] = field(default_factory=dict)
    flexure_strain: dict[str, list[float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        data: dict[str, Any] = {"links": self.links, "output": self.output_mm}
        if self.flexures:
            data["flexures"] = self.flexures
            data["strain"] = self.flexure_strain
        return data


@dataclass(frozen=True)
class ModelSeries:
    """One model's contribution to the scene.

    Attributes
    ----------
    key
        ``"rigid"``, ``"prbm"``, ``"fea"`` or ``"measured"``.
    path_mm
        The coupler path, one point per state.
    frames
        Per-state outlines. Empty for ``measured``, which is a path and nothing
        else: the camera sees marker pads, not links.
    input_angles_deg
        The angles the series was sampled at. For a measured series these are the
        camera's own angles, which need not match the simulated ones.
    torque_nmm
        Input torque per state, where the model computes one.
    deviation_mm
        Distance from the scene's reference path at each of this series' own
        angles. The three predicted paths differ by a few tenths of a
        millimetre over a mechanism 140 mm across, so they are indistinguishable
        drawn on top of each other; this is the series that makes the difference
        visible, and it is the quantity the go/no-go is stated against.
    """

    key: str
    path_mm: list[list[float]]
    input_angles_deg: list[float]
    frames: list[ModelFrame] = field(default_factory=list)
    torque_nmm: list[float] | None = None
    deviation_mm: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        data: dict[str, Any] = {
            "key": self.key,
            "path": self.path_mm,
            "input_angles_deg": self.input_angles_deg,
        }
        if self.frames:
            data["frames"] = [f.to_dict() for f in self.frames]
        if self.torque_nmm is not None:
            data["torque_nmm"] = self.torque_nmm
        if self.deviation_mm is not None:
            data["deviation_mm"] = self.deviation_mm
        return data


@dataclass
class ViewerScene:
    """Everything needed to draw a converted mechanism over its input arc.

    Attributes
    ----------
    allowable_strain
        The strain the colour ramp is normalised against.
    allowable_strain_is_placeholder
        Whether that number was measured. When ``True`` the ramp is a relative
        scale with no physical meaning, and every renderer must say so.
    """

    name: str
    joints: list[str]
    bodies: list[str]
    ground_body: str
    output_name: str
    input_angles_deg: list[float]
    input_sweep_deg: list[float]
    bounds_mm: list[float]
    ground_polyline_mm: list[list[float]]
    link_width_mm: float
    flexure_thickness_mm: dict[str, float]
    flexure_length_mm: dict[str, float]
    prbm_models: dict[str, str]
    allowable_strain: float
    allowable_strain_is_placeholder: bool
    models: dict[str, ModelSeries]
    deviation_reference: str
    comparisons: list[dict[str, Any]]
    provenance: Provenance
    notes: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        """Number of swept states."""
        return len(self.input_angles_deg)

    @property
    def caveat(self) -> str | None:
        """The placeholder caveat, or ``None`` when every input was measured."""
        return self.provenance.caveat()

    def to_dict(self) -> dict[str, Any]:
        """Return the whole scene as plain JSON-serialisable data."""
        return {
            "name": self.name,
            "joints": self.joints,
            "bodies": self.bodies,
            "ground_body": self.ground_body,
            "output_name": self.output_name,
            "input_angles_deg": self.input_angles_deg,
            "input_sweep_deg": self.input_sweep_deg,
            "bounds_mm": self.bounds_mm,
            "ground_polyline_mm": self.ground_polyline_mm,
            "link_width_mm": self.link_width_mm,
            "flexure_thickness_mm": self.flexure_thickness_mm,
            "flexure_length_mm": self.flexure_length_mm,
            "prbm_models": self.prbm_models,
            "allowable_strain": self.allowable_strain,
            "allowable_strain_is_placeholder": self.allowable_strain_is_placeholder,
            "models": {k: v.to_dict() for k, v in self.models.items()},
            "deviation_reference": self.deviation_reference,
            "comparisons": self.comparisons,
            "provenance": self.provenance.to_dict(),
            "caveat": self.caveat,
            "notes": self.notes,
            "meta": self.meta,
        }


def _centreline_frames(
    result: SimulationResult, output_name: str
) -> tuple[list[ModelFrame], list[list[float]]]:
    """Build ghost outlines: one straight bar between each body's two joints."""
    linkage = result.linkage
    two_jointed = {b: linkage.joints_of(b) for b in linkage.bodies}
    frames: list[ModelFrame] = []
    path: list[list[float]] = []
    for state in result.states:
        links: dict[str, list[list[float]]] = {}
        for body, joints in two_jointed.items():
            if len(joints) != 2:
                continue
            links[body] = [
                [float(x) for x in state.joint_positions_mm[joints[0]]],
                [float(x) for x in state.joint_positions_mm[joints[1]]],
            ]
        point = [float(x) for x in state.output_points_mm[output_name]]
        frames.append(ModelFrame(links=links, output_mm=point))
        path.append(point)
    return frames, path


def _fea_frames(
    result: SimulationResult, shape: DeformedShape, output_name: str
) -> tuple[list[ModelFrame], list[list[float]]]:
    """Build the real deformed outline, flexure by flexure, with per-element strain."""
    frames: list[ModelFrame] = []
    path: list[list[float]] = []
    for index, state in enumerate(result.states):
        flexures = {
            joint: [[float(x), float(y)] for x, y in shape.chain_polyline(index, ids)]
            for joint, ids in shape.flexure_elements.items()
        }
        strain = {
            joint: [float(shape.element_strain[index, i]) for i in ids]
            for joint, ids in shape.flexure_elements.items()
        }
        links = {
            body: [[float(x), float(y)] for x, y in shape.chain_polyline(index, ids)]
            for body, ids in shape.body_elements.items()
        }
        point = [float(x) for x in state.output_points_mm[output_name]]
        frames.append(
            ModelFrame(links=links, output_mm=point, flexures=flexures, flexure_strain=strain)
        )
        path.append(point)
    return frames, path


def _bounds(scene_points: list[list[float]], margin_mm: float) -> list[float]:
    array = np.asarray(scene_points, dtype=float)
    low = array.min(axis=0) - margin_mm
    high = array.max(axis=0) + margin_mm
    return [float(low[0]), float(low[1]), float(high[0]), float(high[1])]


def build_scene(
    mechanism: CompliantMechanism,
    *,
    n_steps: int = 41,
    input_angles_deg: FloatArray | None = None,
    output: str | None = None,
    measured: MeasuredPath | None = None,
    include: tuple[str, ...] = ("rigid", "prbm", "fea"),
    margin_mm: float = 6.0,
    **solver_kwargs: Any,
) -> ViewerScene:
    """Simulate a converted mechanism every requested way and assemble a scene.

    All simulated models are evaluated at **the same input angles**, which is what
    makes a pointwise path comparison meaningful and lets one slider drive all
    three outlines at once.

    Parameters
    ----------
    mechanism
        The converted mechanism, from :func:`cmtool.api.convert`.
    n_steps
        Samples across the arc. The FEA is the expensive one: a few seconds at
        41 steps for a four-bar.
    input_angles_deg
        Explicit angles, overriding ``n_steps``.
    output
        Tracked output point. Defaults to the linkage's only one.
    measured
        A measured path from :func:`cmtool.metrics.paths.read_path_csv`, drawn as
        a fourth series. Omitted entirely when absent -- never stubbed.
    include
        Which simulated models to run. Dropping ``"fea"`` makes the scene fast
        but leaves it with no flexure shapes and no strain to colour.
    margin_mm
        Padding around the geometry in the reported bounds.

    Returns
    -------
    ViewerScene
        Whose ``provenance`` carries the placeholders every solver used, so a
        renderer can state plainly that the numbers are not physical.

    Notes
    -----
    Every simulated model gets an explicit arc, taken from the mechanism. There is
    no full-revolution default anywhere in the toolkit, and this is no exception.
    """
    from cmtool.api import simulate
    from cmtool.solvers.beam_fea import solve_with_shape

    unknown = set(include) - {"rigid", "prbm", "fea"}
    if unknown:
        raise ValueError(f"unknown models {sorted(unknown)}; choose from rigid, prbm, fea")
    if not include:
        raise ValueError("include at least one of rigid, prbm, fea")

    linkage = mechanism.base
    output_name = output or next(iter(linkage.outputs))

    if input_angles_deg is not None:
        angles_deg = np.asarray(input_angles_deg, dtype=float)
    else:
        angles_deg = np.linspace(*mechanism.input_range_deg, int(n_steps))

    provenance = Provenance(
        config_hash=mechanism.provenance.config_hash,
        seed=mechanism.provenance.seed,
        notes={"scene": mechanism.name, **mechanism.provenance.notes},
    )
    provenance.merge(mechanism.provenance)

    models: dict[str, ModelSeries] = {}
    notes: list[str] = []
    all_points: list[list[float]] = []

    # The rigid model is the original pin-jointed linkage: pure geometry, and the
    # only series in the whole toolkit that is always a physical prediction.
    if "rigid" in include:
        rigid = simulate(linkage, solver="rigid", input_angles_deg=angles_deg)
        frames, path = _centreline_frames(rigid, output_name)
        models["rigid"] = ModelSeries(
            key="rigid",
            path_mm=path,
            input_angles_deg=[float(a) for a in angles_deg],
            frames=frames,
        )
        all_points.extend(path)

    if "prbm" in include:
        prbm = simulate(mechanism, solver="prbm", input_angles_deg=angles_deg)
        provenance.merge(prbm.provenance)
        frames, path = _centreline_frames(prbm, output_name)
        models["prbm"] = ModelSeries(
            key="prbm",
            path_mm=path,
            input_angles_deg=[float(a) for a in angles_deg],
            frames=frames,
            torque_nmm=[float(t) for t in prbm.input_torque_nmm]
            if prbm.input_torque_nmm is not None
            else None,
        )
        all_points.extend(path)

    if "fea" in include:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fea, shape = solve_with_shape(mechanism, np.radians(angles_deg), **solver_kwargs)
        provenance.merge(fea.provenance)
        frames, path = _fea_frames(fea, shape, output_name)
        models["fea"] = ModelSeries(
            key="fea",
            path_mm=path,
            input_angles_deg=[float(a) for a in angles_deg],
            frames=frames,
            torque_nmm=[float(t) for t in fea.input_torque_nmm]
            if fea.input_torque_nmm is not None
            else None,
        )
        all_points.extend(path)
        for frame in frames:
            for polyline in list(frame.links.values()) + list(frame.flexures.values()):
                all_points.extend(polyline)
    else:
        notes.append(
            "no beam FEA in this scene: flexures are drawn as pin joints and there is "
            "no strain to colour"
        )

    if measured is not None:
        models["measured"] = ModelSeries(
            key="measured",
            path_mm=[[float(x), float(y)] for x, y in measured.points_mm],
            input_angles_deg=[float(a) for a in measured.input_angles_deg],
        )
        all_points.extend(models["measured"].path_mm)
        if not measured.has_angles:
            notes.append(
                "the measured take has frames without an input angle, so it is compared "
                "by shape (Frechet) only"
            )
    else:
        notes.append("no measured path supplied: the measured series is absent, not zero")

    # The ground body never moves, so its outline is drawn once rather than
    # carried in every frame.
    ground_joints = linkage.joints_of(linkage.ground)
    ground_polyline = [
        [float(c) for c in linkage.joints[name].position_mm] for name in ground_joints
    ]

    reference_key = _add_deviations(models, angles_deg)
    comparisons = _comparisons(models, angles_deg, measured)

    sizing = mechanism.sizing
    allowable = mechanism.feasibility.allowable_strain
    placeholder_strain = any(
        name.endswith("allowable_strain") for name in provenance.placeholders_used
    )

    return ViewerScene(
        name=mechanism.name,
        joints=list(linkage.joints),
        bodies=list(linkage.bodies),
        ground_body=linkage.ground,
        output_name=output_name,
        input_angles_deg=[float(a) for a in angles_deg],
        input_sweep_deg=[float(a - angles_deg[0]) for a in angles_deg],
        bounds_mm=_bounds(all_points, margin_mm),
        ground_polyline_mm=ground_polyline,
        link_width_mm=float(solver_kwargs.get("link_width_mm", 8.0)),
        flexure_thickness_mm={n: float(s.geometry.thickness_mm) for n, s in sizing.items()},
        flexure_length_mm={n: float(s.geometry.length_mm) for n, s in sizing.items()},
        prbm_models=dict(mechanism.feasibility.prbm_models),
        allowable_strain=float(allowable),
        allowable_strain_is_placeholder=placeholder_strain,
        models=models,
        deviation_reference=reference_key,
        comparisons=comparisons,
        provenance=provenance,
        notes=notes,
        meta={
            "material": mechanism.material_name,
            "printer": mechanism.printer_name,
            "placement": mechanism.placement,
            "unstressed_at": mechanism.unstressed_at,
            "reference_input_deg": mechanism.reference_input_deg,
            "input_range_deg": list(mechanism.input_range_deg),
            "feasible": mechanism.feasibility.feasible,
            "binding_joint": mechanism.feasibility.binding_joint,
            "max_utilisation": mechanism.feasibility.max_utilisation,
        },
    )


def _add_deviations(models: dict[str, ModelSeries], angles_deg: FloatArray) -> str:
    """Fill in each series' distance from the reference path, and name that reference.

    The rigid path is the reference when it is present: it is pure geometry, the
    one series in the toolkit that is always a physical prediction, and the thing
    every other model is an approximation *of*. Failing that, whichever simulated
    model came first stands in, and the scene says which.
    """
    order = [k for k in ("rigid", "prbm", "fea") if k in models]
    if not order:
        return ""
    reference_key = order[0]
    reference = np.asarray(models[reference_key].path_mm, dtype=float)
    simulated = np.asarray(angles_deg, dtype=float)

    for key, series in models.items():
        points = np.asarray(series.path_mm, dtype=float)
        if key == "measured":
            angles = np.asarray(series.input_angles_deg, dtype=float)
            if np.any(np.isnan(angles)) or angles.size < 2:
                # Without angles there is nothing to line the two curves up by, and
                # inventing a correspondence is exactly what the Frechet distance
                # exists to avoid.
                continue
            matched = resample_to_angles(simulated, reference, angles)
        elif points.shape != reference.shape:
            continue
        else:
            matched = reference
        object.__setattr__(
            series,
            "deviation_mm",
            [float(v) for v in np.linalg.norm(points - matched, axis=1)],
        )
    return reference_key


def _comparisons(
    models: dict[str, ModelSeries],
    angles_deg: FloatArray,
    measured: MeasuredPath | None,
) -> list[dict[str, Any]]:
    """Pairwise path gaps between whichever models the scene has."""
    out: list[dict[str, Any]] = []
    simulated = [k for k in ("rigid", "prbm", "fea") if k in models]

    for index, first in enumerate(simulated):
        for second in simulated[index + 1 :]:
            comparison = compare_paths(
                np.asarray(models[first].path_mm, dtype=float),
                np.asarray(models[second].path_mm, dtype=float),
                labels=(first, second),
            )
            out.append(comparison.to_dict())

    if measured is None:
        return out

    measured_points = np.asarray(measured.points_mm, dtype=float)
    for key in simulated:
        predicted = np.asarray(models[key].path_mm, dtype=float)
        if measured.has_angles and measured_points.shape[0] >= 2:
            # Match by input angle: the camera's frames land wherever they land,
            # so the prediction is resampled onto them rather than the reverse.
            matched = resample_to_angles(
                np.asarray(angles_deg, dtype=float), predicted, measured.input_angles_deg
            )
            out.append(compare_paths(matched, measured_points, labels=(key, "measured")).to_dict())
        else:
            from cmtool.metrics.paths import discrete_frechet

            out.append(
                {
                    "a": key,
                    "b": "measured",
                    "mean_mm": None,
                    "max_mm": None,
                    "rms_mm": None,
                    "frechet_mm": discrete_frechet(predicted, measured_points),
                    "n_points": int(measured_points.shape[0]),
                    "note": "shape only: the measured take has no input angles to match on",
                }
            )
    return out
