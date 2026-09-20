"""Linkages as graphs.

A planar linkage is a graph: **bodies are nodes, joints are edges**, with tags
marking which body is ground, which joint is driven, and which points are
outputs. A four-bar is simply a graph with one independent loop and mobility 1;
nothing in this module knows what a four-bar is.

That is the whole point. Adding a five-bar (two independent inputs), a six-bar
(two loops) or a spatial mechanism later means adding a solver that declares it
can handle that topology -- not editing this file.

Angle convention
----------------
Body orientations and the input angle are **absolute**, measured from the world
+x axis, matching the usual four-bar ``theta2`` convention and the world
``joints_mm`` coordinates in the sample schema. Quantities that matter to a
compliant mechanism (flexure bend angle, joint excursion) are always reported
*relative to the reference configuration*, because that is the as-printed,
unstressed state.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cmtool.core.units import FloatArray, angle_of, rotation_matrix

#: Constraints removed by each joint kind in a planar mechanism.
PLANAR_CONSTRAINTS: dict[str, int] = {
    "revolute": 2,
    "prismatic": 2,
    # A compliant pivot still removes 2 DOF in the rigid-topology sense; its
    # compliance is a constitutive property, not a change of mobility.
    "flexure": 2,
}


class LinkageError(ValueError):
    """Raised when a linkage definition is inconsistent."""


@dataclass(frozen=True)
class Body:
    """A rigid body (link) of the mechanism."""

    name: str
    is_ground: bool = False


@dataclass(frozen=True)
class Joint:
    """A joint connecting exactly two bodies.

    Attributes
    ----------
    name
        Joint label, e.g. ``"A"``.
    bodies
        The two body names it connects.
    xy
        Position in the **reference configuration**, in mm, world frame.
    kind
        Joint type, a key of :data:`PLANAR_CONSTRAINTS`.
    """

    name: str
    bodies: tuple[str, str]
    xy: tuple[float, float]
    kind: str = "revolute"

    @property
    def position_mm(self) -> FloatArray:
        """Reference position as a ``(2,)`` array."""
        return np.asarray(self.xy, dtype=float)

    def other(self, body: str) -> str:
        """Return the body on the far side of this joint from ``body``."""
        if body == self.bodies[0]:
            return self.bodies[1]
        if body == self.bodies[1]:
            return self.bodies[0]
        raise LinkageError(f"joint {self.name!r} does not touch body {body!r}")


@dataclass(frozen=True)
class OutputPoint:
    """A tracked point rigidly attached to a body (e.g. the coupler point)."""

    name: str
    body: str
    xy: tuple[float, float]

    @property
    def position_mm(self) -> FloatArray:
        """Reference position as a ``(2,)`` array."""
        return np.asarray(self.xy, dtype=float)


@dataclass
class Linkage:
    """A planar linkage described as a tagged graph.

    Parameters
    ----------
    bodies, joints, outputs
        The graph itself plus tracked points.
    input_joint
        Name of the driven joint. The driven body is the non-ground body at that
        joint; the input coordinate is that body's absolute orientation.
    input_range_deg
        Absolute orientation range of the input body, ``(start, end)``.
        Flexures cannot rotate continuously, so this is always a limited arc --
        see ``docs/physics.md``.
    """

    bodies: dict[str, Body]
    joints: dict[str, Joint]
    outputs: dict[str, OutputPoint] = field(default_factory=dict)
    input_joint: str = ""
    input_range_deg: tuple[float, float] | None = None
    name: str = "linkage"
    meta: dict[str, Any] = field(default_factory=dict)

    # ---------------------------------------------------------------- topology

    def joints_of(self, body: str) -> list[str]:
        """Return the names of joints touching ``body``, in definition order."""
        return [j.name for j in self.joints.values() if body in j.bodies]

    def neighbours(self, body: str) -> list[str]:
        """Return the bodies adjacent to ``body``."""
        return [j.other(body) for j in self.joints.values() if body in j.bodies]

    @property
    def ground_bodies(self) -> list[str]:
        """Names of bodies tagged as ground."""
        return [b.name for b in self.bodies.values() if b.is_ground]

    @property
    def ground(self) -> str:
        """The single ground body name."""
        grounds = self.ground_bodies
        if len(grounds) != 1:
            raise LinkageError(f"expected exactly one ground body, found {grounds}")
        return grounds[0]

    @property
    def input_body(self) -> str:
        """The moving body driven at :attr:`input_joint`."""
        joint = self.joints[self.input_joint]
        moving = [b for b in joint.bodies if not self.bodies[b].is_ground]
        if len(moving) != 1:
            raise LinkageError(
                f"input joint {self.input_joint!r} must connect ground to exactly one "
                f"moving body, got bodies {joint.bodies}"
            )
        return moving[0]

    def mobility(self) -> int:
        """Planar mobility from the Gruebler/Kutzbach criterion.

        ``M = 3 (n - 1) - sum(constraints)``. A four-bar gives 1, a five-bar 2.
        Solvers use this, with the loop count, to decide whether they apply.
        """
        n = len(self.bodies)
        constraints = sum(PLANAR_CONSTRAINTS.get(j.kind, 2) for j in self.joints.values())
        return 3 * (n - 1) - constraints

    def independent_loops(self) -> list[list[str]]:
        """Return a fundamental cycle basis, each cycle as a list of joint names.

        Built from a spanning tree: every non-tree edge closes exactly one
        independent loop. A four-bar returns one loop of four joints.
        """
        parent: dict[str, tuple[str, str] | None] = {}
        visited: set[str] = set()
        tree_edges: set[str] = set()

        for root in self.bodies:
            if root in visited:
                continue
            parent[root] = None
            visited.add(root)
            stack = [root]
            while stack:
                body = stack.pop()
                for jname in self.joints_of(body):
                    if jname in tree_edges:
                        continue
                    nxt = self.joints[jname].other(body)
                    if nxt not in visited:
                        visited.add(nxt)
                        parent[nxt] = (body, jname)
                        tree_edges.add(jname)
                        stack.append(nxt)

        loops: list[list[str]] = []
        for jname, joint in self.joints.items():
            if jname in tree_edges:
                continue
            first, second = joint.bodies
            if first == second:
                continue
            bodies_a, joints_a = self._path_to_root(first, parent)
            bodies_b, joints_b = self._path_to_root(second, parent)
            common = set(bodies_a) & set(bodies_b)
            if not common:
                continue
            # Trim both root-paths at their first shared body.
            cut_a = next(i for i, body in enumerate(bodies_a) if body in common)
            cut_b = next(i for i, body in enumerate(bodies_b) if body in common)
            loops.append([jname, *joints_a[:cut_a], *reversed(joints_b[:cut_b])])
        return loops

    @staticmethod
    def _path_to_root(
        body: str, parent: dict[str, tuple[str, str] | None]
    ) -> tuple[list[str], list[str]]:
        """Return ``(bodies, joints)`` walking from ``body`` up to its tree root."""
        bodies = [body]
        joints: list[str] = []
        current = body
        while parent.get(current) is not None:
            up, jname = parent[current]  # type: ignore[misc]
            joints.append(jname)
            bodies.append(up)
            current = up
        return bodies, joints

    # ---------------------------------------------------------------- geometry

    def link_length(self, body: str) -> float:
        """Distance in mm between the first two joints of ``body``.

        For a binary link this is *the* link length.
        """
        jnames = self.joints_of(body)
        if len(jnames) < 2:
            raise LinkageError(f"body {body!r} has fewer than two joints")
        start = self.joints[jnames[0]].position_mm
        end = self.joints[jnames[1]].position_mm
        return float(np.linalg.norm(end - start))

    def body_frame(
        self, body: str, joint_positions: dict[str, FloatArray] | None = None
    ) -> tuple[FloatArray, float]:
        """Return ``(origin_mm, angle_rad)`` of a body's local frame.

        The frame origin sits at the body's first joint and its x axis points at
        the body's second joint. Ground uses the world frame, so ground-fixed
        geometry keeps its world coordinates.
        """
        if self.bodies[body].is_ground:
            return np.zeros(2), 0.0
        positions = joint_positions or {n: j.position_mm for n, j in self.joints.items()}
        jnames = self.joints_of(body)
        if len(jnames) < 2:
            raise LinkageError(f"body {body!r} has fewer than two joints; no frame is defined")
        origin = np.asarray(positions[jnames[0]], dtype=float)
        along = np.asarray(positions[jnames[1]], dtype=float) - origin
        return origin, angle_of(along)

    def to_local(self, body: str, point_mm: Sequence[float] | FloatArray) -> FloatArray:
        """Express a world point (reference configuration) in a body's local frame."""
        origin, angle = self.body_frame(body)
        return rotation_matrix(-angle) @ (np.asarray(point_mm, dtype=float) - origin)

    def to_world(
        self,
        body: str,
        local_mm: Sequence[float] | FloatArray,
        joint_positions: dict[str, FloatArray],
    ) -> FloatArray:
        """Map a body-local point into world coordinates for a solved configuration."""
        origin, angle = self.body_frame(body, joint_positions)
        return origin + rotation_matrix(angle) @ np.asarray(local_mm, dtype=float)

    # -------------------------------------------------------------- validation

    def validate(self) -> None:
        """Check internal consistency, raising :class:`LinkageError` on any problem."""
        problems: list[str] = []

        for jname, joint in self.joints.items():
            if jname != joint.name:
                problems.append(f"joint key {jname!r} != joint.name {joint.name!r}")
            for body in joint.bodies:
                if body not in self.bodies:
                    problems.append(f"joint {jname!r} references unknown body {body!r}")
            if joint.bodies[0] == joint.bodies[1]:
                problems.append(f"joint {jname!r} connects body {joint.bodies[0]!r} to itself")
            if not np.all(np.isfinite(joint.position_mm)):
                problems.append(f"joint {jname!r} has a non-finite position")
            if joint.kind not in PLANAR_CONSTRAINTS:
                problems.append(f"joint {jname!r} has unknown kind {joint.kind!r}")

        for oname, out in self.outputs.items():
            if out.body not in self.bodies:
                problems.append(f"output {oname!r} references unknown body {out.body!r}")

        if len(self.ground_bodies) != 1:
            problems.append(f"expected exactly one ground body, found {self.ground_bodies}")

        for bname in self.bodies:
            if not self.joints_of(bname):
                problems.append(f"body {bname!r} has no joints")

        if not self.input_joint:
            problems.append("input_joint is not set")
        elif self.input_joint not in self.joints:
            problems.append(f"input_joint {self.input_joint!r} is not a joint")
        else:
            joint = self.joints[self.input_joint]
            grounded = [b for b in joint.bodies if b in self.bodies and self.bodies[b].is_ground]
            if len(grounded) != 1:
                problems.append(
                    f"input_joint {self.input_joint!r} must connect ground to one moving body"
                )

        if self.input_range_deg is not None:
            lo, hi = self.input_range_deg
            if not np.isfinite([lo, hi]).all():
                problems.append("input_range_deg is not finite")
            elif lo == hi:
                problems.append("input_range_deg is empty")

        if self.bodies and not self._is_connected():
            problems.append("linkage graph is not connected")

        if problems:
            raise LinkageError(f"invalid linkage {self.name!r}:\n  - " + "\n  - ".join(problems))

    def _is_connected(self) -> bool:
        start = next(iter(self.bodies))
        seen = {start}
        stack = [start]
        while stack:
            body = stack.pop()
            for nxt in self.neighbours(body):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        return len(seen) == len(self.bodies)

    # ---------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description matching the sample schema."""
        return {
            "name": self.name,
            "bodies": [{"name": b.name, "is_ground": b.is_ground} for b in self.bodies.values()],
            "joints": {
                j.name: {"bodies": list(j.bodies), "xy_mm": list(j.xy), "kind": j.kind}
                for j in self.joints.values()
            },
            "outputs": {
                o.name: {"body": o.body, "xy_mm": list(o.xy)} for o in self.outputs.values()
            },
            "input_joint": self.input_joint,
            "input_range_deg": list(self.input_range_deg) if self.input_range_deg else None,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Linkage:
        """Build a linkage from the dictionary produced by :meth:`to_dict`."""
        bodies = {
            b["name"]: Body(name=b["name"], is_ground=bool(b.get("is_ground", False)))
            for b in data["bodies"]
        }
        joints = {
            name: Joint(
                name=name,
                bodies=(spec["bodies"][0], spec["bodies"][1]),
                xy=(float(spec["xy_mm"][0]), float(spec["xy_mm"][1])),
                kind=spec.get("kind", "revolute"),
            )
            for name, spec in data["joints"].items()
        }
        outputs = {
            name: OutputPoint(
                name=name,
                body=spec["body"],
                xy=(float(spec["xy_mm"][0]), float(spec["xy_mm"][1])),
            )
            for name, spec in data.get("outputs", {}).items()
        }
        rng = data.get("input_range_deg")
        linkage = cls(
            bodies=bodies,
            joints=joints,
            outputs=outputs,
            input_joint=data.get("input_joint", ""),
            input_range_deg=(float(rng[0]), float(rng[1])) if rng else None,
            name=data.get("name", "linkage"),
            meta=data.get("meta", {}),
        )
        linkage.validate()
        return linkage

    @classmethod
    def from_json(cls, path: str | Path) -> Linkage:
        """Load a linkage from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_json(self, path: str | Path, *, indent: int = 2) -> None:
        """Write this linkage to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=indent) + "\n", encoding="utf-8")

    # --------------------------------------------------------- convenience API

    @classmethod
    def four_bar(
        cls,
        *,
        ground_mm: float,
        input_mm: float,
        coupler_mm: float,
        output_mm: float,
        coupler_point_mm: Sequence[float] | None = None,
        input_angle_deg: float = 60.0,
        input_range_deg: tuple[float, float] | None = None,
        branch: int = 1,
        name: str = "four_bar",
    ) -> Linkage:
        """Build a four-bar in the canonical frame, assembled at ``input_angle_deg``.

        Ground joint ``A`` sits at the origin and ``D`` on the +x axis, so the
        reference configuration is fully determined by the four lengths, the
        input angle and the assembly branch.

        Parameters
        ----------
        ground_mm, input_mm, coupler_mm, output_mm
            Lengths of links AD, AB, BC and CD.
        coupler_point_mm
            World position of the tracked coupler point in the reference
            configuration. Defaults to the midpoint of BC.
        branch
            Assembly mode, ``+1`` or ``-1``; see
            :func:`cmtool.kinematics.fourbar.solve_dyad`.

        Notes
        -----
        This constructor exists for convenience and tests. It is the only place
        outside :mod:`cmtool.kinematics.fourbar` that assumes a four-bar.
        """
        from cmtool.kinematics.fourbar import solve_dyad

        theta2 = np.radians(input_angle_deg)
        pos_a = np.zeros(2)
        pos_d = np.array([ground_mm, 0.0])
        pos_b = pos_a + input_mm * np.array([np.cos(theta2), np.sin(theta2)])
        pos_c = solve_dyad(pos_b, coupler_mm, pos_d, output_mm, branch=branch)
        if pos_c is None:
            raise LinkageError(
                f"four-bar does not assemble at input_angle_deg={input_angle_deg} "
                f"with lengths ground={ground_mm}, input={input_mm}, "
                f"coupler={coupler_mm}, output={output_mm}"
            )

        bodies = {
            "ground": Body("ground", is_ground=True),
            "input": Body("input"),
            "coupler": Body("coupler"),
            "output": Body("output"),
        }
        joints = {
            "A": Joint("A", ("ground", "input"), (float(pos_a[0]), float(pos_a[1]))),
            "B": Joint("B", ("input", "coupler"), (float(pos_b[0]), float(pos_b[1]))),
            "C": Joint("C", ("coupler", "output"), (float(pos_c[0]), float(pos_c[1]))),
            "D": Joint("D", ("output", "ground"), (float(pos_d[0]), float(pos_d[1]))),
        }
        if coupler_point_mm is not None:
            point = np.asarray(coupler_point_mm, dtype=float)
        else:
            point = (pos_b + pos_c) / 2.0
        outputs = {"P": OutputPoint("P", "coupler", (float(point[0]), float(point[1])))}

        linkage = cls(
            bodies=bodies,
            joints=joints,
            outputs=outputs,
            input_joint="A",
            input_range_deg=input_range_deg,
            name=name,
            meta={"assembly_branch": branch, "reference_input_deg": float(input_angle_deg)},
        )
        linkage.validate()
        return linkage


def link_lengths(linkage: Linkage, bodies: Iterable[str] | None = None) -> dict[str, float]:
    """Return the first-two-joint distance for each named body."""
    if bodies is not None:
        names = list(bodies)
    else:
        names = [b for b in linkage.bodies if len(linkage.joints_of(b)) >= 2]
    return {b: linkage.link_length(b) for b in names}
