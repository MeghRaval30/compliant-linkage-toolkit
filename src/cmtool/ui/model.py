"""What the UI sends, and what it gets back.

The UI is a thin shell over the same API the CLI uses: a set of design
parameters in, a :class:`~cmtool.viz.scene.ViewerScene` out. Keeping that
boundary narrow is what stops the UI becoming a second implementation of the
toolkit -- every number it shows came from ``convert``, ``simulate`` and
``build_scene``, not from anything written here.

Everything in this module is pure: no server, no threads, no I/O beyond reading
configs. That makes the solve path testable without starting anything.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass, field
from typing import Any

from cmtool.core.graph import Linkage
from cmtool.core.provenance import canonical_hash

#: Presets offered in the UI. The three pilots plus the demo pair, so a live
#: audience can be shown a design that has actually been printed.
PRESETS: dict[str, str] = {
    "demo_pair": "examples/designs/demo_pair.json",
    "fb_02_0052": "examples/designs/fb_02_0052.json",
    "fb_02_0090": "examples/designs/fb_02_0090.json",
    "fb_02_0203": "examples/designs/fb_02_0203.json",
}


@dataclass
class DesignParams:
    """Everything the UI can change about a design.

    Deliberately flat and JSON-shaped: it is the thing hashed for the cache, the
    thing saved by "Save design JSON", and the thing a URL could carry.
    """

    ground_mm: float = 40.0
    input_mm: float = 32.0
    coupler_mm: float = 44.0
    output_mm: float = 36.0
    coupler_x_mm: float = 22.0
    coupler_y_mm: float = 38.0
    input_angle_deg: float = 90.0
    arc_start_deg: float | None = None
    arc_end_deg: float | None = None
    #: When no arc is given, shrink one to this worst-joint excursion.
    target_excursion_deg: float = 22.0
    flexure_type: str = "small_length_pivot"
    thickness_mm: float | None = None
    material: str = "PLA"
    printer: str = "kobra2_neo"
    placement: str = "pivot_matched"
    unstressed_at: str = "mid_arc"
    n_steps: int = 31
    solvers: list[str] = field(default_factory=lambda: ["rigid", "prbm", "fea"])
    name: str = "ui_design"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable form."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DesignParams:
        """Build from a possibly-partial dictionary, ignoring unknown keys.

        Unknown keys are dropped rather than raising: a design JSON saved by a
        later version should still open, minus whatever it gained.
        """
        fields = {f for f in cls.__dataclass_fields__}
        clean = {k: v for k, v in data.items() if k in fields}
        params = cls(**clean)
        params.validate()
        return params

    @property
    def cache_key(self) -> str:
        """Stable digest of every parameter that changes the answer."""
        return canonical_hash(self.to_dict())

    def validate(self) -> None:
        """Reject a design that cannot be solved, with a message a user can act on."""
        problems: list[str] = []
        for name in ("ground_mm", "input_mm", "coupler_mm", "output_mm"):
            if getattr(self, name) <= 0.0:
                problems.append(f"{name} must be positive")
        if self.n_steps < 2:
            problems.append("n_steps must be at least 2")
        if self.n_steps > 121:
            problems.append("n_steps above 121 makes the beam FEA too slow to be interactive")
        if self.thickness_mm is not None and self.thickness_mm <= 0.0:
            problems.append("thickness_mm must be positive")
        if (self.arc_start_deg is None) != (self.arc_end_deg is None):
            problems.append("give both arc ends or neither")
        if (
            self.arc_start_deg is not None
            and self.arc_end_deg is not None
            and abs(self.arc_end_deg - self.arc_start_deg) < 1e-6
        ):
            problems.append("the input arc has zero width")
        unknown = set(self.solvers) - {"rigid", "prbm", "fea"}
        if unknown:
            problems.append(f"unknown solvers {sorted(unknown)}")
        if not self.solvers:
            problems.append("choose at least one solver")

        # The four-bar inequality: without it there is no assembly to solve.
        lengths = sorted([self.ground_mm, self.input_mm, self.coupler_mm, self.output_mm])
        if lengths[3] >= lengths[0] + lengths[1] + lengths[2]:
            problems.append(
                f"the longest link ({lengths[3]:g} mm) is longer than the other three "
                f"together ({lengths[0] + lengths[1] + lengths[2]:g} mm), so the loop "
                "cannot close"
            )
        if problems:
            raise ValueError("; ".join(problems))

    def to_linkage(self) -> Linkage:
        """Build the rigid linkage, with its input arc resolved.

        The arc is fitted when the caller did not give one, using the same
        ``fit_input_arc`` the CLI uses -- there is no separate UI notion of a
        reasonable arc.
        """
        from cmtool.convert.arc import fit_input_arc

        linkage = Linkage.four_bar(
            ground_mm=self.ground_mm,
            input_mm=self.input_mm,
            coupler_mm=self.coupler_mm,
            output_mm=self.output_mm,
            coupler_point_mm=(self.coupler_x_mm, self.coupler_y_mm),
            input_angle_deg=self.input_angle_deg,
            name=self.name,
        )
        if self.arc_start_deg is not None and self.arc_end_deg is not None:
            linkage.input_range_deg = (self.arc_start_deg, self.arc_end_deg)
        else:
            fit = fit_input_arc(linkage, max_joint_excursion_deg=self.target_excursion_deg)
            linkage.input_range_deg = fit.input_range_deg
        return linkage


def solve(params: DesignParams) -> dict[str, Any]:
    """Convert, simulate and assemble everything the UI draws.

    Returns a plain dictionary so the caller can serialise it without knowing
    about the scene types. Warnings are captured rather than printed: a
    placeholder warning belongs on the page, where it cannot be missed, not in
    the terminal the user is not looking at.
    """
    from cmtool.api import convert
    from cmtool.viz.scene import build_scene

    params.validate()
    linkage = params.to_linkage()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mechanism = convert(
            linkage,
            flexures=params.flexure_type,
            material=params.material,
            printer=params.printer,
            input_range_deg=linkage.input_range_deg,
            thickness_mm=params.thickness_mm,
            placement=params.placement,
            unstressed_at=params.unstressed_at,
        )
        scene = build_scene(
            mechanism,
            n_steps=params.n_steps,
            include=tuple(params.solvers),
        )

    feasibility = mechanism.feasibility
    binding = feasibility.binding_joint
    sizing = mechanism.sizing[binding]

    joints = {
        name: {
            "excursion_deg": sized.excursion_deg,
            "max_bend_deg": sized.max_bend_deg,
            "length_mm": sized.geometry.length_mm,
            "thickness_mm": sized.geometry.thickness_mm,
            "min_length_strain_mm": sized.min_length_strain_mm,
            "max_length_geometric_mm": sized.max_length_geometric_mm,
            "utilisation": sized.utilisation,
            "peak_strain": sized.strain.peak_strain,
            "feasible": sized.feasible,
            "limit_reason": sized.limit_reason,
            "prbm_model": sized.prbm_model,
            "stiffness_nmm_per_rad": sized.stiffness_nmm_per_rad,
        }
        for name, sized in mechanism.sizing.items()
    }

    return {
        "params": params.to_dict(),
        "cache_key": params.cache_key,
        "scene": scene.to_dict(),
        "feasibility": {
            "feasible": feasibility.feasible,
            "binding_joint": binding,
            "max_utilisation": feasibility.max_utilisation,
            "why": sizing.limit_reason
            or (
                f"joint {binding} is the closest to its limit: it needs a "
                f"{sizing.min_length_strain_mm:.2f} mm flexure for "
                f"{sizing.max_bend_deg:.1f} deg of bend and "
                f"{sizing.max_length_geometric_mm:.2f} mm fits, so it is using "
                f"{sizing.utilisation * 100:.0f}% of what is available"
            ),
            "allowable_strain": feasibility.allowable_strain,
            "strain_safety_factor": feasibility.strain_safety_factor,
            "all_small_length": feasibility.all_small_length,
            "prbm_notes": feasibility.prbm_notes(),
            "reasons": feasibility.reasons(),
            "joints": joints,
        },
        "strain_margin": _strain_margin(scene, feasibility.allowable_strain),
        "arc_deg": list(mechanism.input_range_deg),
        "reference_input_deg": mechanism.reference_input_deg,
        "is_physical": scene.provenance.is_physical,
        "caveat": scene.caveat,
        "placeholders": list(scene.provenance.placeholders_used),
        "warnings": sorted({str(w.message) for w in caught}),
    }


def _strain_margin(scene: Any, allowable: float) -> dict[str, Any]:
    """Worst flexure strain over the arc, and how much of the allowable it uses."""
    fea = scene.models.get("fea")
    if fea is None or not fea.frames:
        return {"available": False, "reason": "no beam FEA in this run"}
    peaks = {
        joint: max(max(frame.flexure_strain[joint]) for frame in fea.frames)
        for joint in scene.joints
    }
    worst = max(peaks, key=lambda j: peaks[j])
    return {
        "available": True,
        "peak_strain": peaks,
        "worst_joint": worst,
        "worst_strain": peaks[worst],
        "allowable_strain": allowable,
        "utilisation": peaks[worst] / allowable if allowable else float("inf"),
        "margin": allowable / peaks[worst] if peaks[worst] else float("inf"),
        "allowable_is_placeholder": scene.allowable_strain_is_placeholder,
    }


def preset_params(name: str) -> DesignParams:
    """Load a preset design into editable parameters.

    Raises
    ------
    KeyError
        If the preset is unknown, naming the ones that exist.
    """
    from pathlib import Path

    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    path = Path(PRESETS[name])
    if not path.exists():
        raise FileNotFoundError(f"preset {name!r} expects {path}, which is not there")

    linkage = Linkage.from_json(path)
    joints = {n: j.position_mm for n, j in linkage.joints.items()}
    import numpy as np

    def length(first: str, second: str) -> float:
        # Rounded: a length reconstructed from stored coordinates comes back as
        # 44.00000000000001, and a field full of float noise is unusable.
        return round(float(np.linalg.norm(joints[second] - joints[first])), 6)

    # World position in the reference configuration, which is what
    # ``Linkage.four_bar`` takes back -- not the coupler-local coordinate.
    output = next(iter(linkage.outputs.values()))
    point = output.position_mm
    arc = linkage.input_range_deg
    return DesignParams(
        ground_mm=length("A", "D"),
        input_mm=length("A", "B"),
        coupler_mm=length("B", "C"),
        output_mm=length("C", "D"),
        coupler_x_mm=round(float(point[0]), 6),
        coupler_y_mm=round(float(point[1]), 6),
        input_angle_deg=float(linkage.meta.get("reference_input_deg", 90.0)),
        arc_start_deg=float(arc[0]) if arc else None,
        arc_end_deg=float(arc[1]) if arc else None,
        name=linkage.name,
    )
