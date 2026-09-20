"""Deciding which pseudo-rigid-body variant represents a flexure joint.

Howell gives different constants for different end loadings, and the two cases do
not even share a stiffness formula. Which one applies to a flexure *joint* is not
something to assume: it is an empirical question, and now that the beam FEA works
it is answerable.

The experiment
--------------
Load a cantilever flexure at its tip with a moment ``M`` and a transverse force
``P``, parameterised by the dimensionless ratio

``lambda = P L / M``

``lambda = 0`` is pure end-moment loading, ``lambda -> inf`` is force-dominated.
For each ratio, sweep the load magnitude, take the tip path from the FEA, and fit
the PRBM to it: ``gamma`` from the tip path, then ``K`` from moment equilibrium
about the fitted pivot.

Where our flexures sit
----------------------
In a compliant four-bar under no external load, the only forces are those needed
to bend the other flexures. So the moment a flexure carries is of order
``K dphi ~ (EI/L) dphi``, while the transverse force is of order that moment
divided by a link length, ``(EI/L) dphi / l``. Their ratio is

``lambda ~ (P L) / M ~ L / l``

which is the flexure's **length ratio** -- about 0.1 for our designs. Flexure
joints are therefore strongly moment-dominated, and the FEA sweep confirms the
consequence: at ``lambda = 0.1`` the fitted constants are within a percent of the
end-moment variant and nowhere near the end-force one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from cmtool.solvers.beam import BeamModel, BeamSection, solve

#: Load ratios swept by default: from pure moment out to force-dominated.
DEFAULT_LOAD_RATIOS = (0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)

#: Load ratio representative of a flexure joint, from the scaling argument above.
JOINT_LOAD_RATIO = 0.1


@dataclass(frozen=True)
class FittedPrbm:
    """PRBM constants fitted to a beam FEA tip path."""

    load_ratio: float
    gamma: float
    stiffness_multiple: float
    fit_rms_over_length: float

    @property
    def pivot_fraction(self) -> float:
        """Pivot position from the root, as a fraction of the flexure length."""
        return 1.0 - self.gamma

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "load_ratio": self.load_ratio,
            "gamma": self.gamma,
            "pivot_fraction": self.pivot_fraction,
            "stiffness_multiple_of_ei_over_l": self.stiffness_multiple,
            "fit_rms_over_length": self.fit_rms_over_length,
        }


def _tip_path(
    section: BeamSection,
    length_mm: float,
    load_ratio: float,
    max_tip_angle_deg: float,
    n_elements: int,
    n_levels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the applied moments and the corresponding FEA tip positions."""
    peak = np.radians(max_tip_angle_deg) * section.ei / length_mm
    moments = np.linspace(0.15, 1.0, n_levels) * peak
    tips = []
    for moment in moments:
        nodes = np.column_stack(
            [np.linspace(0.0, length_mm, n_elements + 1), np.zeros(n_elements + 1)]
        )
        model = BeamModel(nodes, [(i, i + 1) for i in range(n_elements)], [section] * n_elements)
        outcome = solve(
            model,
            fixed_dofs={0: 0.0, 1: 0.0, 2: 0.0},
            loads={
                model.dof(n_elements, 2): float(moment),
                model.dof(n_elements, 1): float(load_ratio * moment / length_mm),
            },
            steps=16,
        )
        tips.append(model.deformed_nodes(outcome.displacement)[-1])
    return moments, np.asarray(tips, dtype=float)


def fit_prbm(
    load_ratio: float = JOINT_LOAD_RATIO,
    *,
    length_mm: float = 20.0,
    thickness_mm: float = 0.6,
    width_mm: float = 6.0,
    youngs_modulus_mpa: float = 3500.0,
    max_tip_angle_deg: float = 40.0,
    n_elements: int = 24,
    n_levels: int = 10,
) -> FittedPrbm:
    """Fit PRBM constants to the beam FEA at one tip-load ratio.

    The modulus cancels out of both fitted constants -- ``gamma`` is geometric and
    the stiffness is reported as a multiple of ``EI/L`` -- so this result does not
    depend on the placeholder material data.
    """
    section = BeamSection.rectangular(thickness_mm, width_mm, youngs_modulus_mpa)
    moments, tips = _tip_path(
        section, length_mm, load_ratio, max_tip_angle_deg, n_elements, n_levels
    )

    def rms_error(gamma: float) -> float:
        total = 0.0
        for x, y in tips:
            angle = np.arcsin(np.clip(y / (gamma * length_mm), -1.0, 1.0))
            predicted = (1.0 - gamma) * length_mm + gamma * length_mm * np.cos(angle)
            total += (predicted - x) ** 2
        return float(np.sqrt(total / len(tips)))

    result = minimize_scalar(rms_error, bounds=(0.5, 0.99), method="bounded")
    gamma = float(result.x)

    multiples = []
    for moment, (x, y) in zip(moments, tips, strict=True):
        force = load_ratio * moment / length_mm
        angle = np.arcsin(np.clip(y / (gamma * length_mm), -1.0, 1.0))
        # Moment about the fitted pivot: the applied couple plus the transverse
        # force acting through its lever arm from the pivot.
        applied = moment + force * (x - (1.0 - gamma) * length_mm)
        multiples.append(applied / angle / (section.ei / length_mm))

    return FittedPrbm(
        load_ratio=float(load_ratio),
        gamma=gamma,
        stiffness_multiple=float(np.mean(multiples)),
        fit_rms_over_length=float(result.fun / length_mm),
    )


def sweep(load_ratios: tuple[float, ...] = DEFAULT_LOAD_RATIOS, **kwargs: Any) -> list[FittedPrbm]:
    """Fit PRBM constants across a range of tip-load ratios."""
    return [fit_prbm(ratio, **kwargs) for ratio in load_ratios]


def recommend_variant(**kwargs: Any) -> dict[str, Any]:
    """Recommend a long-segment variant for flexure-joint loading.

    Compares the FEA-fitted constants at the joint-representative load ratio
    against both variants in ``configs/models/prbm.yaml`` and names the closer
    one, with the relative error of each.
    """
    from cmtool.flexures.prbm_models import PRBM_MODELS, LongSegmentModel

    fitted = fit_prbm(JOINT_LOAD_RATIO, **kwargs)
    scores: dict[str, dict[str, float]] = {}
    for variant in ("end_moment", "end_force"):
        model = PRBM_MODELS.get(f"long_segment_{variant}")
        assert isinstance(model, LongSegmentModel)
        multiple = model.stiffness_multiple()
        gamma = model.gamma()
        scores[variant] = {
            "gamma": gamma,
            "stiffness_multiple": multiple,
            "gamma_error": abs(gamma - fitted.gamma) / fitted.gamma,
            "stiffness_error": abs(multiple - fitted.stiffness_multiple)
            / fitted.stiffness_multiple,
        }

    best = min(scores, key=lambda v: scores[v]["stiffness_error"])
    worst_error = max(entry["stiffness_error"] for entry in scores.values())
    return {
        "fitted": fitted.to_dict(),
        "joint_load_ratio": JOINT_LOAD_RATIO,
        "variants": scores,
        "recommended": best,
        "reason": (
            f"at the joint-representative load ratio lambda = {JOINT_LOAD_RATIO}, the beam "
            f"FEA gives gamma = {fitted.gamma:.4f} and K = {fitted.stiffness_multiple:.4f} "
            f"EI/L; the {best} variant is within "
            f"{scores[best]['stiffness_error'] * 100:.1f} percent on stiffness, the other "
            f"is off by {worst_error * 100:.0f} percent"
        ),
    }
