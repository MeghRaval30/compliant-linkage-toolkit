"""How well we can measure, and whether that is well enough.

The Phase A go/no-go: **is the tracking uncertainty clearly smaller than the
simulation-to-reality gap we are trying to resolve?** If it is not, no amount of
solver work helps, and the measurement setup has to improve before Phase B.

Two ways of asking, and they answer different questions.

**Static jitter** repeats the measurement on a stationary scene. It captures
detector noise and nothing else. It is the optimistic number, and quoting it
alone would overstate the rig.

**Known-motion ground truth** sweeps a rigid bar on a pin through an arc. The
true path is a circle, computable exactly, so the residual against a fitted
circle contains everything jitter misses: lens distortion left over after
calibration, out-of-plane wander, fixture flex, and any error in the marker
layout itself. This is the number that belongs in the paper.

The reference signal
--------------------
What has to be resolved is the disagreement between models and reality, and the
closest thing available before any part is printed is the **PRBM-versus-FEA**
disagreement on the coupler path -- about 0.46 mm mean on the pilot designs. If
the measurement cannot separate two models that differ by that much, it cannot
say anything useful about which is closer to a real part.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from cmtool.core.units import FloatArray

#: Mean PRBM-versus-beam-FEA coupler path disagreement on the pilot designs, mm.
#: Measured with `cmtool simulate` on examples/designs/, milestone A4.
DEFAULT_SIGNAL_MM = 0.46

#: How much smaller than the signal the uncertainty should be to call it usable.
#: Five to one means the two models are separated by five standard deviations;
#: below about three to one, a single measurement cannot tell them apart.
GOOD_RATIO = 5.0
MARGINAL_RATIO = 3.0


@dataclass(frozen=True)
class UncertaintyEstimate:
    """One uncertainty figure and how it was obtained."""

    method: str
    sigma_mm: float
    max_deviation_mm: float
    n_samples: int
    detail: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "method": self.method,
            "sigma_mm": self.sigma_mm,
            "max_deviation_mm": self.max_deviation_mm,
            "n_samples": self.n_samples,
            "detail": self.detail,
        }


def jitter(points_mm: FloatArray) -> UncertaintyEstimate:
    """Return static repeatability: the scatter of a stationary point across frames.

    Optimistic by construction -- it sees detector noise and nothing else.
    """
    points = np.asarray(points_mm, dtype=float).reshape(-1, 2)
    if len(points) < 2:
        raise ValueError("need at least two frames to estimate jitter")
    centre = points.mean(axis=0)
    radial = np.linalg.norm(points - centre, axis=1)
    return UncertaintyEstimate(
        method="static_jitter",
        sigma_mm=float(np.sqrt(np.mean(radial**2))),
        max_deviation_mm=float(np.max(radial)),
        n_samples=len(points),
        detail={
            "centre_mm": [float(centre[0]), float(centre[1])],
            "sigma_x_mm": float(np.std(points[:, 0])),
            "sigma_y_mm": float(np.std(points[:, 1])),
            "caveat": (
                "detector noise only; excludes distortion residual, out-of-plane "
                "motion and fixture flex"
            ),
        },
    )


def fit_circle(points_mm: FloatArray) -> tuple[FloatArray, float, FloatArray]:
    """Least-squares circle through measured points.

    Returns the centre, the radius, and the signed radial residual of each point.
    """
    points = np.asarray(points_mm, dtype=float).reshape(-1, 2)
    if len(points) < 3:
        raise ValueError("need at least three points to fit a circle")

    guess = np.array([points[:, 0].mean(), points[:, 1].mean(), float(np.ptp(points) / 2.0 or 1.0)])

    def residual(params: FloatArray) -> FloatArray:
        centre = params[:2]
        return np.linalg.norm(points - centre, axis=1) - params[2]

    solution = least_squares(residual, guess, method="lm")
    centre = np.asarray(solution.x[:2], dtype=float)
    radius = float(solution.x[2])
    return centre, radius, np.asarray(residual(solution.x), dtype=float)


def known_motion(
    points_mm: FloatArray, *, expected_radius_mm: float | None = None
) -> UncertaintyEstimate:
    """Ground-truth check against a rigid bar swept about a pin.

    The true path is a circle, so the residual against a fitted circle bounds
    everything the measurement gets wrong -- distortion residual, out-of-plane
    wander, fixture flex, layout error -- not just detector noise.

    Parameters
    ----------
    expected_radius_mm
        If the bar's radius was measured with calipers, pass it: the difference
        between it and the fitted radius is a **scale** error, which a circle fit
        alone cannot see because a wrong scale still fits a circle perfectly.
    """
    centre, radius, residual = fit_circle(points_mm)
    scatter = float(np.sqrt(np.mean(residual**2)))
    detail: dict[str, Any] = {
        "fitted_centre_mm": [float(centre[0]), float(centre[1])],
        "fitted_radius_mm": radius,
        "shape_residual_mm": scatter,
        "captures": (
            "distortion residual, out-of-plane motion, fixture flex and layout error, "
            "as well as detector noise"
        ),
        "limitation": (
            "the circle residual is a SHAPE check, not an absolute-position one: a "
            "uniform offset or scale error still fits a circle perfectly, so it is "
            "invisible here"
        ),
    }

    sigma = scatter
    if expected_radius_mm is not None:
        error = radius - float(expected_radius_mm)
        detail["expected_radius_mm"] = float(expected_radius_mm)
        detail["radius_error_mm"] = float(error)
        detail["scale_error_fraction"] = float(error / expected_radius_mm)
        detail["scale_note"] = (
            "a scale error comes from the fiducial spacing or the printed marker size "
            "being wrong; measure a printed marker with calipers"
        )
        # A scale error displaces points by roughly the radius error, so it adds to
        # the scatter rather than being separate from it.
        sigma = float(np.hypot(scatter, error))
        detail["combined_note"] = (
            "sigma_mm combines the shape residual with the radius error in quadrature"
        )

    return UncertaintyEstimate(
        method="known_motion_circle",
        sigma_mm=sigma,
        max_deviation_mm=float(np.max(np.abs(residual))),
        n_samples=len(np.asarray(points_mm).reshape(-1, 2)),
        detail=detail,
    )


@dataclass
class GoNoGoReport:
    """Whether the measurement can resolve the signal it needs to."""

    estimates: list[UncertaintyEstimate]
    signal_mm: float = DEFAULT_SIGNAL_MM
    signal_source: str = "PRBM vs beam FEA coupler path, pilot designs (A4)"

    @property
    def governing(self) -> UncertaintyEstimate:
        """The estimate the verdict rests on: the **largest** available.

        Not the most complete one. The two methods capture different error
        sources and neither contains the other -- a circle fit absorbs a uniform
        offset or scale that static jitter would never see, while jitter ignores
        everything systematic. Taking the smaller would flatter the rig, so the
        verdict takes the worst of what was measured.
        """
        if not self.estimates:
            raise ValueError("no uncertainty estimates were provided")
        return max(self.estimates, key=lambda e: e.sigma_mm)

    @property
    def ratio(self) -> float:
        """Signal divided by uncertainty. Higher is better."""
        sigma = self.governing.sigma_mm
        return float("inf") if sigma <= 0.0 else self.signal_mm / sigma

    @property
    def verdict(self) -> str:
        """``"go"``, ``"marginal"`` or ``"no-go"``."""
        if self.ratio >= GOOD_RATIO:
            return "go"
        if self.ratio >= MARGINAL_RATIO:
            return "marginal"
        return "no-go"

    def explain(self) -> str:
        """One-paragraph statement of the verdict, for the report."""
        estimate = self.governing
        others = ", ".join(
            f"{e.method} {e.sigma_mm:.3f} mm" for e in self.estimates if e is not estimate
        )
        text = (
            f"Tracking uncertainty is {estimate.sigma_mm:.3f} mm, taken as the worst of "
            f"the methods used ({estimate.method}, {estimate.n_samples} samples, worst "
            f"deviation {estimate.max_deviation_mm:.3f} mm"
            + (f"; also measured: {others}" if others else "")
            + f"). The signal to resolve is {self.signal_mm:.3f} mm "
            f"({self.signal_source}), a ratio of {self.ratio:.1f} to 1. "
        )
        if self.verdict == "go":
            return text + (
                "That is enough to separate the models cleanly, so Phase B can proceed "
                "on this measurement setup."
            )
        if self.verdict == "marginal":
            return text + (
                "That is enough to see the difference but not to characterise it well. "
                "Improving the setup -- more resolution, larger markers, better lighting, "
                "a squarer camera -- would pay for itself before scaling up."
            )
        return text + (
            "The measurement cannot separate the models. Fix the setup before Phase B: "
            "no amount of solver work helps when the instrument cannot see the effect."
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "estimates": [e.to_dict() for e in self.estimates],
            "signal_mm": self.signal_mm,
            "signal_source": self.signal_source,
            "governing_method": self.governing.method,
            "uncertainty_mm": self.governing.sigma_mm,
            "signal_to_uncertainty": self.ratio,
            "verdict": self.verdict,
            "thresholds": {"go": GOOD_RATIO, "marginal": MARGINAL_RATIO},
            "explanation": self.explain(),
        }
