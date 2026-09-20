"""Exact reference solutions for large-deflection cantilevers.

These are the yardsticks milestone A4's beam FEA is measured against. Neither is
a stored number: both are computed here from the governing equations, so the
comparison is against physics rather than against a previous run of our own code.

End moment
----------
A cantilever under a pure end moment has constant curvature, so it bends into a
circular arc and the tip position is elementary:

``kappa = M / (EI)``, ``theta = kappa L``, tip at ``(sin(theta), 1 - cos(theta)) L / theta``.

End load
--------
A cantilever under a transverse tip force has no elementary solution. From
``EI theta'' + P cos(theta) = 0`` with ``theta'(L) = 0``, one integration gives

``theta' = sqrt(2P/EI) sqrt(sin(theta0) - sin(theta))``

with ``theta0`` the tip slope, and the arc-length and coordinate integrals follow.
The substitution ``sin(theta) = sin(theta0)(1 - u^2)`` removes the square-root
singularity at the tip, leaving integrals that quadrature handles to machine
precision. This is the classical Bisshopp-Drucker solution; writing it as
quadrature rather than in elliptic integrals avoids depending on a remembered
formula, and one of the integrals collapses to a closed form on the way:

``x_tip / L = 2 sqrt(sin(theta0)) / I1``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq

from cmtool.core.units import FloatArray


@dataclass(frozen=True)
class CantileverState:
    """Tip state of a deflected cantilever, normalised by its length."""

    tip_x_over_l: float
    tip_y_over_l: float
    tip_slope_rad: float
    load_parameter: float

    @property
    def tip(self) -> FloatArray:
        """Tip position as a ``(2,)`` array, normalised by length."""
        return np.array([self.tip_x_over_l, self.tip_y_over_l])


def end_moment_tip(length_mm: float, ei_nmm2: float, moment_nmm: float) -> CantileverState:
    """Exact tip state of a cantilever under a pure end moment.

    Constant curvature, so the deflected shape is a circular arc of radius
    ``EI / M``.
    """
    if length_mm <= 0.0 or ei_nmm2 <= 0.0:
        raise ValueError("length and EI must be positive")
    theta = moment_nmm * length_mm / ei_nmm2
    if abs(theta) < 1e-12:
        return CantileverState(1.0, 0.0, 0.0, moment_nmm * length_mm / ei_nmm2)
    return CantileverState(
        tip_x_over_l=float(np.sin(theta) / theta),
        tip_y_over_l=float((1.0 - np.cos(theta)) / theta),
        tip_slope_rad=float(theta),
        load_parameter=float(theta),
    )


def arc_shape(theta: float, samples: int = 50) -> FloatArray:
    """Sample the exact circular-arc shape, normalised by length.

    Returns an ``(n, 2)`` array from the clamped root to the tip.
    """
    fractions = np.linspace(0.0, 1.0, samples)
    if abs(theta) < 1e-12:
        return np.column_stack([fractions, np.zeros_like(fractions)])
    angles = theta * fractions
    return np.column_stack([np.sin(angles) / theta, (1.0 - np.cos(angles)) / theta])


def _arc_length_integral(theta0: float) -> float:
    """``I1(theta0)``: the normalised arc-length integral for end-load bending."""
    sin0 = np.sin(theta0)

    def integrand(u: float) -> float:
        sin_theta = sin0 * (1.0 - u * u)
        return 2.0 * np.sqrt(sin0) / np.sqrt(max(1.0 - sin_theta**2, 1e-16))

    value, _ = quad(integrand, 0.0, 1.0, limit=200)
    return float(value)


def _vertical_integral(theta0: float) -> float:
    """Return the normalised vertical-coordinate integral for end-load bending."""
    sin0 = np.sin(theta0)

    def integrand(u: float) -> float:
        sin_theta = sin0 * (1.0 - u * u)
        return 2.0 * np.sqrt(sin0) * sin_theta / np.sqrt(max(1.0 - sin_theta**2, 1e-16))

    value, _ = quad(integrand, 0.0, 1.0, limit=200)
    return float(value)


def end_load_tip(alpha: float, *, max_slope_deg: float = 88.0) -> CantileverState:
    """Exact tip state of a cantilever under a transverse tip force.

    Parameters
    ----------
    alpha
        Dimensionless load ``P L^2 / (E I)``.
    max_slope_deg
        Upper bound for the tip-slope search. The formulation used here assumes
        the slope stays below 90 degrees, which covers the whole range any
        flexure in this project reaches.

    Notes
    -----
    In the small-load limit this reduces to the linear cantilever,
    ``tip_y / L -> alpha / 3``, which is one of the checks in the test suite.
    """
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative")
    if alpha == 0.0:
        return CantileverState(1.0, 0.0, 0.0, 0.0)

    def residual(theta0: float) -> float:
        return _arc_length_integral(theta0) ** 2 / 2.0 - alpha

    upper = float(np.radians(max_slope_deg))
    if residual(upper) < 0.0:
        raise ValueError(
            f"alpha={alpha} implies a tip slope beyond {max_slope_deg} degrees, outside "
            "the range this reference covers"
        )
    theta0 = brentq(residual, 1e-9, upper, xtol=1e-14, rtol=1e-14)

    integral = _arc_length_integral(theta0)
    return CantileverState(
        tip_x_over_l=float(2.0 * np.sqrt(np.sin(theta0)) / integral),
        tip_y_over_l=float(_vertical_integral(theta0) / integral),
        tip_slope_rad=float(theta0),
        load_parameter=float(alpha),
    )
