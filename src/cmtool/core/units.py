"""Unit discipline for cmtool.

The whole library works internally in **millimetres and radians**. Degrees appear
only at the public API boundary, where every parameter and field name says so
explicitly (``input_range_deg``, ``joint_excursion_deg``).

Keeping the conversions in one module means a unit bug is a bug in one file
rather than a bug spread across four solvers.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Internal length unit, stated once so downstream code can be explicit about it.
LENGTH_UNIT = "mm"
#: Internal angle unit.
ANGLE_UNIT = "rad"


def deg(radians: float | FloatArray) -> FloatArray:
    """Convert radians to degrees."""
    return np.degrees(np.asarray(radians, dtype=float))


def rad(degrees: float | FloatArray) -> FloatArray:
    """Convert degrees to radians."""
    return np.radians(np.asarray(degrees, dtype=float))


def wrap_to_pi(angle: float | FloatArray) -> FloatArray:
    """Wrap an angle (rad) into the half-open interval ``(-pi, pi]``.

    Used wherever an angle *difference* is formed, so that a sweep crossing the
    branch cut of :func:`numpy.arctan2` does not produce a spurious 2*pi jump.
    """
    a = np.asarray(angle, dtype=float)
    return -(np.remainder(-a + np.pi, 2.0 * np.pi) - np.pi)


def unwrap(angles: FloatArray) -> FloatArray:
    """Remove 2*pi discontinuities along a swept sequence of angles (rad)."""
    return np.unwrap(np.asarray(angles, dtype=float))


def angle_of(vector: FloatArray) -> float:
    """Return the angle (rad) of a 2-vector measured from the +x axis."""
    v = np.asarray(vector, dtype=float)
    return float(np.arctan2(v[1], v[0]))


def rotation_matrix(angle_rad: float) -> FloatArray:
    """Return the 2x2 rotation matrix for ``angle_rad``."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def cross2(a: FloatArray, b: FloatArray) -> float:
    """Return the scalar (z-component of the) cross product of two 2-vectors."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(a[0] * b[1] - a[1] * b[0])
