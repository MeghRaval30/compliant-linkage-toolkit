r"""Small-length flexural pivot (SLFP).

A short, thin, prismatic strip joining two comparatively rigid segments. It is
the first flexure type implemented because it is the simplest to print, the
simplest to model, and the one whose pseudo-rigid-body model is least ambiguous.

Model
-----
The strip is short enough that essentially all compliance is in it, so it behaves
as a pin joint at the **centre of the flexure** with a torsional spring:

.. math::

    K = \frac{E I}{L}, \qquad I = \frac{w t^3}{12}

Bending it through ``theta`` over length ``L`` gives a radius of curvature
``rho = L / theta``, and the peak surface strain of a strip of thickness ``t``
bent to radius ``rho`` is ``t / (2 rho)``:

.. math::

    \varepsilon = \frac{t \theta}{2 L}

Rearranged, this gives the two numbers that drive design:

* the largest bend a given flexure can take, ``theta_max = 2 L eps_allow / t``
* the shortest flexure that can take a required bend,
  ``L_min = t theta / (2 eps_allow)``

Limits
------
* Uniform curvature is assumed. Real flexures under combined load do not bend in
  a perfect arc; the beam FEA (A4) does not make this assumption, and the
  PRBM-versus-FEA disagreement metric partly measures this error.
* Axial stress from link loads is **not** included here. It is reported
  separately rather than folded in, so the two contributions stay visible.
* ``K = EI/L`` needs the flexure to be short relative to its links and the links
  to be genuinely rigid. Those ratios are checked, not assumed, by
  :func:`prbm_validity`.
"""

from __future__ import annotations

import numpy as np

from cmtool.flexures.base import (
    FLEXURES,
    FlexureGeometry,
    PrbmValidity,
    StrainEstimate,
)

#: Name of the strain model, recorded in every sample.
STRAIN_MODEL = "leaf_uniform_bending"

#: Default upper bound on flexure length divided by the shorter adjacent link.
#: The usual rule of thumb for "small-length"; overridable from config.
DEFAULT_LENGTH_RATIO_LIMIT = 0.1


class SmallLengthFlexuralPivot:
    """Prismatic leaf flexure, modelled as a pin at its centre plus a torsion spring."""

    name = "small_length_pivot"

    def second_moment_mm4(self, geometry: FlexureGeometry) -> float:
        """In-plane second moment ``I = w t^3 / 12``, in mm^4."""
        return geometry.width_mm * geometry.thickness_mm**3 / 12.0

    def out_of_plane_second_moment_mm4(self, geometry: FlexureGeometry) -> float:
        """Out-of-plane second moment ``I = t w^3 / 12``, in mm^4.

        Used for the gravity sag check, not for the mechanism's motion. The ratio
        to the in-plane value is ``(w / t)^2``.
        """
        return geometry.thickness_mm * geometry.width_mm**3 / 12.0

    def stiffness_nmm_per_rad(self, geometry: FlexureGeometry, youngs_modulus_mpa: float) -> float:
        """Torsional spring constant ``K = E I / L``, in N*mm/rad."""
        if youngs_modulus_mpa <= 0.0:
            raise ValueError("Young's modulus must be positive")
        return youngs_modulus_mpa * self.second_moment_mm4(geometry) / geometry.length_mm

    def peak_strain(self, geometry: FlexureGeometry, bend_angle_rad: float) -> StrainEstimate:
        """Peak bending strain ``eps = t |theta| / (2 L)``."""
        strain = geometry.thickness_mm * abs(float(bend_angle_rad)) / (2.0 * geometry.length_mm)
        return StrainEstimate(
            peak_strain=strain,
            model=STRAIN_MODEL,
            bend_angle_rad=float(bend_angle_rad),
            includes_axial=False,
            includes_stress_concentration=False,
        )

    def max_bend_angle_rad(self, geometry: FlexureGeometry, allowable_strain: float) -> float:
        """Largest bend angle within the allowable strain: ``2 L eps / t``."""
        if allowable_strain <= 0.0:
            raise ValueError("allowable strain must be positive")
        return 2.0 * geometry.length_mm * allowable_strain / geometry.thickness_mm

    def min_length_mm(
        self, thickness_mm: float, bend_angle_rad: float, allowable_strain: float
    ) -> float:
        """Shortest flexure that takes ``bend_angle_rad`` within the allowable strain.

        ``L_min = t |theta| / (2 eps_allow)``. This is the inequality that makes
        large joint excursions expensive: a flexure long enough to survive them
        eventually stops being "small-length" at all, which is what
        :func:`prbm_validity` catches.
        """
        if thickness_mm <= 0.0:
            raise ValueError("thickness must be positive")
        if allowable_strain <= 0.0:
            raise ValueError("allowable strain must be positive")
        return thickness_mm * abs(float(bend_angle_rad)) / (2.0 * allowable_strain)

    def characteristic_pivot_fraction(self) -> float:
        """Return 0.5: the PRBM pivot sits at the centre of the flexure."""
        return 0.5

    def parasitic_axial_stiffness_n_per_mm(
        self, geometry: FlexureGeometry, youngs_modulus_mpa: float
    ) -> float:
        """Axial stiffness ``E A / L`` of the strip, in N/mm.

        A real flexure stretches as well as bending. This is not used by the PRBM
        (which treats the pivot as rigid in translation); it is provided so the
        parasitic motion can be estimated and compared against the beam FEA.
        """
        area = geometry.thickness_mm * geometry.width_mm
        return youngs_modulus_mpa * area / geometry.length_mm


def prbm_validity(
    geometry: FlexureGeometry,
    shortest_adjacent_link_mm: float,
    bend_angle_rad: float,
    *,
    length_ratio_limit: float = DEFAULT_LENGTH_RATIO_LIMIT,
) -> PrbmValidity:
    """Report how far inside the PRBM envelope this flexure sits.

    Parameters
    ----------
    geometry
        The flexure.
    shortest_adjacent_link_mm
        Length of the shorter of the two links the flexure joins. The ratio
        ``L_flexure / L_link`` is what "small-length" means.
    bend_angle_rad
        Largest bend from the unstressed configuration.
    length_ratio_limit
        Upper bound on that ratio; the usual rule of thumb is 0.1.
    """
    if shortest_adjacent_link_mm <= 0.0:
        raise ValueError("adjacent link length must be positive")
    return PrbmValidity(
        length_ratio=geometry.length_mm / shortest_adjacent_link_mm,
        length_ratio_limit=length_ratio_limit,
        bend_angle_deg=float(np.degrees(abs(bend_angle_rad))),
        slenderness=geometry.slenderness,
    )


FLEXURES.add("small_length_pivot", SmallLengthFlexuralPivot())
