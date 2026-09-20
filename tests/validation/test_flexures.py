"""Validation of the small-length flexural pivot against closed-form results.

Every check here is exact arithmetic from beam theory, not a stored output:

* ``I = w t^3 / 12`` and ``K = E I / L`` against hand-evaluated values,
* ``eps = t theta / (2 L)`` against the independent statement of the same
  physics, ``eps = t / (2 rho)`` with ``rho = L / theta``,
* the strain limit and length-sizing functions as exact inverses of each other,
* the design bound in :mod:`cmtool.convert.limits` as the exact elimination of
  ``L`` between the strain and PRBM-validity inequalities.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmtool.convert.limits import (
    max_bend_deg,
    min_link_length_for_excursion_mm,
    required_link_length_mm,
)
from cmtool.flexures import FLEXURES, FlexureGeometry
from cmtool.flexures.prbm_models import PRBM_MODELS
from cmtool.flexures.slfp import SmallLengthFlexuralPivot, prbm_validity

pytestmark = pytest.mark.validation

SLFP = SmallLengthFlexuralPivot()


class TestGeometry:
    def test_registered(self):
        assert FLEXURES.get("small_length_pivot").name == "small_length_pivot"

    @pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
    def test_non_positive_dimensions_refused(self, bad):
        with pytest.raises(ValueError, match="positive and finite"):
            FlexureGeometry(thickness_mm=bad, length_mm=10.0, width_mm=6.0)

    def test_slenderness_and_aspect_ratio(self):
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        assert geom.slenderness == pytest.approx(20.0)
        assert geom.aspect_ratio == pytest.approx(12.0)


class TestSecondMoment:
    def test_matches_hand_value(self):
        """I = w t^3 / 12 = 6 * 0.5^3 / 12 = 0.0625 mm^4."""
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        assert SLFP.second_moment_mm4(geom) == pytest.approx(0.0625, rel=1e-12)

    def test_scales_with_the_cube_of_thickness(self):
        thin = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        thick = FlexureGeometry(thickness_mm=1.0, length_mm=10.0, width_mm=6.0)
        assert SLFP.second_moment_mm4(thick) == pytest.approx(8.0 * SLFP.second_moment_mm4(thin))

    def test_out_of_plane_ratio_is_the_aspect_ratio_squared(self):
        """The claim docs/physics.md section 7 rests on: I_out / I_in = (w/t)^2."""
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=8.0, width_mm=6.0)
        ratio = SLFP.out_of_plane_second_moment_mm4(geom) / SLFP.second_moment_mm4(geom)
        assert ratio == pytest.approx(geom.aspect_ratio**2, rel=1e-12)
        assert ratio == pytest.approx(100.0, rel=1e-12)


class TestStiffness:
    def test_matches_hand_value(self):
        """K = E I / L = 3500 * 0.0625 / 10 = 21.875 N*mm/rad."""
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        assert SLFP.stiffness_nmm_per_rad(geom, 3500.0) == pytest.approx(21.875, rel=1e-12)

    def test_inversely_proportional_to_length(self):
        short = FlexureGeometry(thickness_mm=0.5, length_mm=5.0, width_mm=6.0)
        long = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        assert SLFP.stiffness_nmm_per_rad(short, 3500.0) == pytest.approx(
            2.0 * SLFP.stiffness_nmm_per_rad(long, 3500.0)
        )

    def test_non_positive_modulus_refused(self):
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        with pytest.raises(ValueError, match="modulus must be positive"):
            SLFP.stiffness_nmm_per_rad(geom, 0.0)


class TestStrain:
    def test_matches_the_curvature_statement_of_the_same_physics(self):
        """Eps = t*theta/(2L) must equal t/(2 rho) with rho = L/theta."""
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=8.0, width_mm=6.0)
        theta = np.radians(17.0)
        radius = geom.length_mm / theta
        independent = geom.thickness_mm / (2.0 * radius)
        assert SLFP.peak_strain(geom, theta).peak_strain == pytest.approx(independent, rel=1e-12)

    def test_strip_bent_into_a_closed_circle(self):
        """A strip of length L closed into a circle has rho = L/2pi, eps = pi t / L."""
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=20.0, width_mm=6.0)
        strain = SLFP.peak_strain(geom, 2.0 * np.pi).peak_strain
        assert strain == pytest.approx(np.pi * 0.5 / 20.0, rel=1e-12)

    def test_sign_of_bend_does_not_matter(self):
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        assert SLFP.peak_strain(geom, 0.3).peak_strain == pytest.approx(
            SLFP.peak_strain(geom, -0.3).peak_strain
        )

    def test_model_name_and_exclusions_are_recorded(self):
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        estimate = SLFP.peak_strain(geom, 0.2)
        assert estimate.model == "leaf_uniform_bending"
        assert estimate.includes_axial is False
        assert estimate.includes_stress_concentration is False
        assert estimate.bend_angle_deg == pytest.approx(np.degrees(0.2))


class TestSizingInverses:
    @pytest.mark.parametrize("allowable", [0.005, 0.01, 0.02])
    def test_max_bend_angle_produces_exactly_the_allowable_strain(self, allowable):
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=9.0, width_mm=6.0)
        bend = SLFP.max_bend_angle_rad(geom, allowable)
        assert SLFP.peak_strain(geom, bend).peak_strain == pytest.approx(allowable, rel=1e-12)

    @pytest.mark.parametrize("bend_deg", [5.0, 12.5, 30.0])
    def test_min_length_produces_exactly_the_allowable_strain(self, bend_deg):
        bend = np.radians(bend_deg)
        length = SLFP.min_length_mm(0.6, bend, 0.01)
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=length, width_mm=6.0)
        assert SLFP.peak_strain(geom, bend).peak_strain == pytest.approx(0.01, rel=1e-12)

    def test_min_length_and_max_bend_are_mutual_inverses(self):
        geom = FlexureGeometry(
            thickness_mm=0.6, length_mm=SLFP.min_length_mm(0.6, 0.35, 0.012), width_mm=6.0
        )
        assert SLFP.max_bend_angle_rad(geom, 0.012) == pytest.approx(0.35, rel=1e-12)

    def test_characteristic_pivot_is_at_the_centre(self):
        assert SLFP.characteristic_pivot_fraction() == pytest.approx(0.5)

    def test_axial_stiffness_is_ea_over_l(self):
        geom = FlexureGeometry(thickness_mm=0.5, length_mm=10.0, width_mm=6.0)
        assert SLFP.parasitic_axial_stiffness_n_per_mm(geom, 3500.0) == pytest.approx(
            3500.0 * 0.5 * 6.0 / 10.0, rel=1e-12
        )


class TestPrbmValidity:
    """Validity selects the model and is recorded; it never rejects a design."""

    def test_short_flexure_uses_the_small_length_model(self):
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=8.0, width_mm=6.0)
        validity = prbm_validity(geom, 200.0, 0.2)
        assert validity.length_ratio == pytest.approx(0.04)
        assert validity.is_small_length
        assert validity.model == "small_length"

    def test_long_flexure_switches_to_the_howell_model(self):
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=20.0, width_mm=6.0)
        validity = prbm_validity(geom, 100.0, 0.2)
        assert validity.length_ratio == pytest.approx(0.2)
        assert not validity.is_small_length
        assert validity.model == "long_segment"
        assert any("beam FEA is the reference" in n for n in validity.notes())

    def test_models_differ_in_stiffness_by_the_howell_factor(self):
        """Gamma * K_Theta = 0.85 * 2.65, a 2.25x jump at the switch."""
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=8.0, width_mm=6.0)
        short = PRBM_MODELS.get("small_length")
        long_model = PRBM_MODELS.get("long_segment")
        second_moment = SLFP.second_moment_mm4(geom)
        ratio = long_model.stiffness_nmm_per_rad(
            geom, 3500.0, second_moment
        ) / short.stiffness_nmm_per_rad(geom, 3500.0, second_moment)
        assert ratio == pytest.approx(0.85 * 2.65, rel=1e-9)

    def test_boundary_region_is_flagged(self):
        """Neither model is trustworthy where they disagree by 2.25x."""
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=8.6, width_mm=6.0)
        assert prbm_validity(geom, 80.0, 0.2).near_model_boundary
        assert not prbm_validity(geom, 300.0, 0.2).near_model_boundary

    def test_long_segment_pivot_is_one_minus_gamma(self):
        long_model = PRBM_MODELS.get("long_segment")
        assert long_model.characteristic_pivot_fraction() == pytest.approx(0.15, abs=1e-9)

    def test_small_length_pivot_is_the_centre(self):
        assert PRBM_MODELS.get("small_length").characteristic_pivot_fraction() == pytest.approx(0.5)

    def test_constants_are_flagged_unverified_until_the_beam_fea_runs(self):
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=8.0, width_mm=6.0)
        validity = prbm_validity(geom, 200.0, 0.2)
        assert validity.model_verified is False
        assert any("not yet been checked" in n for n in validity.notes())

    def test_non_positive_link_refused(self):
        geom = FlexureGeometry(thickness_mm=0.6, length_mm=8.0, width_mm=6.0)
        with pytest.raises(ValueError, match="link length must be positive"):
            prbm_validity(geom, 0.0, 0.2)


class TestDesignLimits:
    def test_bound_is_the_exact_elimination_of_flexure_length(self):
        """theta_max = 2 f l eps / (t SF), with L at both hard bounds simultaneously."""
        link, thickness, allowable, fraction, safety = 100.0, 0.6, 0.01, 0.8, 1.5
        limit = max_bend_deg(
            link,
            thickness,
            allowable,
            max_length_fraction=fraction,
            strain_safety_factor=safety,
        )
        expected = np.degrees(2.0 * fraction * link * allowable / (thickness * safety))
        assert limit.max_bend_deg == pytest.approx(expected, rel=1e-12)

        # At that bend, the strain-required length equals the length that fits.
        bend = np.radians(limit.max_bend_deg)
        strain_length = SLFP.min_length_mm(thickness, bend, allowable) * safety
        assert strain_length == pytest.approx(fraction * link, rel=1e-12)

    def test_bound_no_longer_involves_prbm_validity(self):
        """Feasibility is geometric now, so the bound is 8x looser than the old rule."""
        geometric = max_bend_deg(100.0, 0.6, 0.01, max_length_fraction=0.8).max_bend_deg
        old_rule = max_bend_deg(100.0, 0.6, 0.01, max_length_fraction=0.1).max_bend_deg
        assert geometric / old_rule == pytest.approx(8.0, rel=1e-9)

    def test_required_link_length_inverts_max_bend(self):
        limit = max_bend_deg(120.0, 0.5, 0.015)
        assert required_link_length_mm(limit.max_bend_deg, 0.5, 0.015) == pytest.approx(
            120.0, rel=1e-12
        )

    def test_mid_arc_printing_halves_the_link_length_required(self):
        """Printing unstressed at mid-arc halves the one-sided bend, so also the link."""
        at_mid = min_link_length_for_excursion_mm(20.0, 0.6, 0.01, unstressed_at="mid_arc")
        at_start = min_link_length_for_excursion_mm(20.0, 0.6, 0.01, unstressed_at="start")
        assert at_mid == pytest.approx(at_start / 2.0, rel=1e-12)

    def test_thinner_flexures_allow_more_bend(self):
        thin = max_bend_deg(100.0, 0.4, 0.01).max_bend_deg
        thick = max_bend_deg(100.0, 0.8, 0.01).max_bend_deg
        assert thin == pytest.approx(2.0 * thick, rel=1e-12)

    def test_excursion_is_twice_the_one_sided_bend(self):
        limit = max_bend_deg(100.0, 0.6, 0.01)
        assert limit.max_excursion_deg == pytest.approx(2.0 * limit.max_bend_deg)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_inputs_refused(self, bad):
        with pytest.raises(ValueError, match="must be positive"):
            max_bend_deg(bad, 0.6, 0.01)
