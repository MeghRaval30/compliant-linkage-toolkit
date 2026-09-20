"""Validation of the pseudo-rigid-body quasi-static solver.

The headline check is a parallelogram. Every joint in a parallelogram rotates
through exactly the input sweep -- proved independently in the four-bar kinematics
tests -- so with equal springs the input torque has a closed form:

``T(theta) = (sum of K) * (theta - theta_reference)``

That exercises kinematics, the energy sum and the numerical differentiation
together, against an answer derived rather than recorded.

The rest pin down properties the model must have: torque vanishes where the part
is unstressed, torque scales linearly with stiffness while the path does not move
at all, and the reported energy and torque are consistent with each other.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmtool import Linkage, convert, simulate
from cmtool.flexures.prbm_models import PRBM_MODELS, select_model

pytestmark = [
    pytest.mark.validation,
    pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning"),
]


def parallelogram(flexure_length_mm: float = 4.0, thickness_mm: float = 0.6):
    """Build a parallelogram four-bar converted with four identical flexures."""
    mech = Linkage.four_bar(
        ground_mm=80.0,
        input_mm=30.0,
        coupler_mm=80.0,
        output_mm=30.0,
        input_angle_deg=70.0,
        input_range_deg=(60.0, 80.0),
        branch=1,
        name="parallelogram",
    )
    return convert(
        mech,
        thickness_mm=thickness_mm,
        flexure_length_mm=flexure_length_mm,
        unstressed_at="mid_arc",
    )


class TestParallelogramClosedForm:
    def test_springs_are_identical(self):
        stiffness = [s.stiffness_nmm_per_rad for s in parallelogram().sizing.values()]
        assert len(set(round(k, 9) for k in stiffness)) == 1

    def test_torque_matches_the_closed_form(self):
        """T = (sum K) * dtheta exactly, because every joint rotates by dtheta."""
        mech = parallelogram()
        result = simulate(mech, solver="prbm", n_steps=41)
        total = sum(s.stiffness_nmm_per_rad for s in mech.sizing.values())
        sweep = np.radians(result.input_angles_deg - mech.reference_input_deg)
        np.testing.assert_allclose(result.input_torque_nmm, total * sweep, atol=1e-6)

    def test_energy_matches_the_closed_form(self):
        mech = parallelogram()
        result = simulate(mech, solver="prbm", n_steps=41)
        total = sum(s.stiffness_nmm_per_rad for s in mech.sizing.values())
        sweep = np.radians(result.input_angles_deg - mech.reference_input_deg)
        np.testing.assert_allclose(
            result.diagnostics["strain_energy_nmm"], 0.5 * total * sweep**2, atol=1e-9
        )

    def test_torque_is_the_derivative_of_the_energy(self):
        """Checks the two reported quantities against each other."""
        result = simulate(parallelogram(), solver="prbm", n_steps=61)
        angles = result.input_angles_rad
        gradient = np.gradient(result.diagnostics["strain_energy_nmm"], angles, edge_order=2)
        np.testing.assert_allclose(result.input_torque_nmm, gradient, atol=1e-9)


class TestQuasiStaticProperties:
    def test_torque_vanishes_where_the_part_is_unstressed(self):
        mech = parallelogram()
        result = simulate(mech, solver="prbm", n_steps=41)
        middle = result.n_states // 2
        assert result.input_angles_deg[middle] == pytest.approx(mech.reference_input_deg)
        assert result.input_torque_nmm[middle] == pytest.approx(0.0, abs=1e-9)

    def test_torque_scales_linearly_with_stiffness(self):
        """Halving every flexure length doubles K, so it doubles the torque."""
        soft = simulate(parallelogram(flexure_length_mm=8.0), solver="prbm", n_steps=21)
        stiff = simulate(parallelogram(flexure_length_mm=4.0), solver="prbm", n_steps=21)
        ratio = np.abs(stiff.input_torque_nmm[1:]) / np.abs(soft.input_torque_nmm[1:])
        np.testing.assert_allclose(ratio, 2.0, rtol=1e-6)

    def test_stiffness_does_not_move_the_path(self):
        """Prescribed input, one degree of freedom, no load: geometry alone fixes the path."""
        soft = simulate(parallelogram(flexure_length_mm=8.0), solver="prbm", n_steps=21)
        stiff = simulate(parallelogram(flexure_length_mm=4.0), solver="prbm", n_steps=21)
        np.testing.assert_allclose(soft.path(), stiff.path(), atol=1e-9)

    def test_pivot_matched_prbm_path_equals_the_rigid_path(self):
        """Matching the pivots is what removes the conversion artefact."""
        mech = parallelogram()
        prbm = simulate(mech, solver="prbm", n_steps=21)
        rigid = simulate(
            mech.base, solver="rigid", input_range_deg=mech.input_range_deg, n_steps=21
        )
        np.testing.assert_allclose(prbm.path(), rigid.path(), atol=1e-9)

    def test_unmatched_placement_shifts_the_path(self):
        """The artefact the matched default exists to remove, made visible."""
        base = parallelogram().base
        unmatched = convert(base, thickness_mm=0.6, flexure_length_mm=4.0, placement="unmatched")
        shifted = simulate(unmatched, solver="prbm", n_steps=21)
        rigid = simulate(
            base, solver="rigid", input_range_deg=unmatched.input_range_deg, n_steps=21
        )
        offsets = np.linalg.norm(shifted.path() - rigid.path(), axis=1)
        # About 0.2 mm for a 4 mm flexure -- small, but the same order as the
        # measurement uncertainty we are trying to resolve, so it cannot be ignored.
        assert offsets.max() > 0.1

    def test_the_unmatched_offset_scales_with_flexure_length(self):
        """Confirms the shift is the pivot displacement, not numerical noise."""
        base = parallelogram().base

        def offset_for(length_mm: float) -> float:
            unmatched = convert(
                base, thickness_mm=0.6, flexure_length_mm=length_mm, placement="unmatched"
            )
            shifted = simulate(unmatched, solver="prbm", n_steps=21)
            rigid = simulate(
                base, solver="rigid", input_range_deg=unmatched.input_range_deg, n_steps=21
            )
            return float(np.linalg.norm(shifted.path() - rigid.path(), axis=1).max())

        assert offset_for(8.0) / offset_for(4.0) == pytest.approx(2.0, rel=0.15)


class TestSolverContract:
    def test_reports_torque_and_strain(self):
        result = simulate(parallelogram(), solver="prbm", n_steps=21)
        assert result.input_torque_nmm is not None
        assert set(result.flexure_strain) == {"A", "B", "C", "D"}
        assert all(np.all(v >= 0.0) for v in result.flexure_strain.values())

    def test_strain_is_zero_at_the_unstressed_configuration(self):
        result = simulate(parallelogram(), solver="prbm", n_steps=41)
        middle = result.n_states // 2
        for values in result.flexure_strain.values():
            assert values[middle] == pytest.approx(0.0, abs=1e-12)

    def test_strain_margin_reflects_the_safety_factor(self):
        """Flexures sized to the strain minimum sit exactly one safety factor inside."""
        mech = parallelogram(flexure_length_mm=None)  # type: ignore[arg-type]
        result = simulate(mech, solver="prbm", n_steps=41)
        assert result.diagnostics["strain_margin"] == pytest.approx(
            mech.feasibility.strain_safety_factor, rel=1e-6
        )

    def test_refuses_to_run_without_the_compliant_mechanism(self):
        mech = parallelogram()
        with pytest.raises(ValueError, match="needs the compliant mechanism"):
            simulate(mech.base, solver="prbm", input_range_deg=(60.0, 80.0))

    def test_records_which_prbm_model_each_joint_uses(self):
        result = simulate(parallelogram(), solver="prbm", n_steps=11)
        assert set(result.diagnostics["prbm_models"]) == {"A", "B", "C", "D"}
        assert isinstance(result.diagnostics["all_small_length"], bool)

    def test_carries_the_placeholder_caveat_through(self):
        result = simulate(parallelogram(), solver="prbm", n_steps=11)
        assert not result.is_physical
        assert "PLA.properties.youngs_modulus_MPa" in result.provenance.placeholders_used


class TestModelSelection:
    def test_ratio_below_the_limit_selects_small_length(self):
        assert select_model(0.05).name == "small_length"

    def test_ratio_above_the_limit_selects_long_segment(self):
        assert select_model(0.25).name.startswith("long_segment")

    def test_the_active_variant_is_the_one_the_fea_chose(self):
        from cmtool.flexures.prbm_models import active_long_segment_variant

        assert active_long_segment_variant() == "end_moment"
        assert select_model(0.25).name == "long_segment_end_moment"

    def test_selection_is_inclusive_at_the_limit(self):
        assert select_model(0.1).name == "small_length"

    def test_long_segment_max_angle_is_reported(self):
        assert PRBM_MODELS.get("long_segment_end_moment").max_angle_deg() == pytest.approx(64.3)

    def test_small_length_has_no_separate_angle_limit(self):
        assert PRBM_MODELS.get("small_length").max_angle_deg() is None
