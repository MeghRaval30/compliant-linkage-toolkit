"""Validation of the co-rotational beam FEA against analytical references.

The two benchmarks milestone A4 is defined by:

1. a cantilever under a **pure end moment**, which bends into an exact circular
   arc, and
2. a cantilever under a **transverse end load**, whose large-deflection solution
   comes from integrating ``EI theta'' + P cos(theta) = 0`` -- the classical
   Bisshopp-Drucker result, evaluated here by quadrature rather than recalled as
   an elliptic-integral formula.

Neither reference is a stored number from a previous run, so agreement means the
solver reproduces the physics rather than reproducing itself.

The pivot-convention and PRBM-constant checks live here too, because the FEA is
what settles them.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from cmtool.solvers.beam import BeamModel, BeamSection, ConvergenceError, solve
from cmtool.solvers.beam_reference import arc_shape, end_load_tip, end_moment_tip

pytestmark = pytest.mark.validation

LENGTH_MM = 100.0
WIDTH_MM = 6.0
THICKNESS_MM = 1.0
MODULUS_MPA = 3500.0


def cantilever(n_elements: int):
    """Build a straight cantilever along +x, clamped at node 0."""
    nodes = np.column_stack([np.linspace(0.0, LENGTH_MM, n_elements + 1), np.zeros(n_elements + 1)])
    section = BeamSection.rectangular(THICKNESS_MM, WIDTH_MM, MODULUS_MPA)
    model = BeamModel(nodes, [(i, i + 1) for i in range(n_elements)], [section] * n_elements)
    return model, section


CLAMPED = {0: 0.0, 1: 0.0, 2: 0.0}


class TestSection:
    def test_rectangular_properties(self):
        section = BeamSection.rectangular(0.6, 6.0, 3500.0)
        assert section.area_mm2 == pytest.approx(3.6)
        assert section.second_moment_mm4 == pytest.approx(6.0 * 0.6**3 / 12.0)
        assert section.ei == pytest.approx(3500.0 * 6.0 * 0.6**3 / 12.0)


class TestEndMomentBenchmark:
    """A pure end moment gives constant curvature: an exact circular arc."""

    @pytest.mark.parametrize("theta_deg", [15.0, 45.0, 90.0])
    def test_tip_position_matches_the_exact_arc(self, theta_deg):
        model, section = cantilever(32)
        moment = np.radians(theta_deg) * section.ei / LENGTH_MM
        outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(32, 2): moment}, steps=12)
        tip = model.deformed_nodes(outcome.displacement)[-1] / LENGTH_MM
        reference = end_moment_tip(LENGTH_MM, section.ei, moment)
        assert float(np.linalg.norm(tip - reference.tip)) < 1e-4

    def test_tip_slope_is_exact(self):
        """The co-rotational element represents constant-curvature rotation exactly."""
        model, section = cantilever(8)
        moment = np.radians(90.0) * section.ei / LENGTH_MM
        outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(8, 2): moment}, steps=12)
        slope = model.node_rotations(outcome.displacement)[-1]
        assert slope == pytest.approx(np.radians(90.0), abs=1e-8)

    def test_converges_at_second_order(self):
        """Halving the element size must quarter the error."""
        errors = []
        for n in (4, 8, 16, 32):
            model, section = cantilever(n)
            moment = np.radians(90.0) * section.ei / LENGTH_MM
            outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(n, 2): moment}, steps=12)
            tip = model.deformed_nodes(outcome.displacement)[-1] / LENGTH_MM
            errors.append(
                float(np.linalg.norm(tip - end_moment_tip(LENGTH_MM, section.ei, moment).tip))
            )
        for coarse, fine in itertools.pairwise(errors):
            assert coarse / fine == pytest.approx(4.0, rel=0.25)

    def test_whole_deflected_shape_matches_the_arc(self):
        """Not just the tip: every node lies on the exact circle."""
        n = 32
        model, section = cantilever(n)
        theta = np.radians(60.0)
        moment = theta * section.ei / LENGTH_MM
        outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(n, 2): moment}, steps=12)
        computed = model.deformed_nodes(outcome.displacement) / LENGTH_MM
        exact = arc_shape(theta, samples=n + 1)
        assert float(np.max(np.linalg.norm(computed - exact, axis=1))) < 1e-4

    def test_curvature_is_uniform_along_the_beam(self):
        n = 16
        model, section = cantilever(n)
        moment = np.radians(45.0) * section.ei / LENGTH_MM
        outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(n, 2): moment}, steps=10)
        curvature = model.element_curvatures(outcome.displacement)
        expected = moment / section.ei
        np.testing.assert_allclose(curvature, expected, rtol=1e-6)


class TestEndLoadBenchmark:
    """A transverse tip load: the Bisshopp-Drucker large-deflection solution."""

    def test_reference_reduces_to_the_linear_cantilever(self):
        """Small load: tip deflection must approach PL^3/3EI, i.e. y/L -> alpha/3."""
        state = end_load_tip(0.01)
        assert state.tip_y_over_l == pytest.approx(0.01 / 3.0, rel=1e-4)

    @pytest.mark.parametrize("alpha", [0.5, 1.0, 2.0, 3.0])
    def test_tip_matches_the_reference(self, alpha):
        n = 32
        model, section = cantilever(n)
        load = alpha * section.ei / LENGTH_MM**2
        outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(n, 1): load}, steps=16)
        tip = model.deformed_nodes(outcome.displacement)[-1] / LENGTH_MM
        reference = end_load_tip(alpha)
        assert abs(tip[0] - reference.tip_x_over_l) < 1e-4
        assert abs(tip[1] - reference.tip_y_over_l) < 1e-4

    def test_tip_slope_matches_the_reference(self):
        n = 32
        model, section = cantilever(n)
        alpha = 2.0
        load = alpha * section.ei / LENGTH_MM**2
        outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(n, 1): load}, steps=16)
        slope = model.node_rotations(outcome.displacement)[-1]
        assert np.degrees(abs(slope - end_load_tip(alpha).tip_slope_rad)) < 0.01

    def test_large_deflection_differs_from_linear_theory(self):
        """Confirms the benchmark is actually in the nonlinear regime."""
        reference = end_load_tip(3.0)
        linear = 3.0 / 3.0
        assert reference.tip_y_over_l < 0.8 * linear
        assert reference.tip_x_over_l < 0.95


class TestSolverBehaviour:
    def test_zero_load_gives_zero_displacement(self):
        model, _ = cantilever(8)
        outcome = solve(model, fixed_dofs=CLAMPED, steps=1)
        np.testing.assert_allclose(outcome.displacement, 0.0, atol=1e-12)

    def test_prescribed_rotation_produces_a_reaction_moment(self):
        """Input torque from the mechanism solve is a reaction at a prescribed DOF."""
        n = 8
        model, section = cantilever(n)
        angle = np.radians(20.0)
        outcome = solve(
            model,
            fixed_dofs=CLAMPED,
            prescribed_dofs={model.dof(n, 2): angle},
            steps=10,
        )
        # A cantilever held at a tip rotation with nothing else applied bends into
        # an arc, so the reaction is the end moment: EI*theta/L.
        assert outcome.reaction[model.dof(n, 2)] == pytest.approx(
            section.ei * angle / LENGTH_MM, rel=1e-3
        )

    def test_unconstrained_model_is_reported_not_silently_wrong(self):
        model, _ = cantilever(4)
        with pytest.raises(ConvergenceError, match=r"singular|under-constrained"):
            solve(model, fixed_dofs={}, loads={model.dof(4, 1): 1.0}, steps=1)

    def test_a_dof_cannot_be_both_fixed_and_prescribed(self):
        model, _ = cantilever(4)
        with pytest.raises(ValueError, match="both fixed and prescribed"):
            solve(model, fixed_dofs={2: 0.0}, prescribed_dofs={2: 0.1})

    def test_newton_converges_in_few_iterations(self):
        """The geometric stiffness terms are what buy quadratic convergence."""
        n = 16
        model, section = cantilever(n)
        moment = np.radians(60.0) * section.ei / LENGTH_MM
        outcome = solve(model, fixed_dofs=CLAMPED, loads={model.dof(n, 2): moment}, steps=10)
        assert max(outcome.iterations) <= 8
        assert outcome.summary()["converged"] is True


class TestPivotConvention:
    """Which end the characteristic pivot is measured from, settled numerically."""

    @staticmethod
    def _fit_gamma(pivot_at_one_minus_gamma: bool, theta_max_deg: float = 45.0) -> float:
        angles = np.linspace(1e-6, np.radians(theta_max_deg), 120)
        exact = np.array([arc_shape(t, samples=2)[-1] for t in angles])

        def error(gamma: float) -> float:
            link = gamma if pivot_at_one_minus_gamma else 1.0 - gamma
            pivot = 1.0 - gamma if pivot_at_one_minus_gamma else gamma
            total = 0.0
            for x, y in exact:
                ratio = np.clip(y / link, -1.0, 1.0)
                angle = np.arcsin(ratio)
                total += (pivot + link * np.cos(angle) - x) ** 2
            return float(np.sqrt(total / len(exact)))

        return float(minimize_scalar(error, bounds=(0.5, 0.95), method="bounded").fun)

    def test_pivot_at_one_minus_gamma_fits_the_exact_arc(self):
        assert self._fit_gamma(True) < 1e-4

    def test_pivot_at_gamma_does_not(self):
        """Roughly a thousand times worse, which is how the convention is decided."""
        assert self._fit_gamma(False) > 100.0 * self._fit_gamma(True)

    def test_fitted_gamma_matches_the_end_moment_constant(self):
        angles = np.linspace(1e-6, np.radians(45.0), 120)
        exact = np.array([arc_shape(t, samples=2)[-1] for t in angles])

        def error(gamma: float) -> float:
            total = 0.0
            for x, y in exact:
                angle = np.arcsin(np.clip(y / gamma, -1.0, 1.0))
                total += ((1.0 - gamma) + gamma * np.cos(angle) - x) ** 2
            return float(np.sqrt(total / len(exact)))

        gamma = float(minimize_scalar(error, bounds=(0.5, 0.95), method="bounded").x)
        # Analytic small-angle limit is exactly 3/4; the config value is 0.7346.
        assert gamma == pytest.approx(0.75, abs=0.01)
        assert gamma == pytest.approx(0.7346, abs=0.02)


@pytest.mark.slow
class TestPrbmStudy:
    """Fitting the PRBM to the FEA, which is what chose the active variant."""

    def test_pure_moment_recovers_the_end_moment_constants(self):
        from cmtool.solvers.prbm_study import fit_prbm

        fitted = fit_prbm(0.0)
        assert fitted.gamma == pytest.approx(0.7346, abs=0.02)
        assert fitted.stiffness_multiple == pytest.approx(1.5164, rel=0.02)

    def test_force_dominated_loading_moves_towards_the_end_force_constants(self):
        from cmtool.solvers.prbm_study import fit_prbm

        moment_like = fit_prbm(0.0)
        force_like = fit_prbm(5.0)
        assert force_like.gamma > moment_like.gamma
        assert force_like.stiffness_multiple > moment_like.stiffness_multiple
        assert force_like.gamma == pytest.approx(0.85, abs=0.05)

    def test_joint_loading_selects_the_end_moment_variant(self):
        """The result that set active_variant in configs/models/prbm.yaml."""
        from cmtool.solvers.prbm_study import recommend_variant

        report = recommend_variant()
        assert report["recommended"] == "end_moment"
        assert report["variants"]["end_moment"]["stiffness_error"] < 0.02
        assert report["variants"]["end_force"]["stiffness_error"] > 0.3
