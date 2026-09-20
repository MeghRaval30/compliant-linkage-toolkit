"""Validation of the four-bar kinematics against independent references.

Three kinds of check, in increasing order of independence:

1. **Loop closure residuals.** The solved configuration must reproduce the link
   lengths exactly. This is unambiguous and needs no external reference.
2. **Exact hand-computed configurations.** A 3-4-5 construction whose joint
   positions and transmission angle are worked out on paper.
3. **Independent formulations.** The same positions obtained by (a) a numerical
   vector-loop solve with ``scipy.optimize.fsolve`` and (b) Freudenstein's
   closed-form equation. Both are different derivations from the circle
   intersection used by the solver, so agreement is meaningful.

Analytical special cases (parallelogram, kite) pin down the behaviour that a
sign error would break.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from scipy.optimize import fsolve

from cmtool.core.graph import Linkage
from cmtool.kinematics.fourbar import (
    AssemblyError,
    FourBarLengths,
    FourBarSolver,
    classify_grashof,
    dyad_branch,
    identify_four_bar,
    link_lengths_of,
    reachable_input_arc,
    solve_dyad,
    transmission_angle,
)

pytestmark = pytest.mark.validation

SOLVER = FourBarSolver()


# ----------------------------------------------------------------- references


def freudenstein_theta4(lengths: FourBarLengths, theta2: float, sign: float = 1.0) -> float | None:
    """Output angle from Freudenstein's equation -- an independent formulation.

    With ground ``d``, input ``a``, coupler ``b`` and output ``c``, and both
    angles measured from the ground link direction,

    ``K1 cos(t4) - K2 cos(t2) + K3 = cos(t2 - t4)``

    which the half-angle substitution turns into a quadratic in ``tan(t4 / 2)``.
    """
    a, b, c, d = lengths.input, lengths.coupler, lengths.output, lengths.ground
    k1, k2 = d / a, d / c
    k3 = (a**2 - b**2 + c**2 + d**2) / (2 * a * c)

    cos2, sin2 = np.cos(theta2), np.sin(theta2)
    quad_a = cos2 - k1 - k2 * cos2 + k3
    quad_b = -2.0 * sin2
    quad_c = k1 - (k2 + 1.0) * cos2 + k3

    disc = quad_b**2 - 4.0 * quad_a * quad_c
    if disc < 0.0:
        return None
    root = np.sqrt(disc)
    return float(2.0 * np.arctan2(-quad_b + sign * root, 2.0 * quad_a))


def vector_loop_solve(
    lengths: FourBarLengths, theta2: float, guess: tuple[float, float]
) -> tuple[float, float]:
    """Solve the vector loop numerically -- an independent formulation.

    Closure is ``r2 u(t2) + r3 u(t3) - r4 u(t4) - r1 u(0) = 0`` with the ground
    link along +x. Returns ``(theta3, theta4)``.
    """

    def residual(angles: np.ndarray) -> np.ndarray:
        t3, t4 = angles
        return np.array(
            [
                lengths.input * np.cos(theta2)
                + lengths.coupler * np.cos(t3)
                - lengths.output * np.cos(t4)
                - lengths.ground,
                lengths.input * np.sin(theta2)
                + lengths.coupler * np.sin(t3)
                - lengths.output * np.sin(t4),
            ]
        )

    solution, _, ok, msg = fsolve(residual, np.array(guess), full_output=True)[:4]
    assert ok == 1, f"reference solve failed: {msg}"
    return float(solution[0]), float(solution[1])


# ------------------------------------------------------------------- the dyad


class TestDyad:
    def test_exact_intersection(self):
        """Circles of radius 3 about (0,3) and radius 4 about (4,0), centres 5 apart."""
        point = solve_dyad(np.array([0.0, 3.0]), 3.0, np.array([4.0, 0.0]), 4.0, branch=1)
        np.testing.assert_allclose(point, [2.88, 3.84], atol=1e-12)

    def test_other_branch_is_the_mirror_solution(self):
        point = solve_dyad(np.array([0.0, 3.0]), 3.0, np.array([4.0, 0.0]), 4.0, branch=-1)
        np.testing.assert_allclose(point, [0.0, 0.0], atol=1e-12)

    def test_both_branches_satisfy_both_radii(self):
        for branch in (1, -1):
            point = solve_dyad(np.array([0.0, 3.0]), 3.0, np.array([4.0, 0.0]), 4.0, branch=branch)
            assert np.linalg.norm(point - np.array([0.0, 3.0])) == pytest.approx(3.0)
            assert np.linalg.norm(point - np.array([4.0, 0.0])) == pytest.approx(4.0)

    def test_branch_label_round_trips(self):
        centre_a, centre_b = np.array([0.0, 3.0]), np.array([4.0, 0.0])
        for branch in (1, -1):
            point = solve_dyad(centre_a, 3.0, centre_b, 4.0, branch=branch)
            assert dyad_branch(centre_a, centre_b, point) == branch

    def test_too_far_apart_does_not_close(self):
        assert solve_dyad(np.zeros(2), 1.0, np.array([10.0, 0.0]), 1.0) is None

    def test_one_circle_inside_the_other_does_not_close(self):
        assert solve_dyad(np.zeros(2), 10.0, np.array([1.0, 0.0]), 1.0) is None

    def test_tangent_circles_give_a_single_point_on_both_branches(self):
        left = solve_dyad(np.zeros(2), 2.0, np.array([5.0, 0.0]), 3.0, branch=1)
        right = solve_dyad(np.zeros(2), 2.0, np.array([5.0, 0.0]), 3.0, branch=-1)
        np.testing.assert_allclose(left, right, atol=1e-9)
        np.testing.assert_allclose(left, [2.0, 0.0], atol=1e-9)


# ----------------------------------------------------- exact hand-computed case


class TestHandComputedConfiguration:
    """Ground 4, input 3 at 90 deg, coupler 3, output 4 -- a 3-4-5 construction.

    ``A = (0,0)``, ``D = (4,0)``, ``B = (0,3)`` so ``|BD| = 5``. Intersecting the
    coupler circle (r=3 about B) with the output circle (r=4 about D) gives
    ``C = (2.88, 3.84)`` on the positive branch. At that pose ``C->B`` and
    ``C->D`` are exactly perpendicular, so the transmission angle is exactly 90
    degrees.
    """

    LENGTHS = FourBarLengths(ground=4.0, input=3.0, coupler=3.0, output=4.0)

    def build(self) -> Linkage:
        return Linkage.four_bar(
            ground_mm=4.0,
            input_mm=3.0,
            coupler_mm=3.0,
            output_mm=4.0,
            input_angle_deg=90.0,
            input_range_deg=(80.0, 100.0),
            branch=1,
        )

    def test_joint_c_position_is_exact(self):
        mech = self.build()
        np.testing.assert_allclose(mech.joints["C"].position_mm, [2.88, 3.84], atol=1e-12)

    def test_link_lengths_recovered_from_the_graph(self):
        lengths = link_lengths_of(self.build())
        assert lengths.as_tuple() == pytest.approx(self.LENGTHS.as_tuple())

    def test_transmission_angle_is_exactly_ninety_degrees(self):
        mech = self.build()
        mu = transmission_angle(
            mech.joints["B"].position_mm,
            mech.joints["C"].position_mm,
            mech.joints["D"].position_mm,
        )
        assert np.degrees(mu) == pytest.approx(90.0, abs=1e-9)

    def test_equal_opposite_pairs_are_a_change_point_mechanism(self):
        result = classify_grashof(self.LENGTHS)
        assert result.condition == "change_point"
        assert result.s_plus_l == pytest.approx(result.p_plus_q)


# ---------------------------------------------------------- analytical cases


class TestParallelogram:
    """AB = CD and BC = AD, assembled on the branch that keeps it a parallelogram.

    The coupler then stays parallel to the ground link at every input angle, the
    output link stays parallel to the input link, and all four joints rotate
    through exactly the input sweep. Any sign error in the branch handling or the
    body frames breaks at least one of those.
    """

    GROUND, CRANK = 40.0, 15.0

    def build(self) -> Linkage:
        return Linkage.four_bar(
            ground_mm=self.GROUND,
            input_mm=self.CRANK,
            coupler_mm=self.GROUND,
            output_mm=self.CRANK,
            input_angle_deg=70.0,
            input_range_deg=(40.0, 110.0),
            branch=1,
        )

    def test_coupler_translates_without_rotating(self):
        mech = self.build()
        states = SOLVER.solve(mech, np.radians(np.linspace(40.0, 110.0, 25)))
        for state in states:
            assert state.body_angles_rad["coupler"] == pytest.approx(0.0, abs=1e-9)

    def test_output_link_stays_parallel_to_the_input_link(self):
        mech = self.build()
        for theta2 in np.radians(np.linspace(40.0, 110.0, 25)):
            state = SOLVER.solve(mech, np.array([theta2]))[0]
            direction = state.joint_positions_mm["C"] - state.joint_positions_mm["D"]
            assert np.arctan2(direction[1], direction[0]) == pytest.approx(theta2, abs=1e-9)

    def test_coupler_offset_stays_the_ground_vector(self):
        mech = self.build()
        for state in SOLVER.solve(mech, np.radians(np.linspace(40.0, 110.0, 15))):
            offset = state.joint_positions_mm["C"] - state.joint_positions_mm["B"]
            np.testing.assert_allclose(offset, [self.GROUND, 0.0], atol=1e-9)

    def test_every_joint_rotates_through_the_full_input_sweep(self):
        from cmtool import simulate

        result = simulate(self.build(), solver="rigid", n_steps=41)
        for joint, excursion in result.joint_excursion_deg().items():
            assert excursion == pytest.approx(70.0, abs=1e-6), joint

    def test_coupler_point_traces_a_circle_congruent_to_the_crank(self):
        from cmtool import simulate

        mech = self.build()
        result = simulate(mech, solver="rigid", n_steps=41)
        path = result.path("P")
        # A parallelogram's coupler curve is a circular arc of the crank radius.
        centre = path[0] - (result.states[0].joint_positions_mm["B"] - mech.joints["A"].position_mm)
        radii = np.linalg.norm(path - centre, axis=1)
        assert radii.std() == pytest.approx(0.0, abs=1e-9)
        assert radii.mean() == pytest.approx(self.CRANK, abs=1e-9)


class TestAntiParallelogram:
    def test_opposite_branch_is_not_the_parallelogram(self):
        """Same link lengths, other branch: the coupler must rotate."""
        mech = Linkage.four_bar(
            ground_mm=40.0,
            input_mm=15.0,
            coupler_mm=40.0,
            output_mm=15.0,
            input_angle_deg=70.0,
            input_range_deg=(40.0, 110.0),
            branch=-1,
        )
        states = SOLVER.solve(mech, np.radians(np.linspace(40.0, 110.0, 15)))
        coupler = np.array([s.body_angles_rad["coupler"] for s in states])
        assert np.ptp(coupler) > np.radians(1.0)


# --------------------------------------------------- independent formulations


class TestAgainstIndependentFormulations:
    LENGTHS = FourBarLengths(ground=50.0, input=18.0, coupler=45.0, output=38.0)

    def build(self, input_angle_deg: float = 60.0, branch: int = 1) -> Linkage:
        return Linkage.four_bar(
            ground_mm=self.LENGTHS.ground,
            input_mm=self.LENGTHS.input,
            coupler_mm=self.LENGTHS.coupler,
            output_mm=self.LENGTHS.output,
            input_angle_deg=input_angle_deg,
            input_range_deg=(30.0, 120.0),
            branch=branch,
        )

    @pytest.mark.parametrize("theta2_deg", [30.0, 55.0, 90.0, 120.0, 160.0])
    def test_matches_numerical_vector_loop_solve(self, theta2_deg):
        mech = self.build()
        theta2 = np.radians(theta2_deg)
        state = SOLVER.solve(mech, np.array([theta2]))[0]

        pos_b = state.joint_positions_mm["B"]
        pos_c = state.joint_positions_mm["C"]
        pos_d = state.joint_positions_mm["D"]
        theta3 = np.arctan2(*(pos_c - pos_b)[::-1])
        theta4 = np.arctan2(*(pos_c - pos_d)[::-1])

        ref3, ref4 = vector_loop_solve(self.LENGTHS, theta2, (theta3 + 0.05, theta4 - 0.05))
        assert np.cos(ref3 - theta3) == pytest.approx(1.0, abs=1e-9)
        assert np.cos(ref4 - theta4) == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("theta2_deg", [30.0, 55.0, 90.0, 120.0, 160.0])
    def test_matches_one_freudenstein_root(self, theta2_deg):
        mech = self.build()
        theta2 = np.radians(theta2_deg)
        state = SOLVER.solve(mech, np.array([theta2]))[0]
        pos_c = state.joint_positions_mm["C"]
        pos_d = state.joint_positions_mm["D"]
        theta4 = np.arctan2(*(pos_c - pos_d)[::-1])

        roots = [freudenstein_theta4(self.LENGTHS, theta2, sign) for sign in (1.0, -1.0)]
        roots = [r for r in roots if r is not None]
        assert roots, "Freudenstein found no real root where the solver assembled"
        assert min(abs(np.cos(r - theta4) - 1.0) for r in roots) < 1e-9


# -------------------------------------------------------- Grashof, arcs, edges


class TestGrashof:
    @pytest.mark.parametrize(
        ("lengths", "expected"),
        [
            ((10.0, 3.0, 8.0, 9.0), "crank_rocker"),  # shortest is the input
            ((3.0, 10.0, 8.0, 9.0), "double_crank"),  # shortest is ground
            ((9.0, 10.0, 3.0, 8.0), "double_rocker"),  # shortest is the coupler
            ((8.0, 9.0, 10.0, 3.0), "rocker_crank"),  # shortest is the output
        ],
    )
    def test_grashof_classification_by_grounded_link(self, lengths, expected):
        result = classify_grashof(FourBarLengths(*lengths))
        assert result.condition == "grashof"
        assert result.classification == expected

    def test_non_grashof_is_a_triple_rocker(self):
        result = classify_grashof(FourBarLengths(ground=4.0, input=5.0, coupler=6.0, output=10.0))
        assert result.condition == "non_grashof"
        assert result.classification == "triple_rocker"
        assert result.input_fully_rotates is False

    def test_change_point_when_sums_are_equal(self):
        result = classify_grashof(FourBarLengths(ground=5.0, input=5.0, coupler=8.0, output=8.0))
        assert result.condition == "change_point"

    @pytest.mark.parametrize(
        ("lengths", "rotates"),
        [
            ((10.0, 3.0, 8.0, 9.0), True),  # crank-rocker: input is the crank
            ((3.0, 10.0, 8.0, 9.0), True),  # drag link: input rotates fully
            ((9.0, 10.0, 3.0, 8.0), False),  # Grashof double-rocker
            ((4.0, 5.0, 6.0, 10.0), False),  # non-Grashof
        ],
    )
    def test_input_full_rotation_flag(self, lengths, rotates):
        """A fully rotating input has no direct compliant equivalent."""
        assert classify_grashof(FourBarLengths(*lengths)).input_fully_rotates is rotates

    def test_classification_does_not_depend_on_scale(self):
        base = FourBarLengths(10.0, 3.0, 8.0, 9.0)
        scaled = FourBarLengths(*(10.0 * v for v in base.as_tuple()))
        assert classify_grashof(base).classification == classify_grashof(scaled).classification


class TestReachableArc:
    @pytest.mark.parametrize(
        "lengths",
        [
            (50.0, 18.0, 45.0, 38.0),
            (4.0, 5.0, 6.0, 10.0),
            (40.0, 15.0, 40.0, 15.0),
            (10.0, 3.0, 8.0, 9.0),
        ],
    )
    def test_analytic_arc_matches_brute_force_scan(self, lengths):
        """The closed-form reachable arc must agree with actually trying to assemble."""
        spec = FourBarLengths(*lengths)
        phi_min, phi_max = reachable_input_arc(spec)

        for phi in np.radians(np.arange(0.0, 180.0, 0.25)):
            pos_b = spec.input * np.array([np.cos(phi), np.sin(phi)])
            closes = (
                solve_dyad(pos_b, spec.coupler, np.array([spec.ground, 0.0]), spec.output)
                is not None
            )
            predicted = (phi_min - 1e-12) <= phi <= (phi_max + 1e-12)
            assert closes == predicted, f"phi={np.degrees(phi):.2f} deg"

    def test_crank_rocker_input_reaches_everything(self):
        phi_min, phi_max = reachable_input_arc(FourBarLengths(10.0, 3.0, 8.0, 9.0))
        assert phi_min == pytest.approx(0.0)
        assert phi_max == pytest.approx(np.pi)


class TestSolverBehaviour:
    def test_unreachable_angle_raises_with_the_reachable_arc_in_the_message(self):
        mech = Linkage.four_bar(
            ground_mm=4.0,
            input_mm=5.0,
            coupler_mm=6.0,
            output_mm=10.0,
            input_angle_deg=90.0,
            input_range_deg=(80.0, 100.0),
        )
        with pytest.raises(AssemblyError, match="reachable arc"):
            SOLVER.solve(mech, np.radians([0.0]))

    def test_non_strict_mode_skips_unreachable_angles(self):
        mech = Linkage.four_bar(
            ground_mm=4.0,
            input_mm=5.0,
            coupler_mm=6.0,
            output_mm=10.0,
            input_angle_deg=90.0,
            input_range_deg=(80.0, 100.0),
        )
        states = SOLVER.solve(mech, np.radians([0.0, 90.0]), strict=False)
        assert len(states) == 1

    def test_solver_claims_only_four_bar_topologies(self):
        four_bar = Linkage.four_bar(
            ground_mm=50.0,
            input_mm=18.0,
            coupler_mm=45.0,
            output_mm=38.0,
            input_range_deg=(30.0, 90.0),
        )
        assert SOLVER.can_solve(four_bar)

        from tests.test_graph import make_four_bar

        chain = make_four_bar()
        del chain.joints["D"]
        del chain.bodies["output"]
        assert not SOLVER.can_solve(chain)

    def test_roles_are_found_regardless_of_naming(self):
        mech = Linkage.four_bar(ground_mm=50.0, input_mm=18.0, coupler_mm=45.0, output_mm=38.0)
        roles = identify_four_bar(mech)
        assert roles.joint_a == "A"
        assert roles.coupler == "coupler"
        assert roles.joints == ("A", "B", "C", "D")

    def test_diagnostics_report_transmission_angle_and_branch(self):
        mech = Linkage.four_bar(
            ground_mm=50.0,
            input_mm=18.0,
            coupler_mm=45.0,
            output_mm=38.0,
            input_range_deg=(40.0, 100.0),
        )
        states = SOLVER.solve(mech, np.radians(np.linspace(40.0, 100.0, 31)))
        diag = SOLVER.diagnostics(mech, states)
        assert diag["branch_flip"] is False
        assert 0.0 < diag["transmission_angle_min_deg"] <= diag["transmission_angle_max_deg"]
        assert diag["toggle_margin_mm"] > 0.0


# ---------------------------------------------------------- property-based


@settings(max_examples=150, deadline=None)
@given(
    ground=st.floats(20.0, 90.0),
    crank=st.floats(8.0, 40.0),
    coupler=st.floats(20.0, 90.0),
    output=st.floats(20.0, 90.0),
    theta2_deg=st.floats(0.0, 359.0),
    branch=st.sampled_from([1, -1]),
)
def test_loop_closure_residual_is_zero_whenever_it_assembles(
    ground, crank, coupler, output, theta2_deg, branch
):
    """Whatever assembles must reproduce the link lengths exactly."""
    lengths = FourBarLengths(ground=ground, input=crank, coupler=coupler, output=output)
    theta2 = np.radians(theta2_deg)
    pos_a = np.zeros(2)
    pos_d = np.array([ground, 0.0])
    pos_b = pos_a + crank * np.array([np.cos(theta2), np.sin(theta2)])
    pos_c = solve_dyad(pos_b, coupler, pos_d, output, branch=branch)
    assume(pos_c is not None)

    assert np.linalg.norm(pos_b - pos_a) == pytest.approx(lengths.input, rel=1e-12)
    assert np.linalg.norm(pos_c - pos_b) == pytest.approx(lengths.coupler, rel=1e-9)
    assert np.linalg.norm(pos_d - pos_c) == pytest.approx(lengths.output, rel=1e-9)
    assert np.linalg.norm(pos_d - pos_a) == pytest.approx(lengths.ground, rel=1e-12)


@settings(max_examples=60, deadline=None)
@given(
    ground=st.floats(30.0, 80.0),
    crank=st.floats(10.0, 25.0),
    coupler=st.floats(30.0, 80.0),
    output=st.floats(30.0, 80.0),
    start_deg=st.floats(20.0, 80.0),
    span_deg=st.floats(5.0, 40.0),
)
def test_sweep_stays_on_one_branch(ground, crank, coupler, output, start_deg, span_deg):
    """A swept path must not jump between assembly modes."""
    try:
        mech = Linkage.four_bar(
            ground_mm=ground,
            input_mm=crank,
            coupler_mm=coupler,
            output_mm=output,
            input_angle_deg=start_deg,
            input_range_deg=(start_deg, start_deg + span_deg),
            branch=1,
        )
    except Exception:
        assume(False)
        return

    try:
        states = SOLVER.solve(mech, np.radians(np.linspace(start_deg, start_deg + span_deg, 25)))
    except AssemblyError:
        assume(False)
        return

    lengths = link_lengths_of(mech)
    for state in states:
        assert state.branch == 1
        assert (
            dyad_branch(
                state.joint_positions_mm["B"],
                state.joint_positions_mm["D"],
                state.joint_positions_mm["C"],
            )
            == 1
        )
        assert np.linalg.norm(
            state.joint_positions_mm["C"] - state.joint_positions_mm["B"]
        ) == pytest.approx(lengths.coupler, rel=1e-9)
