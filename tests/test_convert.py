"""Tests for arc fitting, conversion, feasibility reporting and the design search."""

from __future__ import annotations

import numpy as np
import pytest

from cmtool import Linkage, convert
from cmtool.convert.arc import ArcFitError, fit_input_arc, joint_excursions_deg
from cmtool.convert.search import (
    evaluate,
    reference_transmission_angle_deg,
    sampling_link_floor_mm,
    search,
    static_footprint_mm,
)
from cmtool.materials import Material, Printer

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")


@pytest.fixture
def pilot() -> Linkage:
    """Return the A1 pilot four-bar: a crank-rocker with a short 18 mm input link."""
    return Linkage.from_json("examples/fourbar.json")


@pytest.fixture
def buildable() -> Linkage:
    """Return a four-bar with comparable link lengths, which a flexure version needs."""
    return Linkage.four_bar(
        ground_mm=100.0,
        input_mm=95.0,
        coupler_mm=105.0,
        output_mm=98.0,
        input_angle_deg=95.0,
        input_range_deg=(85.0, 105.0),
        name="buildable",
    )


class TestArcFitting:
    def test_fitted_arc_meets_the_target(self, pilot):
        fit = fit_input_arc(pilot, max_joint_excursion_deg=22.0)
        assert fit.max_excursion_deg <= 22.0 + 1e-6

    def test_arc_is_centred_on_the_linkage_arc_midpoint(self, pilot):
        fit = fit_input_arc(pilot, max_joint_excursion_deg=22.0)
        assert fit.centre_deg == pytest.approx(70.0)
        assert fit.input_range_deg[0] == pytest.approx(70.0 - fit.half_width_deg)
        assert fit.input_range_deg[1] == pytest.approx(70.0 + fit.half_width_deg)

    def test_tighter_target_gives_a_narrower_arc(self, pilot):
        wide = fit_input_arc(pilot, max_joint_excursion_deg=22.0)
        narrow = fit_input_arc(pilot, max_joint_excursion_deg=10.0)
        assert narrow.half_width_deg < wide.half_width_deg

    def test_four_bar_amplifies_rotation_at_some_joint(self, pilot):
        """Joint excursions exceed the input excursion, which is why arcs shrink."""
        fit = fit_input_arc(pilot, max_joint_excursion_deg=22.0)
        assert fit.amplification > 1.0
        assert fit.binding_joint in pilot.joints

    def test_explicit_centre_is_honoured(self, pilot):
        fit = fit_input_arc(pilot, max_joint_excursion_deg=15.0, centre_deg=80.0)
        assert fit.centre_deg == pytest.approx(80.0)

    def test_unreachable_centre_raises(self):
        mech = Linkage.four_bar(
            ground_mm=4.0,
            input_mm=5.0,
            coupler_mm=6.0,
            output_mm=10.0,
            input_angle_deg=90.0,
            input_range_deg=(85.0, 95.0),
        )
        with pytest.raises(ArcFitError, match=r"does not assemble|toggle"):
            fit_input_arc(mech, max_joint_excursion_deg=20.0, centre_deg=0.0)

    def test_non_positive_target_refused(self, pilot):
        with pytest.raises(ValueError, match="must be positive"):
            fit_input_arc(pilot, max_joint_excursion_deg=0.0)

    def test_unsweepable_arc_returns_none(self):
        mech = Linkage.four_bar(
            ground_mm=4.0,
            input_mm=5.0,
            coupler_mm=6.0,
            output_mm=10.0,
            input_angle_deg=90.0,
            input_range_deg=(85.0, 95.0),
        )
        assert joint_excursions_deg(mech, (0.0, 10.0)) is None

    def test_to_dict_is_serialisable(self, pilot):
        import json

        json.dumps(fit_input_arc(pilot, max_joint_excursion_deg=20.0).to_dict())


class TestConversion:
    def test_pilot_is_feasible_but_leaves_the_small_length_regime(self, pilot):
        """The A1 pilot builds fine; it just needs the long-segment PRBM model.

        It used to be reported infeasible, but only because PRBM validity was
        acting as a filter. Validity is metadata now, so the design is kept and
        the model switches instead -- which is exactly the kind of sample the
        fidelity map needs.
        """
        fit = fit_input_arc(pilot, max_joint_excursion_deg=22.0)
        mech = convert(pilot, input_range_deg=fit.input_range_deg, thickness_mm=0.6)
        assert mech.feasibility.feasible
        assert not mech.feasibility.all_small_length
        assert "long_segment" in mech.feasibility.prbm_models.values()
        assert any("beam FEA is the reference" in n for n in mech.feasibility.prbm_notes())

    def test_prbm_validity_never_makes_a_design_infeasible(self, pilot):
        """The rule, stated as a test: only strain and geometry can reject a joint."""
        fit = fit_input_arc(pilot, max_joint_excursion_deg=22.0)
        mech = convert(pilot, input_range_deg=fit.input_range_deg, thickness_mm=0.6)
        for sized in mech.sizing.values():
            assert sized.feasible == (sized.strain_ok and sized.fits_geometrically)

    def test_infeasible_when_strain_needs_more_flexure_than_fits(self):
        """The one way a joint fails now: no length satisfies strain and still fits."""
        small = Linkage.four_bar(
            ground_mm=40.0,
            input_mm=36.0,
            coupler_mm=44.0,
            output_mm=38.0,
            input_angle_deg=90.0,
            input_range_deg=(60.0, 120.0),
            name="small_wide",
        )
        mech = convert(small, thickness_mm=2.0)
        assert not mech.feasibility.feasible
        binding = mech.sizing[mech.feasibility.binding_joint]
        assert not binding.fits_geometrically
        assert "fits on the" in (binding.limit_reason or "")

    def test_buildable_design_is_feasible(self, buildable):
        mech = convert(buildable, thickness_mm=0.6)
        assert mech.feasibility.feasible
        assert all(s.feasible for s in mech.sizing.values())
        assert mech.feasibility.max_utilisation <= 1.0

    def test_mid_arc_printing_roughly_halves_the_worst_bend(self, buildable):
        mid = convert(buildable, unstressed_at="mid_arc", thickness_mm=0.6)
        start = convert(buildable, unstressed_at="start", thickness_mm=0.6)
        worst_mid = max(s.max_bend_deg for s in mid.sizing.values())
        worst_start = max(s.max_bend_deg for s in start.sizing.values())
        assert 0.4 < worst_mid / worst_start < 0.6

    def test_mid_arc_printing_lowers_utilisation(self, buildable):
        mid = convert(buildable, unstressed_at="mid_arc", thickness_mm=0.6)
        start = convert(buildable, unstressed_at="start", thickness_mm=0.6)
        assert mid.feasibility.max_utilisation < start.feasibility.max_utilisation

    def test_excursion_is_the_same_whichever_configuration_is_printed(self, buildable):
        """Peak-to-peak travel is a property of the arc, not of the print pose."""
        mid = convert(buildable, unstressed_at="mid_arc", thickness_mm=0.6)
        start = convert(buildable, unstressed_at="start", thickness_mm=0.6)
        for name in mid.sizing:
            assert mid.sizing[name].excursion_deg == pytest.approx(
                start.sizing[name].excursion_deg, rel=1e-9
            )

    def test_chosen_length_meets_the_allowable_strain_with_the_safety_factor(self, buildable):
        mech = convert(buildable, thickness_mm=0.6)
        allowable = mech.feasibility.allowable_strain
        safety = mech.feasibility.strain_safety_factor
        for sized in mech.sizing.values():
            if sized.geometry.length_mm > sized.min_length_strain_mm + 1e-9:
                continue  # a length floor was applied instead
            assert sized.strain.peak_strain <= allowable / safety + 1e-12

    def test_explicit_flexure_length_is_honoured(self, buildable):
        mech = convert(buildable, thickness_mm=0.6, flexure_length_mm=5.0)
        assert all(s.geometry.length_mm == pytest.approx(5.0) for s in mech.sizing.values())

    def test_per_joint_flexure_lengths(self, buildable):
        mech = convert(buildage := buildable, thickness_mm=0.6, flexure_length_mm={"A": 4.0})
        assert mech.sizing["A"].geometry.length_mm == pytest.approx(4.0)
        assert mech.sizing["B"].geometry.length_mm != pytest.approx(4.0)
        assert buildage is buildable

    def test_thinner_flexures_lower_utilisation(self, buildable):
        thin = convert(buildable, thickness_mm=0.4)
        thick = convert(buildable, thickness_mm=0.8)
        assert thin.feasibility.max_utilisation < thick.feasibility.max_utilisation

    def test_short_flexure_length_floor_is_applied(self, buildable):
        """A near-zero bend must not produce an unprintable sub-millimetre flexure."""
        mech = convert(buildable, thickness_mm=0.6, input_range_deg=(94.9, 95.1))
        assert all(s.geometry.length_mm >= 1.0 for s in mech.sizing.values())

    def test_uses_placeholder_material_data_and_says_so(self, buildable):
        mech = convert(buildable, thickness_mm=0.6)
        assert not mech.is_physical
        assert "PLA.properties.allowable_strain" in mech.provenance.placeholders_used
        assert "not a physical prediction" in (mech.provenance.caveat() or "").lower()

    def test_default_thickness_comes_from_the_printer_and_is_a_placeholder(self, buildable):
        mech = convert(buildable)
        assert "bambu_a1.design_rules.min_flexure_thickness_mm" in (
            mech.provenance.placeholders_used
        )

    def test_missing_arc_is_an_error(self):
        mech = Linkage.four_bar(ground_mm=100.0, input_mm=95.0, coupler_mm=105.0, output_mm=98.0)
        with pytest.raises(ValueError, match="no input arc"):
            convert(mech)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"placement": "nonsense"}, "unknown placement"),
            ({"unstressed_at": "nonsense"}, "unknown unstressed_at"),
            ({"n_steps": 2}, "at least 3"),
        ],
    )
    def test_invalid_options_refused(self, buildable, kwargs, match):
        with pytest.raises(ValueError, match=match):
            convert(buildable, thickness_mm=0.6, **kwargs)

    def test_mixed_flexure_types_are_refused_for_now(self, buildable):
        with pytest.raises(NotImplementedError, match="Phase B"):
            convert(buildable, flexures={"A": "small_length_pivot", "B": "notch"})

    def test_summary_is_serialisable(self, buildable):
        import json

        json.dumps(convert(buildable, thickness_mm=0.6).summary(), default=str)


class TestPlacement:
    def test_pivot_matched_leaves_the_geometry_unchanged(self, buildable):
        mech = convert(buildable, thickness_mm=0.6, placement="pivot_matched")
        effective = mech.effective_linkage()
        for name, joint in buildable.joints.items():
            np.testing.assert_allclose(
                effective.joints[name].position_mm, joint.position_mm, atol=1e-12
            )

    def test_unmatched_displaces_every_pivot_by_half_a_flexure(self, buildable):
        mech = convert(buildable, thickness_mm=0.6, placement="unmatched")
        effective = mech.effective_linkage()
        for name, sized in mech.sizing.items():
            moved = np.linalg.norm(
                effective.joints[name].position_mm - buildable.joints[name].position_mm
            )
            assert moved == pytest.approx(sized.geometry.length_mm / 2.0, rel=1e-9)

    def test_unmatched_changes_the_effective_link_lengths(self, buildable):
        """The conversion artefact that pivot matching exists to remove."""
        mech = convert(buildable, thickness_mm=0.6, placement="unmatched")
        effective = mech.effective_linkage()
        changed = [
            body
            for body in buildable.bodies
            if abs(effective.link_length(body) - buildable.link_length(body)) > 1e-6
        ]
        assert changed

    def test_effective_linkage_marks_joints_as_flexures(self, buildable):
        effective = convert(buildable, thickness_mm=0.6).effective_linkage()
        assert all(j.kind == "flexure" for j in effective.joints.values())

    def test_flexure_joints_keep_mobility_one(self, buildable):
        """A compliant pivot still removes two DOF; the mechanism is still 1-DOF."""
        assert convert(buildable, thickness_mm=0.6).effective_linkage().mobility() == 1


class TestFeasibilityReport:
    def test_binding_joint_has_the_highest_utilisation(self, buildable):
        report = convert(buildable, thickness_mm=0.6).feasibility
        binding = report.binding_joint
        assert all(
            report.joints[binding].utilisation >= sized.utilisation
            for sized in report.joints.values()
        )

    def test_reasons_are_empty_when_feasible(self, buildable):
        assert convert(buildable, thickness_mm=0.6).feasibility.reasons() == []

    def test_reasons_name_every_failing_joint(self):
        small = Linkage.four_bar(
            ground_mm=40.0,
            input_mm=36.0,
            coupler_mm=44.0,
            output_mm=38.0,
            input_angle_deg=90.0,
            input_range_deg=(60.0, 120.0),
            name="small_wide",
        )
        report = convert(small, thickness_mm=2.0).feasibility
        failing = [name for name, s in report.joints.items() if not s.feasible]
        assert failing
        assert len(report.reasons()) == len(failing)

    def test_prbm_notes_are_reported_separately_from_failures(self, pilot):
        """Model caveats must not read as feasibility problems."""
        fit = fit_input_arc(pilot, max_joint_excursion_deg=22.0)
        report = convert(pilot, input_range_deg=fit.input_range_deg, thickness_mm=0.6).feasibility
        assert report.feasible
        assert report.reasons() == []
        assert report.prbm_notes()

    def test_to_dict_round_trips(self, buildable):
        import json

        payload = json.loads(json.dumps(convert(buildable, thickness_mm=0.6).feasibility.to_dict()))
        assert set(payload["joints"]) == set(buildable.joints)


# The search sweeps many candidate linkages, so it runs once and every assertion
# below reads the same result.
@pytest.fixture(scope="module")
def searched():
    return search(n_candidates=60, seed=1, keep=3, thickness_mm=0.6)


class TestDesignSearch:
    def test_search_finds_feasible_designs(self, searched):
        assert searched.kept
        assert all(c.ok and c.compliant is not None for c in searched.kept)
        assert all(c.compliant.feasibility.feasible for c in searched.kept)
        assert searched.n_rejected > 0

    @pytest.mark.slow
    def test_search_is_deterministic_for_a_seed(self):
        first = search(n_candidates=40, seed=3, keep=2, thickness_mm=0.6)
        second = search(n_candidates=40, seed=3, keep=2, thickness_mm=0.6)
        assert [c.linkage.name for c in first.kept] == [c.linkage.name for c in second.kept]

    def test_kept_designs_meet_the_excursion_target(self, searched):
        for candidate in searched.kept:
            assert max(candidate.excursions_deg.values()) <= 22.0 + 1e-6

    def test_kept_designs_fit_the_envelope(self, searched):
        envelope = Printer.load("bambu_a1").design_envelope_mm()
        for candidate in searched.kept:
            assert candidate.footprint_mm[0] <= envelope[0]
            assert candidate.footprint_mm[1] <= envelope[1]

    def test_kept_designs_have_links_of_comparable_length(self, searched):
        """The governing bound rules out short links, so the search must avoid them."""
        for candidate in searched.kept:
            lengths = [candidate.linkage.link_length(body) for body in candidate.linkage.bodies]
            assert min(lengths) >= searched.link_floor_mm - 1e-9

    def test_rejection_codes_are_stable_identifiers(self, searched):
        assert all(" " not in code for code in searched.reasons)

    def test_link_floor_comes_from_the_closed_form_bound(self, searched):
        """Sampling below the bound only generates provably infeasible designs."""
        expected = sampling_link_floor_mm(22.0, thickness_mm=0.6)
        assert searched.link_floor_mm >= expected - 1e-9
        assert searched.link_floor_mm > 0.0

    def test_dropping_prbm_as_a_filter_relaxed_the_link_floor(self):
        """The geometric cap is 8x the old small-length ratio, so the floor fell 8x."""
        geometric = sampling_link_floor_mm(22.0, thickness_mm=0.6)
        old_rule = sampling_link_floor_mm(22.0, thickness_mm=0.6, max_length_fraction=0.1)
        assert old_rule / geometric == pytest.approx(8.0, rel=1e-9)

    def test_search_keeps_designs_outside_the_small_length_regime(self, searched):
        """They are kept on purpose: the fidelity map needs them."""
        for candidate in searched.kept:
            assert candidate.compliant.feasibility.feasible
            for sized in candidate.compliant.sizing.values():
                assert sized.validity.model in {"small_length", "long_segment"}

    def test_report_is_serialisable(self, searched):
        import json

        payload = json.loads(json.dumps(searched.to_dict(), default=str))
        assert payload["n_kept"] == len(searched.kept)

    def test_pilot_now_passes_flexure_feasibility(self, pilot):
        """Whatever else rejects it, it is no longer the flexures."""
        candidate = evaluate(pilot, thickness_mm=0.6)
        code = candidate.rejected_code or ""
        assert not code.startswith("flexure_infeasible_")

    def test_candidate_summary_is_serialisable(self, buildable):
        import json

        json.dumps(evaluate(buildable, thickness_mm=0.6).summary(), default=str)


class TestCheapPreChecks:
    """The pre-checks must only reject what the full evaluation would reject."""

    def test_static_footprint_bounds_the_swept_one(self, buildable):
        from cmtool.convert.search import swept_footprint_mm

        static = static_footprint_mm(buildable)
        swept = swept_footprint_mm(buildable, buildable.input_range_deg)
        assert static[0] <= swept[0] + 1e-9
        assert static[1] <= swept[1] + 1e-9

    def test_reference_transmission_angle_is_inside_the_swept_range(self, buildable):
        from cmtool.convert.arc import sweep

        reference = reference_transmission_angle_deg(buildable)
        result = sweep(buildable, buildable.input_range_deg)
        assert result.diagnostics["transmission_angle_min_deg"] - 1e-6 <= reference
        assert reference <= result.diagnostics["transmission_angle_max_deg"] + 1e-6

    def test_transmission_angle_is_none_for_a_non_four_bar(self):
        from tests.test_graph import make_four_bar

        chain = make_four_bar()
        del chain.joints["D"]
        del chain.bodies["output"]
        assert reference_transmission_angle_deg(chain) is None


class TestConfigLoading:
    def test_material_loads_with_supplier_metadata(self):
        material = Material.load("PLA")
        assert material.name == "PLA"
        assert material.supplier["brand"] == "eSun"

    def test_material_properties_are_all_placeholders_for_now(self):
        unresolved = Material.load("PLA").unresolved()
        assert "PLA.properties.youngs_modulus_MPa" in unresolved
        assert "PLA.properties.allowable_strain" in unresolved

    def test_primary_printer_is_the_bambu_a1(self):
        printer = Printer.load("bambu_a1")
        assert printer.model == "Bambu Lab A1"
        assert printer.role == "primary"
        assert printer.nozzle_mm() == pytest.approx(0.4)
        assert printer.layer_height_mm() == pytest.approx(0.2)

    def test_secondary_printer_is_recorded_but_flagged(self):
        printer = Printer.load("kobra2_neo")
        assert printer.role == "secondary"
        assert "min_flexure_thickness_mm" in " ".join(printer.unresolved())

    def test_both_printers_share_the_design_envelope(self):
        """Parts must fit both beds, so a design moves between printers unchanged."""
        assert Printer.load("bambu_a1").design_envelope_mm() == (180.0, 180.0)
        assert Printer.load("kobra2_neo").design_envelope_mm() == (180.0, 180.0)

    def test_slicer_guidance_names_arachne(self):
        """The setting that makes sub-2-perimeter flexures printable at all."""
        settings = Printer.load("bambu_a1").slicer["settings"]
        assert settings["wall_generator"]["value"] == "Arachne"
        assert settings["sparse_infill_density_percent"]["value"] == 100

    def test_unknown_config_name_lists_alternatives(self):
        from cmtool.core.config import ConfigError

        with pytest.raises(ConfigError, match="available:"):
            Printer.load("does_not_exist")
