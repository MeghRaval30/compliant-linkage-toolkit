"""Tests for coupon geometry, printability checks, export and print sheets."""

from __future__ import annotations

import pytest

from cmtool.cad import (
    CantileverCouponSpec,
    CouponSet,
    FlexureCouponSpec,
    bounding_box_mm,
    build_cantilever_coupon,
    build_flexure_coupon,
    check_printability,
    export_solid,
    write_print_sheet,
)
from cmtool.materials import Material, Printer

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")


@pytest.fixture(scope="module")
def printer() -> Printer:
    return Printer.load("bambu_a1")


@pytest.fixture(scope="module")
def material() -> Material:
    return Material.load("PLA")


@pytest.fixture(scope="module")
def flexure_solid():
    return build_flexure_coupon()


@pytest.fixture(scope="module")
def cantilever_solid():
    return build_cantilever_coupon()


class TestFlexureCoupon:
    def test_default_thicknesses_span_the_nozzle_width(self):
        spec = FlexureCouponSpec()
        assert spec.thicknesses_mm[0] == pytest.approx(0.4)
        assert min(spec.thicknesses_mm) <= 0.4
        assert max(spec.thicknesses_mm) >= 1.0

    def test_bounding_box_matches_the_spec(self, flexure_solid):
        spec = FlexureCouponSpec()
        extent = bounding_box_mm(flexure_solid)
        assert extent[0] == pytest.approx(spec.total_width_mm)
        assert extent[1] == pytest.approx(spec.total_height_mm)
        assert extent[2] == pytest.approx(spec.part_thickness_mm)

    def test_fits_the_print_envelope(self, flexure_solid, printer):
        extent = bounding_box_mm(flexure_solid)
        envelope = printer.design_envelope_mm()
        assert extent[0] <= envelope[0]
        assert extent[1] <= envelope[1]

    def test_identification_is_one_pip_per_index(self):
        pips = FlexureCouponSpec().identification()
        assert pips[0.4] == 1
        assert pips[1.0] == 5
        assert sorted(pips.values()) == [1, 2, 3, 4, 5]

    def test_is_one_connected_solid(self, flexure_solid):
        """Rail, strips and paddles must fuse, or strips print as loose parts."""
        assert len(flexure_solid.vals()[0].Solids()) == 1

    def test_custom_spec_changes_the_geometry(self):
        spec = FlexureCouponSpec(thicknesses_mm=(0.5, 0.7), pitch_mm=25.0)
        extent = bounding_box_mm(build_flexure_coupon(spec))
        assert extent[0] == pytest.approx(50.0)


class TestCantileverCoupon:
    def test_strip_count_and_layout(self):
        spec = CantileverCouponSpec()
        assert spec.strip_count == 6
        assert len(spec.layout()) == 6
        assert {t for _, t, _ in spec.layout()} == {1.0, 2.0}

    def test_two_thicknesses_so_wall_structure_can_be_compared(self):
        """A 1 mm strip is nearly all perimeter; a 2 mm strip has infill."""
        assert len(CantileverCouponSpec().thicknesses_mm) >= 2

    def test_strips_stay_separate_solids(self, cantilever_solid):
        assert len(cantilever_solid.vals()[0].Solids()) == 6

    def test_bounding_box_covers_every_strip(self, cantilever_solid):
        spec = CantileverCouponSpec()
        extent = bounding_box_mm(cantilever_solid)
        assert extent[0] == pytest.approx(spec.total_width_mm)
        assert extent[1] == pytest.approx(spec.length_mm)
        assert extent[2] == pytest.approx(max(spec.thicknesses_mm))

    def test_fits_the_print_envelope(self, cantilever_solid, printer):
        extent = bounding_box_mm(cantilever_solid)
        envelope = printer.design_envelope_mm()
        assert extent[0] <= envelope[0] and extent[1] <= envelope[1]


class TestPrintability:
    def test_flexure_coupon_passes_with_an_intentional_below_minimum_note(
        self, flexure_solid, printer
    ):
        check = check_printability(
            flexure_solid, printer, min_feature_mm=0.4, allow_below_minimum=True
        )
        assert check.ok
        assert check.below_minimum
        assert any("intentional" in note for note in check.notes)

    def test_below_minimum_fails_when_not_allowed(self, flexure_solid, printer):
        check = check_printability(
            flexure_solid, printer, min_feature_mm=0.4, allow_below_minimum=False
        )
        assert not check.ok

    def test_oversized_part_fails_the_envelope(self, printer):
        import cadquery as cq

        big = cq.Workplane("XY").box(300.0, 50.0, 6.0)
        check = check_printability(big, printer)
        assert not check.fits_envelope
        assert any("larger than" in note for note in check.notes)

    def test_placeholder_minimum_is_recorded_in_provenance(self, flexure_solid, printer):
        from cmtool.core.provenance import Provenance

        prov = Provenance()
        check_printability(flexure_solid, printer, min_feature_mm=0.4, provenance=prov)
        assert "bambu_a1.design_rules.min_flexure_thickness_mm" in prov.placeholders_used
        assert not prov.is_physical

    def test_to_dict_is_serialisable(self, flexure_solid, printer):
        import json

        json.dumps(check_printability(flexure_solid, printer, min_feature_mm=0.4).to_dict())


class TestExport:
    def test_writes_step_and_stl(self, tmp_path, flexure_solid):
        paths = export_solid(flexure_solid, tmp_path, "coupon")
        assert set(paths) == {"step", "stl"}
        for path in paths.values():
            assert path.is_file() and path.stat().st_size > 0

    def test_stl_is_a_wellformed_binary_mesh(self, tmp_path, cantilever_solid):
        """Binary STL: 80-byte header, 4-byte triangle count, then 50 bytes each."""
        import struct

        path = export_solid(cantilever_solid, tmp_path, "strips", formats=("stl",))["stl"]
        data = path.read_bytes()
        triangles = struct.unpack("<I", data[80:84])[0]
        assert triangles > 0
        assert len(data) == 84 + 50 * triangles
        # Six box-shaped strips: 12 triangles each at minimum.
        assert triangles >= 6 * 12

    def test_step_carries_a_step_header(self, tmp_path, flexure_solid):
        path = export_solid(flexure_solid, tmp_path, "coupon", formats=("step",))["step"]
        assert path.read_text(encoding="utf-8", errors="ignore").startswith("ISO-10303-21")

    def test_unknown_format_refused(self, tmp_path, flexure_solid):
        with pytest.raises(ValueError, match="unsupported export format"):
            export_solid(flexure_solid, tmp_path, "coupon", formats=("obj",))


class TestPrintSheet:
    @pytest.fixture
    def sheet_text(self, tmp_path, flexure_solid, printer, material) -> str:
        check = check_printability(
            flexure_solid, printer, min_feature_mm=0.4, allow_below_minimum=True
        )
        path = write_print_sheet(
            tmp_path / "sheet.md",
            title="flexure coupon",
            printer=printer,
            material=material,
            purpose="Find the minimum printable flexure thickness.",
            check=check,
            details={"thicknesses_mm": [0.4, 0.5, 0.6, 0.8, 1.0]},
            instructions=["Print flat on the bed.", "Measure every strip."],
        )
        return path.read_text(encoding="utf-8")

    def test_names_the_printer_used(self, sheet_text):
        assert "Bambu Lab A1" in sheet_text
        assert "bambu_a1" in sheet_text

    def test_carries_the_slicer_settings(self, sheet_text):
        assert "Arachne" in sheet_text
        assert "Bambu Studio" in sheet_text

    def test_flags_settings_that_still_need_values(self, sheet_text):
        assert "_to be set_" in sheet_text

    def test_includes_the_metadata_checklist(self, sheet_text):
        for field in ("Lot number", "Room temperature", "Print date", "Printer used"):
            assert field in sheet_text

    def test_includes_the_instructions(self, sheet_text):
        assert "1. Print flat on the bed." in sheet_text

    def test_records_the_code_commit(self, sheet_text):
        assert "Generated by cmtool at commit" in sheet_text


class TestCouponSet:
    def test_builds_both_coupons(self):
        solids = CouponSet().build()
        assert set(solids) == {"flexure_coupon", "cantilever_coupon"}

    def test_metadata_is_serialisable_and_complete(self):
        import json

        meta = json.loads(json.dumps(CouponSet().metadata()))
        assert meta["flexure_coupon"]["thicknesses_mm"] == [0.4, 0.5, 0.6, 0.8, 1.0]
        assert meta["cantilever_coupon"]["strip_count"] == 6
