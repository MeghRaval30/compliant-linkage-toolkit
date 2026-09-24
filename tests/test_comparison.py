"""Tests for the demo pair's comparison sheet.

Mostly about the cells that are deliberately empty. A sheet like this is read by
someone who was not in the room, so a blank that looks like an oversight is worse
than no sheet at all -- and a number where there should be a blank is worse
still.
"""

from __future__ import annotations

import pytest

from cmtool import Linkage, convert
from cmtool.metrics.comparison import build_comparison, write_comparison
from cmtool.viz.scene import build_scene

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")


@pytest.fixture(scope="module")
def pieces():
    linkage = Linkage.four_bar(
        ground_mm=40.0,
        input_mm=32.0,
        coupler_mm=44.0,
        output_mm=36.0,
        coupler_point_mm=(22.0, 38.0),
        input_angle_deg=90.0,
        input_range_deg=(79.013, 100.987),
        name="demo_pair",
    )
    mechanism = convert(linkage, input_range_deg=linkage.input_range_deg, thickness_mm=0.6)
    scene = build_scene(mechanism, n_steps=5)
    return linkage, mechanism, scene


@pytest.fixture(scope="module")
def sheet(pieces):
    from cmtool.cad.rigid import RigidCadSpec, build_rigid_mechanism

    linkage, mechanism, scene = pieces
    _, layout = build_rigid_mechanism(linkage, spec=RigidCadSpec(clearance_mm=0.35))
    return build_comparison(scene, mechanism, rigid_layout=layout, clearance_mm=0.35)


def cell(sheet, label):
    return next(row for row in sheet.rows if row.label.startswith(label))


class TestWhatIsNotClaimed:
    def test_the_rigid_part_s_torque_is_not_predicted(self, sheet):
        """There is no friction model, so claiming a number would claim the result."""
        row = cell(sheet, "peak input torque")
        assert row.rigid == "not predicted"
        assert "PRBM" in row.compliant

    def test_the_blanks_are_listed_with_reasons(self, sheet):
        assert len(sheet.unmeasured) >= 3
        assert any("friction" in item for item in sheet.unmeasured)
        assert any("printed" in item for item in sheet.unmeasured)

    def test_the_rigid_path_figure_is_a_bound_not_a_prediction(self, sheet):
        """A pin can sit anywhere in its hole; that is a bound, and it says so."""
        row = cell(sheet, "coupler path")
        assert "0.35" in row.rigid
        assert "bound" in row.note

    def test_the_placeholder_caveat_is_carried(self, sheet):
        assert sheet.caveat
        assert any("allowable_strain" in p for p in sheet.placeholders)

    def test_the_strain_row_flags_the_placeholder_allowable(self, sheet):
        assert "PLACEHOLDER" in cell(sheet, "flexure strain margin").compliant


class TestWhatIsClaimed:
    def test_counts_bodies_and_fasteners(self, sheet):
        assert cell(sheet, "printed bodies").rigid == "1"
        assert cell(sheet, "fasteners").rigid == "0"
        assert cell(sheet, "fasteners").compliant == "0"

    def test_states_the_compliant_range_limit(self, sheet):
        """The honest cost of the technology, and it belongs on the sheet."""
        row = cell(sheet, "range of motion")
        assert "22.0 deg" in row.compliant
        assert "continuous" in row.rigid

    def test_the_strain_margin_matches_the_scene(self, sheet, pieces):
        _, mechanism, scene = pieces
        fea = scene.models["fea"]
        peak = max(max(max(frame.flexure_strain[j]) for frame in fea.frames) for j in scene.joints)
        expected = mechanism.feasibility.allowable_strain / peak
        assert f"{expected:.2f}x" in cell(sheet, "flexure strain margin").compliant


class TestRendering:
    def test_markdown_has_the_table_and_the_blanks(self, sheet):
        text = sheet.to_markdown()
        assert "| **printed bodies** |" in text
        assert "## Deliberately blank" in text
        assert "NOT A PHYSICAL PREDICTION" in text

    def test_writes_to_disk(self, sheet, tmp_path):
        path = write_comparison(sheet, tmp_path / "nested" / "sheet.md")
        assert path.exists()
        assert path.read_text(encoding="utf-8").startswith("# demo_pair")

    def test_is_json_serialisable(self, sheet):
        import json

        json.dumps(sheet.to_dict(), allow_nan=False)

    def test_works_without_the_rigid_half(self, pieces):
        """The compliant side alone still renders, with the rigid column blank."""
        _, mechanism, scene = pieces
        sheet = build_comparison(scene, mechanism)
        assert cell(sheet, "printed bodies").rigid == "-"
        assert "| **fasteners** | - | 0 |" in sheet.to_markdown()
