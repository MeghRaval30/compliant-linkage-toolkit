"""Tests for the scene model, the self-contained HTML viewer and the figures.

Three things these exist to protect, in order of how badly they would hurt:

1. **The viewer must not need the network.** A file that silently pulls a CDN is
   useless at a bench, on a phone with no signal, or five years from now.
   ``test_has_no_external_references`` fails the build over a single URL.
2. **Missing data must stay missing.** No figure and no viewer may invent a
   measured series, and the absence has to be visible rather than silent.
3. **The drawn geometry must be the solved geometry.** The flexure polylines are
   checked against attachment points computed independently of the solver, not
   against a stored picture.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cmtool import Linkage, convert
from cmtool.convert.placement import attachment_points
from cmtool.viz.html import external_references, render_html, write_html
from cmtool.viz.palette import SERIES, STATUS, STRAIN_RAMP_LIGHT, strain_colour
from cmtool.viz.scene import build_scene

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")

STEPS = 5


@pytest.fixture(scope="module")
def mechanism():
    linkage = Linkage.four_bar(
        ground_mm=60.0,
        input_mm=55.0,
        coupler_mm=62.0,
        output_mm=58.0,
        coupler_point_mm=(35.0, 70.0),
        input_angle_deg=95.0,
        input_range_deg=(88.0, 102.0),
        name="viz_demo",
    )
    return convert(linkage, thickness_mm=0.6)


@pytest.fixture(scope="module")
def scene(mechanism):
    return build_scene(mechanism, n_steps=STEPS)


@pytest.fixture(scope="module")
def unstressed_frame(mechanism):
    """Return the FEA frame at the **as-printed** angle, where nothing is bent.

    Not the first frame of a sweep: the part is printed unstressed at mid-arc, so
    the sweep starts with every flexure already bent through half its excursion.
    """
    reference = mechanism.reference_input_deg
    scene = build_scene(
        mechanism,
        input_angles_deg=np.array([reference, reference + 1.0]),
        include=("fea",),
    )
    return scene.models["fea"].frames[0]


class TestSceneGeometry:
    def test_every_model_is_solved_at_the_same_angles(self, scene):
        """One slider drives all three, and a pointwise comparison needs it."""
        for series in scene.models.values():
            assert series.input_angles_deg == scene.input_angles_deg

    def test_flexure_polylines_start_and_end_at_the_attachment_points(
        self, unstressed_frame, mechanism
    ):
        """The drawn flexure spans what the placement rule says it spans.

        Checked against :func:`attachment_points` computed here, not against the
        mesh the solver built, so a placement change cannot pass by moving both
        sides at once. At the as-printed angle the part is undeformed, so the
        drawn geometry is the nominal geometry.
        """
        attachments, _ = attachment_points(mechanism)
        for joint_name, joint in mechanism.base.joints.items():
            host = mechanism.sizing[joint_name].host_body
            other = joint.other(host)
            polyline = np.asarray(unstressed_frame.flexures[joint_name], dtype=float)
            assert polyline[0] == pytest.approx(attachments[host][joint_name], abs=1e-6)
            assert polyline[-1] == pytest.approx(attachments[other][joint_name], abs=1e-6)

    def test_the_as_printed_flexure_is_straight_and_the_right_length(
        self, unstressed_frame, mechanism
    ):
        """Unstressed means a straight strip of exactly the sized length."""
        for joint_name, sizing in mechanism.sizing.items():
            polyline = np.asarray(unstressed_frame.flexures[joint_name], dtype=float)
            segments = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
            end_to_end = float(np.linalg.norm(polyline[-1] - polyline[0]))
            assert segments.sum() == pytest.approx(end_to_end, rel=1e-9)  # collinear
            assert end_to_end == pytest.approx(sizing.geometry.length_mm, rel=1e-6)

    def test_strain_has_one_value_per_element_not_per_point(self, scene):
        frame = scene.models["fea"].frames[0]
        for joint, polyline in frame.flexures.items():
            assert len(frame.flexure_strain[joint]) == len(polyline) - 1

    def test_the_as_printed_state_carries_no_strain(self, unstressed_frame):
        """The part is printed in this configuration, so nothing is bent yet."""
        for values in unstressed_frame.flexure_strain.values():
            assert max(values) == pytest.approx(0.0, abs=1e-9)

    def test_the_sweep_does_not_start_unstressed(self, scene, mechanism):
        """Printing at mid-arc is the point: the arc ends are where the bend is.

        A sweep that began unstressed would mean the part was printed at one end
        of its travel, which doubles the peak strain for nothing.
        """
        assert mechanism.unstressed_at == "mid_arc"
        first = scene.models["fea"].frames[0]
        assert max(max(v) for v in first.flexure_strain.values()) > 1e-4

    def test_ground_is_drawn_once_rather_than_per_frame(self, scene):
        assert len(scene.ground_polyline_mm) == 2

    def test_bounds_contain_every_drawn_point(self, scene):
        x0, y0, x1, y1 = scene.bounds_mm
        for series in scene.models.values():
            points = np.asarray(series.path_mm, dtype=float)
            assert points[:, 0].min() >= x0 and points[:, 0].max() <= x1
            assert points[:, 1].min() >= y0 and points[:, 1].max() <= y1


class TestSceneComparisons:
    @pytest.mark.validation
    def test_the_prbm_path_is_the_rigid_path_exactly(self, scene):
        """With a prescribed input, one DOF and no load, stiffness cannot move it.

        Pivot-matched placement puts the characteristic pivots back on the rigid
        joints, so the two are the same linkage. Anything but zero here means
        either the placement rule or the PRBM kinematics has drifted.
        """
        rigid = np.asarray(scene.models["rigid"].path_mm, dtype=float)
        prbm = np.asarray(scene.models["prbm"].path_mm, dtype=float)
        assert np.abs(rigid - prbm).max() == pytest.approx(0.0, abs=1e-9)
        assert max(scene.models["prbm"].deviation_mm) == pytest.approx(0.0, abs=1e-9)

    def test_the_fea_path_does_move(self, scene):
        """Distributed compliance is exactly what the PRBM cannot represent."""
        assert max(scene.models["fea"].deviation_mm) > 1e-4

    def test_the_reference_series_has_no_deviation_from_itself(self, scene):
        assert scene.deviation_reference == "rigid"
        assert max(scene.models["rigid"].deviation_mm) == pytest.approx(0.0, abs=1e-12)

    def test_frechet_never_exceeds_the_matched_pointwise_maximum(self, scene):
        for comparison in scene.comparisons:
            assert comparison["frechet_mm"] <= comparison["max_mm"] + 1e-9


class TestSceneHonesty:
    def test_placeholders_make_the_scene_not_a_physical_prediction(self, scene):
        assert not scene.provenance.is_physical
        assert scene.caveat is not None
        assert "allowable_strain" in scene.caveat

    def test_a_missing_measured_path_is_absent_and_said_so(self, scene):
        assert "measured" not in scene.models
        assert any("absent, not zero" in note for note in scene.notes)

    def test_the_strain_ramp_is_flagged_as_relative_while_unmeasured(self, scene):
        assert scene.allowable_strain_is_placeholder is True

    def test_a_measured_path_becomes_a_fourth_series(self, mechanism, tmp_path):
        from cmtool.metrics.paths import read_path_csv

        csv_path = tmp_path / "measured.csv"
        rows = ["frame,input_angle_deg,coupler_x_mm,coupler_y_mm"]
        for index, angle in enumerate(np.linspace(88.0, 102.0, 6)):
            rows.append(f"{index},{angle:.4f},{35.0 + index * 0.4:.4f},{70.0 - index * 0.2:.4f}")
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        scene = build_scene(
            mechanism, n_steps=STEPS, measured=read_path_csv(csv_path), include=("rigid",)
        )
        assert "measured" in scene.models
        assert not any("absent, not zero" in note for note in scene.notes)
        pairs = {(c["a"], c["b"]) for c in scene.comparisons}
        assert ("rigid", "measured") in pairs

    def test_a_measured_take_without_angles_is_compared_by_shape_only(self, mechanism, tmp_path):
        """No angles means no correspondence, and inventing one is not allowed."""
        from cmtool.metrics.paths import read_path_csv

        csv_path = tmp_path / "no_angles.csv"
        rows = ["frame,input_angle_deg,coupler_x_mm,coupler_y_mm"]
        for index in range(6):
            rows.append(f"{index},,{35.0 + index * 0.4:.4f},{70.0 - index * 0.2:.4f}")
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        scene = build_scene(
            mechanism, n_steps=STEPS, measured=read_path_csv(csv_path), include=("rigid",)
        )
        comparison = next(c for c in scene.comparisons if c["b"] == "measured")
        assert comparison["mean_mm"] is None
        assert comparison["frechet_mm"] is not None
        assert "shape only" in comparison["note"]
        assert any("Frechet" in note for note in scene.notes)


class TestSceneSerialisation:
    def test_round_trips_through_json(self, scene):
        data = json.loads(json.dumps(scene.to_dict(), allow_nan=False))
        assert data["name"] == scene.name
        assert len(data["input_angles_deg"]) == STEPS
        assert set(data["models"]) == set(scene.models)

    def test_is_deterministic(self, scene):
        assert json.dumps(scene.to_dict()) == json.dumps(scene.to_dict())

    def test_rejects_an_unknown_model(self, mechanism):
        with pytest.raises(ValueError, match="unknown models"):
            build_scene(mechanism, n_steps=STEPS, include=("rigid", "magic"))

    def test_rejects_an_empty_model_list(self, mechanism):
        with pytest.raises(ValueError, match="at least one"):
            build_scene(mechanism, n_steps=STEPS, include=())

    def test_without_the_fea_there_is_no_strain_to_colour(self, mechanism):
        scene = build_scene(mechanism, n_steps=STEPS, include=("rigid", "prbm"))
        assert "fea" not in scene.models
        assert any("no strain to colour" in note for note in scene.notes)


@pytest.fixture(scope="module")
def document(scene):
    return render_html(scene)


class TestHtmlViewer:
    def test_has_no_external_references(self, document):
        """The one check that makes the file usable offline, forever."""
        assert external_references(document) == []

    def test_a_namespace_uri_is_not_an_external_reference(self):
        """``createElementNS`` needs the SVG namespace by name; it fetches nothing."""
        assert external_references('xmlns="http://www.w3.org/2000/svg"') == []
        assert external_references('<script src="https://cdn.example/x.js">') != []

    def test_embeds_every_state(self, document, scene):
        payload = _payload(document)
        assert len(payload["scene"]["input_angles_deg"]) == scene.n_frames
        assert len(payload["scene"]["models"]["fea"]["frames"]) == scene.n_frames

    def test_shows_the_placeholder_caveat(self, document):
        assert "Not a physical prediction" in document
        assert "PLA.properties.allowable_strain" in document

    def test_says_the_strain_scale_is_relative_while_unmeasured(self, document):
        assert "placeholder: relative scale only" in document

    def test_a_script_tag_in_the_data_cannot_close_the_block(self, mechanism):
        """Otherwise a design name could break out of the JSON and into the page."""
        mechanism.base.name = "</script><img>"
        scene = build_scene(mechanism, n_steps=2, include=("rigid",))
        document = render_html(scene)
        mechanism.base.name = "viz_demo"
        assert "</script><img>" not in _payload_text(document)
        assert _payload(document)["scene"]["name"] == "</script><img>"

    def test_renders_the_same_bytes_twice(self, scene):
        assert render_html(scene) == render_html(scene)

    def test_write_html_creates_parents(self, scene, tmp_path):
        out = write_html(scene, tmp_path / "nested" / "view.html")
        assert out.exists()
        assert out.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")

    def test_carries_the_provenance(self, document, scene):
        payload = _payload(document)
        assert payload["scene"]["provenance"]["code_commit"] == scene.provenance.code_commit


def _payload_text(document: str) -> str:
    start = document.index('<script type="application/json" id="payload">') + len(
        '<script type="application/json" id="payload">'
    )
    return document[start : document.index("</script>", start)]


def _payload(document: str) -> dict:
    return json.loads(_payload_text(document))


class TestPalette:
    def test_zero_strain_is_the_lightest_ramp_step(self):
        assert strain_colour(0.0) == STRAIN_RAMP_LIGHT[0]

    def test_at_and_above_the_allowable_leaves_the_ramp_entirely(self):
        """Over the limit is a different statement, not a darker shade of the same one."""
        assert strain_colour(1.0) == STATUS["critical"]
        assert strain_colour(4.2) == STATUS["critical"]

    def test_interpolates_between_steps(self):
        midpoint = strain_colour(0.125)
        assert midpoint not in STRAIN_RAMP_LIGHT
        assert midpoint.startswith("#") and len(midpoint) == 7

    def test_negative_utilisation_clamps_rather_than_wrapping(self):
        assert strain_colour(-1.0) == STRAIN_RAMP_LIGHT[0]

    def test_the_measured_series_is_ink_not_a_hue(self):
        """The measurement is the reference the models are judged against."""
        assert SERIES["measured"].light == "#0b0b0b"
        assert SERIES["measured"].dark == "#ffffff"

    def test_no_model_reuses_a_status_colour(self):
        assert not set(STATUS.values()) & {s.light for s in SERIES.values()}
