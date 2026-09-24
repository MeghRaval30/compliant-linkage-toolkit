"""Tests for the figure scripts, and for the rule about missing measurements.

The point of ``cmtool figures`` is that it can be run today and again after A6
and produce the right thing both times. So the checks here are mostly about what
it does when a measurement does not exist yet: it must say so, in the figure and
in the manifest, and it must never draw a series it does not have.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cmtool import Linkage, convert
from cmtool.core.provenance import Provenance
from cmtool.viz.figures import (
    SINGLE_COLUMN_IN,
    FigureInputs,
    fig_conversion_artefact,
    fig_design_rule,
    fig_feasibility,
    fig_path_overlay,
    fig_prbm_variant,
    fig_strain,
    fig_torque_curves,
    fig_uncertainty,
    generate_all,
    layout_for,
)
from cmtool.viz.scene import build_scene

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")

matplotlib = pytest.importorskip("matplotlib")

STEPS = 5


@pytest.fixture(scope="module")
def inputs():
    linkage = Linkage.four_bar(
        ground_mm=60.0,
        input_mm=55.0,
        coupler_mm=62.0,
        output_mm=58.0,
        coupler_point_mm=(35.0, 70.0),
        input_angle_deg=95.0,
        input_range_deg=(88.0, 102.0),
        name="fig_demo",
    )
    mechanism = convert(linkage, thickness_mm=0.6)
    scene = build_scene(mechanism, n_steps=STEPS)
    provenance = Provenance(notes={"figures": ["fig_demo"]})
    provenance.merge(scene.provenance)
    return FigureInputs(
        scenes={"fig_demo": scene},
        mechanisms={"fig_demo": mechanism},
        provenance=provenance,
        primary="fig_demo",
    )


class TestMissingMeasurementsStayMissing:
    def test_the_go_no_go_is_skipped_rather_than_simulated(self, inputs, tmp_path):
        """There is no honest way to draw this one without real footage.

        The synthetic harness validates the tracking software, not the rig, so
        its numbers would answer the go/no-go with the wrong measurement.
        """
        result = fig_uncertainty(inputs, tmp_path)
        assert result.produced is False
        assert result.path is None
        assert "cmtool uncertainty" in (result.reason or "")

    def test_the_path_overlay_records_the_absent_measured_series(self, inputs, tmp_path):
        result = fig_path_overlay(inputs, tmp_path)
        assert result.produced is True
        assert result.missing == ["measured"]

    def test_the_torque_figure_records_it_too(self, inputs, tmp_path):
        result = fig_torque_curves(inputs, tmp_path)
        assert result.produced is True
        assert result.missing == ["measured"]

    def test_strain_flags_that_the_allowable_is_a_placeholder(self, inputs, tmp_path):
        result = fig_strain(inputs, tmp_path)
        assert result.produced is True
        assert result.missing == ["measured_allowable_strain"]


class TestEachFigure:
    @pytest.mark.parametrize(
        "builder",
        [fig_path_overlay, fig_torque_curves, fig_strain, fig_feasibility, fig_design_rule],
    )
    def test_writes_a_non_empty_file(self, inputs, tmp_path, builder):
        result = builder(inputs, tmp_path)
        assert result.produced and result.path is not None
        assert result.path.stat().st_size > 5_000

    def test_writes_every_requested_format(self, inputs, tmp_path):
        result = fig_feasibility(inputs, tmp_path, formats=("png", "svg"))
        assert result.path is not None
        assert (tmp_path / "feasibility.png").exists()
        assert (tmp_path / "feasibility.svg").exists()

    def test_the_conversion_artefact_is_measured_not_assumed(self, inputs, tmp_path):
        """Both curves are rigid, so whatever it reports came from placement alone."""
        result = fig_conversion_artefact(inputs, tmp_path)
        assert result.produced
        assert result.inputs["mean_mm"] > 0.0
        assert result.inputs["max_mm"] >= result.inputs["mean_mm"]

    def test_the_prbm_variant_figure_carries_no_false_caveat(self, tmp_path):
        """Its constants are dimensionless, so no placeholder reaches it.

        Stamping the run's material caveat on this figure would be a disclaimer
        that is not true -- the opposite failure to inventing data, and just as
        much a misstatement of what is known.
        """
        result = fig_prbm_variant(tmp_path, load_ratios=(0.0, 0.1, 1.0))
        assert result.produced
        assert result.inputs["recommended"] == "end_moment"
        assert result.inputs["stiffness_errors"]["end_moment"] < 0.05
        assert result.inputs["stiffness_errors"]["end_force"] > 0.3

    def test_strain_is_skipped_without_the_fea(self, inputs, tmp_path):
        mechanism = inputs.primary_mechanism
        scene = build_scene(mechanism, n_steps=STEPS, include=("rigid",))
        without = FigureInputs(
            scenes={"fig_demo": scene},
            mechanisms={"fig_demo": mechanism},
            provenance=scene.provenance,
            primary="fig_demo",
        )
        result = fig_strain(without, tmp_path)
        assert result.produced is False
        assert "no beam FEA" in (result.reason or "")


@pytest.fixture(scope="module")
def run(inputs, tmp_path_factory):
    out = tmp_path_factory.mktemp("figures")
    return out, generate_all(out, inputs)


class TestGenerateAll:
    def test_writes_a_manifest_naming_every_figure(self, run):
        out, results = run
        manifest = json.loads((out / "figures.json").read_text(encoding="utf-8"))
        assert {entry["name"] for entry in manifest["generated"]} == {r.name for r in results}

    def test_the_manifest_explains_every_skip(self, run):
        _, results = run
        for result in results:
            assert result.produced or result.reason

    def test_the_manifest_records_what_was_missing(self, run):
        out, _ = run
        manifest = json.loads((out / "figures.json").read_text(encoding="utf-8"))
        assert manifest["measured_path"] is None
        assert manifest["n_torque_readings"] == 0
        assert manifest["has_uncertainty_report"] is False

    def test_only_draws_what_was_asked_for(self, inputs, tmp_path):
        results = generate_all(tmp_path, inputs, only=("feasibility",))
        assert [r.name for r in results] == ["feasibility"]
        assert (tmp_path / "feasibility.png").exists()
        assert not (tmp_path / "strain.png").exists()


class TestUncertaintyFigureWhenAReportExists:
    def test_draws_the_verdict_from_the_report(self, inputs, tmp_path):
        report = {
            "signal_mm": 0.46,
            "verdict": "go",
            "estimates": [
                {"method": "static_jitter", "sigma_mm": 0.02, "max_deviation_mm": 0.05},
                {"method": "known_motion_circle", "sigma_mm": 0.06, "max_deviation_mm": 0.09},
            ],
        }
        with_report = FigureInputs(
            scenes=inputs.scenes,
            mechanisms=inputs.mechanisms,
            provenance=inputs.provenance,
            uncertainty=report,
            primary=inputs.primary,
        )
        result = fig_uncertainty(with_report, tmp_path)
        assert result.produced
        assert result.inputs["verdict"] == "go"
        assert result.inputs["signal_mm"] == pytest.approx(0.46)

    def test_uses_the_reports_own_signal_not_the_default(self, inputs, tmp_path):
        report = {
            "signal_mm": 1.25,
            "verdict": "no-go",
            "estimates": [
                {"method": "known_motion_circle", "sigma_mm": 0.8, "max_deviation_mm": 1.4}
            ],
        }
        with_report = FigureInputs(
            scenes=inputs.scenes,
            mechanisms=inputs.mechanisms,
            provenance=inputs.provenance,
            uncertainty=report,
            primary=inputs.primary,
        )
        result = fig_uncertainty(with_report, tmp_path)
        assert result.inputs["signal_mm"] == pytest.approx(1.25)


class TestTorqueReadingsWhenTheyExist:
    def test_the_measured_series_stops_being_missing(self, inputs, tmp_path):
        from cmtool.metrics.torque import TorqueReading

        readings = [
            TorqueReading(input_angle_deg=angle, torque_nmm=value, direction=direction)
            for angle, value, direction in (
                (90.0, 12.0, "loading"),
                (98.0, 18.0, "loading"),
                (98.0, 16.0, "unloading"),
            )
        ]
        with_torque = FigureInputs(
            scenes=inputs.scenes,
            mechanisms=inputs.mechanisms,
            provenance=inputs.provenance,
            torque_readings=readings,
            primary=inputs.primary,
        )
        result = fig_torque_curves(with_torque, tmp_path)
        assert result.produced
        assert result.missing == []
        assert result.inputs["n_readings"] == 3


class TestMeasuredPathWhenItExists:
    def test_the_overlay_gains_a_fourth_series(self, inputs, tmp_path):
        from cmtool.metrics.paths import read_path_csv

        csv_path = tmp_path / "measured.csv"
        rows = ["frame,input_angle_deg,coupler_x_mm,coupler_y_mm"]
        scene = inputs.primary_scene
        for index, (angle, point) in enumerate(
            zip(scene.input_angles_deg, scene.models["fea"].path_mm, strict=True)
        ):
            rows.append(f"{index},{angle:.4f},{point[0] + 0.3:.4f},{point[1] - 0.2:.4f}")
        csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")

        measured = read_path_csv(csv_path)
        with_measured = build_scene(inputs.primary_mechanism, n_steps=STEPS, measured=measured)
        figure_inputs = FigureInputs(
            scenes={"fig_demo": with_measured},
            mechanisms=inputs.mechanisms,
            provenance=with_measured.provenance,
            measured=measured,
            primary="fig_demo",
        )
        result = fig_path_overlay(figure_inputs, tmp_path)
        assert result.produced
        assert result.missing == []

        gap = next(c for c in with_measured.comparisons if c["b"] == "measured" and c["a"] == "fea")
        assert gap["mean_mm"] == pytest.approx(float(np.hypot(0.3, 0.2)), rel=1e-3)


class TestJournalGeometry:
    """Every figure has to survive being dropped into one column of a journal."""

    def test_an_unknown_width_is_refused_by_name(self):
        with pytest.raises(ValueError, match="unknown column width"):
            layout_for("quarter")

    @pytest.mark.parametrize(
        "builder",
        [fig_path_overlay, fig_torque_curves, fig_strain, fig_feasibility, fig_design_rule],
    )
    def test_fits_a_single_journal_column(self, inputs, tmp_path, builder):
        """Laid out at the final printed width, not shrunk into it afterwards.

        ``savefig`` uses a tight bounding box, so an unwrapped title or an
        over-wide caption silently widens the saved file past the column. A
        small tolerance covers the tight-box padding.
        """
        from PIL import Image

        result = builder(inputs, tmp_path, column="single")
        assert result.path is not None
        with Image.open(result.path) as image:
            width_in = image.size[0] / image.info.get("dpi", (300, 300))[0]
        assert width_in <= SINGLE_COLUMN_IN * 1.08, f"{result.name} is {width_in:.2f} in wide"

    def test_double_column_is_wider_than_single(self, inputs, tmp_path):
        from PIL import Image

        widths = {}
        for column in ("single", "double"):
            out = tmp_path / column
            result = fig_torque_curves(inputs, out, column=column)
            assert result.path is not None
            with Image.open(result.path) as image:
                widths[column] = image.size[0]
        assert widths["double"] > widths["single"] * 1.5

    def test_the_manifest_records_the_width_it_was_built_for(self, inputs, tmp_path):
        generate_all(tmp_path, inputs, only=("feasibility",), column="double")
        manifest = json.loads((tmp_path / "figures.json").read_text(encoding="utf-8"))
        assert manifest["column"] == "double"
        assert manifest["figure_width_in"] == pytest.approx(7.0)
