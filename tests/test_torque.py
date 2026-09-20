"""Tests for the mechanism beam FEA and the input-torque measurement chain."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from cmtool import Linkage, convert, simulate
from cmtool.metrics import (
    compare,
    hole_position_mm,
    moment_arm_mm,
    read_measurements,
    read_weight_measurements,
    weight_to_force_n,
    write_scale_template,
    write_weight_template,
)

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")


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
        name="torque_demo",
    )
    return convert(linkage, thickness_mm=0.6)


@pytest.fixture(scope="module")
def results(mechanism):
    return (
        simulate(mechanism, solver="prbm", n_steps=11),
        simulate(mechanism, solver="beam_fea", n_steps=11),
    )


class TestMechanismBeamFea:
    def test_solves_every_input_angle(self, results):
        _, fea = results
        assert fea.n_states == 11
        assert fea.path().shape == (11, 2)

    def test_mesh_has_flexure_and_link_elements(self, results):
        _, fea = results
        assert fea.diagnostics["n_elements"] > fea.diagnostics["flexure_elements"] * 4

    def test_reports_torque_and_strain(self, results):
        _, fea = results
        assert fea.input_torque_nmm is not None
        assert set(fea.flexure_strain) == {"A", "B", "C", "D"}

    def test_torque_vanishes_at_the_unstressed_configuration(self, results, mechanism):
        _, fea = results
        index = int(np.argmin(np.abs(fea.input_angles_deg - mechanism.reference_input_deg)))
        assert abs(fea.input_torque_nmm[index]) < 0.05 * np.max(np.abs(fea.input_torque_nmm))

    def test_disagrees_with_the_prbm_by_a_measurable_amount(self, results):
        """The PRBM-vs-FEA gap is a dataset metric, not a bug: it is the point."""
        prbm, fea = results
        path_gap = np.linalg.norm(fea.path() - prbm.path(), axis=1)
        assert 0.0 < path_gap.max() < 10.0
        torque_gap = abs(
            np.max(np.abs(fea.input_torque_nmm)) - np.max(np.abs(prbm.input_torque_nmm))
        )
        assert torque_gap > 0.0

    def test_path_difference_comes_from_compliance_not_kinematics(self, results, mechanism):
        """Both solve the same linkage, so any path gap is compliance the PRBM lacks."""
        prbm, fea = results
        rigid = simulate(
            mechanism.base,
            solver="rigid",
            input_range_deg=mechanism.input_range_deg,
            n_steps=11,
        )
        np.testing.assert_allclose(prbm.path(), rigid.path(), atol=1e-9)
        assert np.linalg.norm(fea.path() - rigid.path(), axis=1).max() > 1e-6

    def test_fea_strain_is_the_same_order_as_prbm_strain(self, results):
        prbm, fea = results
        for joint in fea.flexure_strain:
            ratio = fea.diagnostics["peak_strain"][joint] / prbm.diagnostics["peak_strain"][joint]
            assert 0.5 < ratio < 2.0

    def test_refuses_to_run_without_the_compliant_mechanism(self, mechanism):
        with pytest.raises(ValueError, match="needs the compliant mechanism"):
            simulate(mechanism.base, solver="beam_fea", input_range_deg=(88.0, 102.0))

    def test_carries_the_placeholder_caveat(self, results):
        _, fea = results
        assert not fea.is_physical


class TestTorqueTemplate:
    def test_template_has_loading_and_unloading_rows(self, tmp_path):
        path = write_scale_template(
            tmp_path / "t.csv",
            lever_radius_mm=36.0,
            input_range_deg=(88.0, 102.0),
            n_points=5,
            repeats=2,
        )
        rows = [
            r
            for r in csv.DictReader(
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if not line.startswith("#") and line.strip()
            )
        ]
        assert len(rows) == 5 * 2 * 2
        assert {r["direction"] for r in rows} == {"loading", "unloading"}

    def test_template_prefills_the_lever_radius_and_printer(self, tmp_path):
        path = write_scale_template(
            tmp_path / "t.csv",
            lever_radius_mm=36.0,
            input_range_deg=(88.0, 102.0),
            printer="kobra2_neo",
        )
        text = path.read_text(encoding="utf-8")
        assert "36.00" in text
        assert "kobra2_neo" in text
        assert "print_purpose" in text

    def test_template_explains_the_torque_formula(self, tmp_path):
        path = write_scale_template(
            tmp_path / "t.csv", lever_radius_mm=36.0, input_range_deg=(88.0, 102.0)
        )
        text = path.read_text(encoding="utf-8")
        assert "T = F * r * sin" in text
        assert "hysteresis" in text


class TestTorqueComparison:
    @staticmethod
    def _write(path, rows):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "specimen_id",
                    "printer",
                    "print_purpose",
                    "direction",
                    "input_angle_deg",
                    "force_N",
                    "pull_angle_from_lever_deg",
                    "lever_radius_mm",
                    "repeat",
                    "notes",
                ]
            )
            writer.writerows(rows)

    def test_force_and_radius_become_torque(self, tmp_path):
        path = tmp_path / "m.csv"
        self._write(path, [["s1", "kobra2_neo", "data", "loading", "95", "2.0", "90", "40", 1, ""]])
        readings = read_measurements(path)
        assert len(readings) == 1
        assert readings[0].torque_nmm == pytest.approx(80.0)

    def test_off_perpendicular_pull_reduces_the_torque(self, tmp_path):
        path = tmp_path / "m.csv"
        self._write(path, [["s1", "k", "data", "loading", "95", "2.0", "30", "40", 1, ""]])
        assert read_measurements(path)[0].torque_nmm == pytest.approx(40.0)

    def test_blank_force_rows_are_skipped(self, tmp_path):
        path = tmp_path / "m.csv"
        self._write(
            path,
            [
                ["s1", "k", "data", "loading", "95", "", "90", "40", 1, ""],
                ["s1", "k", "data", "loading", "96", "1.0", "90", "40", 1, ""],
            ],
        )
        assert len(read_measurements(path)) == 1

    def test_comparison_reports_error_per_model(self, tmp_path, results):
        prbm, fea = results
        path = tmp_path / "m.csv"
        # Fabricate readings that sit exactly on the FEA prediction.
        rows = []
        for angle, torque in zip(fea.input_angles_deg, fea.input_torque_nmm, strict=True):
            rows.append(
                [
                    "s1",
                    "kobra2_neo",
                    "data",
                    "loading",
                    f"{angle:.4f}",
                    f"{torque / 40.0:.6f}",
                    "90",
                    "40",
                    1,
                    "",
                ]
            )
        self._write(path, rows)

        result = compare(
            read_measurements(path),
            {
                "prbm": (prbm.input_angles_deg, prbm.input_torque_nmm),
                "beam_fea": (fea.input_angles_deg, fea.input_torque_nmm),
            },
        )
        # The template stores force to six decimals, so the round trip is exact to
        # about 1e-5 N*mm -- orders of magnitude below any real disagreement.
        fea_error = result.model_errors["beam_fea"]["mean_abs_error_nmm"]
        prbm_error = result.model_errors["prbm"]["mean_abs_error_nmm"]
        assert fea_error < 1e-4
        assert prbm_error > 1000.0 * max(fea_error, 1e-12)

    def test_hysteresis_is_reported_separately_from_model_error(self, tmp_path):
        path = tmp_path / "m.csv"
        rows = []
        for angle in (90.0, 95.0, 100.0):
            rows.append(["s1", "k", "data", "loading", f"{angle}", "2.0", "90", "40", 1, ""])
            rows.append(["s1", "k", "data", "unloading", f"{angle}", "1.5", "90", "40", 1, ""])
        self._write(path, rows)

        result = compare(read_measurements(path), {})
        assert result.hysteresis_nmm == pytest.approx(20.0)
        assert result.hysteresis_fraction == pytest.approx(0.25)

    def test_empty_measurements_do_not_crash(self):
        result = compare([], {})
        assert result.n_readings == 0
        assert result.hysteresis_nmm is None

    def test_to_dict_is_serialisable(self, tmp_path):
        import json

        path = tmp_path / "m.csv"
        self._write(path, [["s1", "k", "data", "loading", "95", "2.0", "90", "40", 1, ""]])
        json.dumps(compare(read_measurements(path), {}).to_dict())


class TestDeadWeightRig:
    """Mode A: thread over a pulley, because the pull on a flat mechanism is horizontal."""

    PIVOT = np.array([0.0, 0.0])
    HOLE = np.array([-36.0, 0.0])

    def test_mass_becomes_tension(self):
        assert weight_to_force_n(100.0) == pytest.approx(0.980665, rel=1e-9)

    def test_hole_swings_with_the_lever(self):
        moved = hole_position_mm(self.PIVOT, self.HOLE, 90.0, 0.0)
        np.testing.assert_allclose(moved, [0.0, -36.0], atol=1e-9)
        assert np.linalg.norm(moved - self.PIVOT) == pytest.approx(36.0)

    def test_moment_arm_is_the_cross_product_not_the_radius(self):
        """A thread pulling along the lever exerts no torque, whatever the radius."""
        along = moment_arm_mm(self.PIVOT, self.HOLE, np.array([-200.0, 0.0]))
        assert along == pytest.approx(0.0, abs=1e-9)

        perpendicular = moment_arm_mm(self.PIVOT, self.HOLE, np.array([-36.0, 200.0]))
        assert abs(perpendicular) == pytest.approx(36.0, rel=1e-9)

    def test_moment_arm_changes_as_the_lever_rotates(self):
        """The reason a fixed radius is not good enough."""
        pulley = np.array([-150.0, 120.0])
        arms = [
            abs(moment_arm_mm(self.PIVOT, hole_position_mm(self.PIVOT, self.HOLE, a, 0.0), pulley))
            for a in (-20.0, 0.0, 20.0)
        ]
        assert max(arms) - min(arms) > 1.0

    def test_pulley_on_the_hole_is_refused(self):
        with pytest.raises(ValueError, match="cannot sit on the lever hole"):
            moment_arm_mm(self.PIVOT, self.HOLE, self.HOLE)

    def test_template_has_both_sequences_and_setup_notes(self, tmp_path):
        path = write_weight_template(
            tmp_path / "w.csv", hole_radius_mm=36.0, masses_g=(0.0, 50.0), repeats=2
        )
        text = path.read_text(encoding="utf-8")
        assert "hung_mass_g" in text
        assert "pulley_x_mm" in text
        assert "MID-THICKNESS" in text
        assert "moment arm is not" in text

    def test_reading_a_weight_template_resolves_the_moment_arm(self, tmp_path):
        path = tmp_path / "w.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "specimen_id",
                    "printer",
                    "print_purpose",
                    "sequence",
                    "step",
                    "hung_mass_g",
                    "measured_input_angle_deg",
                    "pulley_x_mm",
                    "pulley_y_mm",
                    "hole_radius_mm",
                    "settle_time_s",
                    "repeat",
                    "notes",
                ]
            )
            writer.writerow(
                [
                    "s1",
                    "kobra2_neo",
                    "data",
                    "loading",
                    1,
                    "100",
                    "0",
                    "-36.0",
                    "200.0",
                    "36.0",
                    "10",
                    1,
                    "",
                ]
            )
        readings = read_weight_measurements(
            path, pivot_mm=self.PIVOT, hole_mm=self.HOLE, reference_angle_deg=0.0
        )
        assert len(readings) == 1
        assert readings[0].mode == "weight"
        # 100 g over a perpendicular 36 mm arm.
        assert abs(readings[0].torque_nmm) == pytest.approx(0.980665 * 36.0, rel=1e-6)

    def test_mode_is_detected_from_the_columns(self, tmp_path):
        path = tmp_path / "w.csv"
        write_weight_template(path, hole_radius_mm=36.0, masses_g=(50.0,), repeats=1)
        readings = read_measurements(
            path, pivot_mm=self.PIVOT, hole_mm=self.HOLE, reference_angle_deg=0.0
        )
        assert readings == []  # blank angles, nothing to read, but no crash

    def test_comparison_reports_angle_residuals_too(self, results):
        from cmtool.metrics.torque import TorqueReading

        fea = results[1]
        torque = float(np.max(np.abs(fea.input_torque_nmm)) * 0.5)
        readings = [
            TorqueReading(
                input_angle_deg=float(fea.input_angles_deg[-1]),
                torque_nmm=torque,
                direction="loading",
                mode="weight",
            )
        ]
        comparison = compare(readings, {"beam_fea": (fea.input_angles_deg, fea.input_torque_nmm)})
        assert comparison.mode == "weight"
        assert "beam_fea" in comparison.angle_errors
        assert comparison.angle_errors["beam_fea"]["n_in_range"] == 1

    def test_hysteresis_is_labelled_as_including_rig_friction(self):
        payload = compare([], {}).to_dict()
        assert "rig friction" in payload["hysteresis_note"]
