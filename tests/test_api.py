"""Tests for the public API, the CLI and the generated JSON Schema."""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest
from typer.testing import CliRunner

from cmtool import Linkage, available_solvers, simulate
from cmtool.cli import app
from cmtool.schema.export import export
from cmtool.schema.sample import Sample

runner = CliRunner()


@pytest.fixture
def mech() -> Linkage:
    return Linkage.four_bar(
        ground_mm=50.0,
        input_mm=18.0,
        coupler_mm=45.0,
        output_mm=38.0,
        coupler_point_mm=(30.0, 40.0),
        input_angle_deg=60.0,
        input_range_deg=(40.0, 100.0),
    )


class TestSimulate:
    def test_rigid_solver_is_registered(self):
        assert "rigid" in available_solvers()

    def test_sweep_has_the_requested_number_of_states(self, mech):
        result = simulate(mech, solver="rigid", n_steps=25)
        assert result.n_states == 25
        assert result.path().shape == (25, 2)

    def test_arc_endpoints_are_honoured(self, mech):
        result = simulate(mech, solver="rigid", input_range_deg=(45.0, 95.0), n_steps=11)
        assert result.input_angles_deg[0] == pytest.approx(45.0)
        assert result.input_angles_deg[-1] == pytest.approx(95.0)

    def test_input_sweep_is_measured_from_the_reference_state(self, mech):
        result = simulate(mech, solver="rigid", input_range_deg=(40.0, 100.0), n_steps=7)
        assert result.input_sweep_deg[0] == pytest.approx(0.0)
        assert result.input_sweep_deg[-1] == pytest.approx(60.0)

    def test_explicit_angles_override_the_range(self, mech):
        angles = np.array([50.0, 60.0, 70.0])
        result = simulate(mech, solver="rigid", input_angles_deg=angles)
        np.testing.assert_allclose(result.input_angles_deg, angles)

    def test_missing_arc_is_an_error(self):
        mech = Linkage.four_bar(ground_mm=50.0, input_mm=18.0, coupler_mm=45.0, output_mm=38.0)
        with pytest.raises(ValueError, match="no input arc"):
            simulate(mech, solver="rigid")

    def test_unknown_solver_lists_alternatives(self, mech):
        from cmtool.core.registry import RegistryError

        with pytest.raises(RegistryError, match="available: beam_fea, prbm, rigid"):
            simulate(mech, solver="magic")

    def test_rigid_result_is_physical_because_it_uses_no_material_data(self, mech):
        result = simulate(mech, solver="rigid", n_steps=5)
        assert result.is_physical
        assert result.provenance.caveat() is None

    def test_kinematic_solver_reports_no_forces(self, mech):
        result = simulate(mech, solver="rigid", n_steps=5)
        assert result.input_torque_nmm is None
        assert result.flexure_strain is None

    def test_joint_excursion_is_peak_to_peak_and_non_negative(self, mech):
        result = simulate(mech, solver="rigid", n_steps=31)
        excursions = result.joint_excursion_deg()
        assert set(excursions) == {"A", "B", "C", "D"}
        assert all(v >= 0.0 for v in excursions.values())
        # The driven joint A rotates by exactly the commanded arc.
        assert excursions["A"] == pytest.approx(60.0, abs=1e-6)

    def test_joint_rotation_starts_at_zero(self, mech):
        result = simulate(mech, solver="rigid", n_steps=11)
        for joint in mech.joints:
            assert result.joint_rotation_deg(joint)[0] == pytest.approx(0.0)

    def test_summary_is_json_serialisable(self, mech):
        result = simulate(mech, solver="rigid", n_steps=5)
        json.dumps(result.summary(), default=str)

    def test_named_output_is_required_when_ambiguous(self, mech):
        from cmtool.core.graph import OutputPoint

        mech.outputs["Q"] = OutputPoint("Q", "coupler", (35.0, 45.0))
        result = simulate(mech, solver="rigid", n_steps=5)
        with pytest.raises(ValueError, match="name one explicitly"):
            result.path()
        assert result.path("Q").shape == (5, 2)


class TestCli:
    def test_info_lists_registered_plugins(self):
        result = runner.invoke(app, ["info"])
        assert result.exit_code == 0
        assert "rigid" in result.stdout
        assert "four_bar" in result.stdout

    def test_simulate_writes_csv(self, tmp_path, mech):
        linkage_json = tmp_path / "mech.json"
        mech.to_json(linkage_json)
        csv_path = tmp_path / "out" / "path.csv"

        result = runner.invoke(
            app, ["simulate", str(linkage_json), "--steps", "11", "--csv", str(csv_path)]
        )
        assert result.exit_code == 0, result.stdout

        rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        assert len(rows) == 11
        assert set(rows[0]) == {"input_angle_deg", "input_sweep_deg", "x_mm", "y_mm"}
        assert float(rows[0]["input_sweep_deg"]) == pytest.approx(0.0)

    def test_simulate_writes_json_summary(self, tmp_path, mech):
        linkage_json = tmp_path / "mech.json"
        mech.to_json(linkage_json)
        json_path = tmp_path / "summary.json"

        result = runner.invoke(
            app, ["simulate", str(linkage_json), "--steps", "9", "--json", str(json_path)]
        )
        assert result.exit_code == 0, result.stdout
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["n_states"] == 9
        assert payload["provenance"]["is_physical"] is True

    def test_half_specified_arc_is_refused(self, tmp_path, mech):
        linkage_json = tmp_path / "mech.json"
        mech.to_json(linkage_json)
        result = runner.invoke(app, ["simulate", str(linkage_json), "--start-deg", "40"])
        assert result.exit_code == 2

    @pytest.mark.parametrize("command", ["generate"])
    def test_future_milestone_commands_fail_clearly(self, command):
        result = runner.invoke(app, [command])
        assert result.exit_code == 2
        assert "not implemented yet" in result.stdout


class TestSchema:
    def test_export_writes_json_schema_files(self, tmp_path):
        paths = export(tmp_path)
        assert {p.name for p in paths} == {"sample.schema.json", "linkage.schema.json"}
        schema = json.loads((tmp_path / "sample.schema.json").read_text(encoding="utf-8"))
        assert schema["$schema"].startswith("https://json-schema.org/")

    def test_minimal_sample_validates(self, mech):
        sample = Sample.model_validate(
            {
                "id": "fb4_000001",
                "rigid": {
                    "type": "four_bar",
                    "bodies": [
                        {"name": b.name, "is_ground": b.is_ground} for b in mech.bodies.values()
                    ],
                    "joints": {
                        j.name: {"bodies": list(j.bodies), "xy_mm": list(j.xy)}
                        for j in mech.joints.values()
                    },
                    "outputs": {"P": {"body": "coupler", "xy_mm": [30.0, 40.0]}},
                    "input_joint": "A",
                    "input_range_deg": [40.0, 100.0],
                },
                "provenance": {
                    "code_commit": "abc123",
                    "version": "0.1.0.dev0",
                    "created_utc": "2026-09-20T00:00:00+00:00",
                    "is_physical": True,
                },
            }
        )
        assert sample.version == "0.1"
        assert sample.measured == []

    def test_unknown_field_is_rejected(self):
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Sample.model_validate({"id": "x", "typo_field": 1})

    def test_placeholder_provenance_marks_sample_as_not_physical(self, mech):
        from cmtool.schema.sample import ProvenanceSpec

        prov = ProvenanceSpec(
            code_commit="abc",
            version="0.1",
            created_utc="2026-09-20T00:00:00+00:00",
            placeholders_used=["PLA.youngs_modulus_MPa"],
            is_physical=False,
        )
        assert prov.is_physical is False
        assert prov.placeholders_used == ["PLA.youngs_modulus_MPa"]
