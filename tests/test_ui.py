"""Tests for the local UI: parameters, the solve path, and the HTTP layer.

The thing worth protecting is that the UI stays a shell. Every number it shows
has to come from the same ``convert`` / ``simulate`` / ``build_scene`` path the
CLI uses, and the placeholder caveat has to reach the page rather than being
swallowed by the server that captured the warning.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cmtool.core.graph import Linkage
from cmtool.ui.app import Solver, create_app
from cmtool.ui.model import PRESETS, DesignParams, preset_params, solve
from cmtool.ui.server import find_port

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")

DEMO = {
    "ground_mm": 40.0,
    "input_mm": 32.0,
    "coupler_mm": 44.0,
    "output_mm": 36.0,
    "coupler_x_mm": 22.0,
    "coupler_y_mm": 38.0,
    "input_angle_deg": 90.0,
    "n_steps": 5,
    "name": "test_design",
}


class TestDesignParams:
    def test_rejects_a_loop_that_cannot_close(self):
        """The four-bar inequality, caught before a solver has to fail on it."""
        with pytest.raises(ValueError, match="cannot close"):
            DesignParams(ground_mm=200.0, input_mm=10.0, coupler_mm=10.0, output_mm=10.0).validate()

    def test_rejects_half_an_arc(self):
        with pytest.raises(ValueError, match="both arc ends"):
            DesignParams(arc_start_deg=80.0).validate()

    def test_rejects_an_unknown_solver(self):
        with pytest.raises(ValueError, match="unknown solvers"):
            DesignParams(solvers=["rigid", "magic"]).validate()

    def test_refuses_a_step_count_that_would_stall_the_page(self):
        with pytest.raises(ValueError, match="too slow to be interactive"):
            DesignParams(n_steps=400).validate()

    def test_ignores_unknown_keys_when_loading(self):
        """A design saved by a later version should still open."""
        params = DesignParams.from_dict(dict(DEMO, future_option="whatever"))
        assert params.ground_mm == 40.0

    def test_the_cache_key_tracks_every_parameter(self):
        first = DesignParams.from_dict(DEMO)
        second = DesignParams.from_dict(dict(DEMO, coupler_x_mm=23.0))
        assert first.cache_key != second.cache_key
        assert first.cache_key == DesignParams.from_dict(DEMO).cache_key

    def test_fits_an_arc_when_none_is_given(self):
        """There is no full-revolution default anywhere, including here."""
        linkage = DesignParams.from_dict(DEMO).to_linkage()
        assert linkage.input_range_deg is not None
        start, end = linkage.input_range_deg
        assert 0.0 < abs(end - start) < 90.0

    def test_keeps_an_arc_that_was_given(self):
        params = DesignParams.from_dict(dict(DEMO, arc_start_deg=85.0, arc_end_deg=95.0))
        assert params.to_linkage().input_range_deg == (85.0, 95.0)


class TestPresets:
    @pytest.mark.parametrize("name", sorted(PRESETS))
    def test_every_preset_round_trips_to_its_own_geometry(self, name):
        """Editing a preset must start from the preset, not near it."""
        original = Linkage.from_json(PRESETS[name])
        rebuilt = preset_params(name).to_linkage()
        for joint in original.joints:
            assert rebuilt.joints[joint].position_mm == pytest.approx(
                original.joints[joint].position_mm, abs=1e-6
            )
        first = next(iter(original.outputs.values()))
        second = next(iter(rebuilt.outputs.values()))
        assert second.position_mm == pytest.approx(first.position_mm, abs=1e-6)

    def test_an_unknown_preset_names_the_real_ones(self):
        with pytest.raises(KeyError, match="choose from"):
            preset_params("not_a_design")


@pytest.fixture(scope="module")
def solved():
    return solve(DesignParams.from_dict(DEMO))


class TestSolve:
    def test_returns_a_scene_the_viewer_could_draw(self, solved):
        scene = solved["scene"]
        assert len(scene["input_angles_deg"]) == DEMO["n_steps"]
        assert set(scene["models"]) == {"rigid", "prbm", "fea"}
        assert scene["models"]["fea"]["frames"]

    def test_names_the_binding_joint_and_says_why(self, solved):
        f = solved["feasibility"]
        assert f["binding_joint"] in solved["scene"]["joints"]
        assert f["why"]
        assert f["joints"][f["binding_joint"]]["utilisation"] == pytest.approx(f["max_utilisation"])

    def test_the_placeholder_caveat_reaches_the_payload(self, solved):
        """Captured warnings must surface on the page, not vanish into the server."""
        assert solved["is_physical"] is False
        assert solved["caveat"]
        assert any("allowable_strain" in p for p in solved["placeholders"])
        assert any("PLACEHOLDER" in w for w in solved["warnings"])

    def test_reports_a_strain_margin_against_the_allowable(self, solved):
        margin = solved["strain_margin"]
        assert margin["available"] is True
        assert margin["margin"] > 0.0
        assert margin["allowable_is_placeholder"] is True
        assert margin["worst_joint"] in margin["peak_strain"]

    def test_is_json_serialisable(self, solved):
        """It is going over HTTP, so a NaN or a numpy scalar is a 500."""
        json.dumps(solved, allow_nan=False)

    def test_without_the_fea_the_margin_says_so_rather_than_guessing(self):
        result = solve(DesignParams.from_dict(dict(DEMO, solvers=["rigid"])))
        assert result["strain_margin"]["available"] is False
        assert "no beam FEA" in result["strain_margin"]["reason"]


class TestSolverCache:
    def test_a_repeat_request_is_served_from_cache(self):
        engine = Solver()
        params = DesignParams.from_dict(DEMO)
        first = engine.submit(params)
        _wait(engine, first.id)
        second = engine.submit(params)
        assert second.state == "done"
        assert second.message == "from cache"

    def test_a_changed_parameter_is_a_different_key(self):
        engine = Solver()
        engine.submit(DesignParams.from_dict(DEMO))
        other = engine.submit(DesignParams.from_dict(dict(DEMO, coupler_y_mm=40.0)))
        assert other.message != "from cache"

    def test_the_cache_evicts_oldest_first(self):
        engine = Solver(cache_size=2)
        for value in (38.0, 39.0, 40.0):
            job = engine.submit(DesignParams.from_dict(dict(DEMO, coupler_y_mm=value)))
            _wait(engine, job.id)
        assert len(engine.cache_keys) == 2

    def test_a_failing_solve_is_reported_not_raised(self):
        """A bad design must land on the page as a message, not kill the server."""
        engine = Solver()
        params = DesignParams.from_dict(DEMO)
        params.ground_mm = -5.0  # past validation, so the solve itself fails
        job = engine.submit(params)
        _wait(engine, job.id)
        assert job.state == "error"
        assert job.error


def _wait(engine: Solver, job_id: str, timeout: float = 120.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = engine.job(job_id)
        assert job is not None
        if job.state != "running":
            return
        time.sleep(0.05)
    raise AssertionError("solve did not finish")


@pytest.fixture(scope="module")
def client():
    """Return a test client, skipping rather than failing when the ui extra is absent."""
    testclient = pytest.importorskip("fastapi.testclient")
    with testclient.TestClient(create_app()) as test_client:
        yield test_client


class TestHttp:
    def test_the_page_is_served_and_needs_no_network(self, client):
        """Offline is a requirement, not a nicety: this runs on a laptop at a bench."""
        body = client.get("/").text
        assert body.startswith("<!DOCTYPE html>")
        for marker in ("http://", "https://", "//cdn"):
            assert marker not in body, f"the page references {marker}"

    def test_the_shared_drawing_module_is_served(self, client):
        """The UI and the standalone viewer draw from one file, not two copies."""
        body = client.get("/shared/scene_draw.js").text
        assert "CmtoolScene" in body
        assert "drawMechanism" in body

    def test_meta_offers_the_presets_and_the_palette(self, client):
        meta = client.get("/api/meta").json()
        assert set(meta["presets"]) == set(PRESETS)
        assert "rigid" in meta["style"]["series"]
        assert meta["defaults"]["ground_mm"] > 0

    def test_a_preset_can_be_loaded(self, client):
        params = client.get("/api/preset/demo_pair").json()
        assert params["name"] == "demo_pair"
        assert params["ground_mm"] == pytest.approx(40.0)

    def test_an_unknown_preset_is_a_404(self, client):
        assert client.get("/api/preset/nope").status_code == 404

    def test_a_bad_design_is_a_422_with_a_readable_reason(self, client):
        response = client.post(
            "/api/solve",
            json={"ground_mm": 200.0, "input_mm": 10.0, "coupler_mm": 10.0, "output_mm": 10.0},
        )
        assert response.status_code == 422
        assert "cannot close" in response.json()["detail"]

    def test_a_solve_runs_and_its_result_can_be_collected(self, client):
        job = client.post("/api/solve", json=DEMO).json()
        _wait_http(client, job["id"])
        payload = client.get(f"/api/job/{job['id']}/result").json()
        assert payload["scene"]["name"] == "test_design"
        assert payload["caveat"]

    def test_the_result_of_an_unknown_job_is_a_404(self, client):
        assert client.get("/api/job/deadbeef/result").status_code == 404

    def test_design_json_is_what_the_cli_would_read(self, client):
        """Save design JSON has to produce a file the rest of the toolkit accepts."""
        response = client.post("/api/design.json", json=DEMO)
        assert response.status_code == 200
        assert "attachment" in response.headers["content-disposition"]
        linkage = Linkage.from_dict(json.loads(response.text))
        linkage.validate()
        assert linkage.input_range_deg is not None
        assert set(linkage.joints) == {"A", "B", "C", "D"}

    def test_an_unknown_export_is_a_404(self, client):
        assert client.post("/api/export/dxf", json=DEMO).status_code == 404


def _wait_http(client, job_id: str, timeout: float = 120.0) -> None:
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = client.get(f"/api/job/{job_id}").json()
        if state["state"] != "running":
            assert state["state"] == "done", state
            return
        time.sleep(0.05)
    raise AssertionError("solve did not finish")


class TestExports:
    def test_stl_and_print_sheet_build(self, client):
        cadquery = pytest.importorskip("cadquery")
        assert cadquery is not None
        stl = client.post("/api/export/stl", json=DEMO)
        assert stl.status_code == 200
        assert len(stl.content) > 10_000
        sheet = client.post("/api/export/sheet", json=DEMO)
        assert sheet.status_code == 200
        assert "Print sheet" in sheet.text


class TestServer:
    def test_finds_a_port_even_when_the_preferred_one_is_taken(self):
        """'Port 8765 is busy' is not a thing to discover in front of an audience."""
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as taken:
            taken.bind(("127.0.0.1", 0))
            taken.listen(1)
            busy = taken.getsockname()[1]
            chosen = find_port("127.0.0.1", busy)
        assert chosen != busy
        assert 1024 < chosen < 65536


class TestDrawnSceneMatchesTheSolvers:
    def test_the_scene_paths_are_the_solver_paths(self):
        """The UI draws solver output, never its own idea of where the point went."""
        from cmtool.api import convert, simulate

        params = DesignParams.from_dict(DEMO)
        payload = solve(params)
        linkage = params.to_linkage()
        angles = np.asarray(payload["scene"]["input_angles_deg"], dtype=float)
        rigid = simulate(linkage, solver="rigid", input_angles_deg=angles)
        assert np.asarray(payload["scene"]["models"]["rigid"]["path"]) == pytest.approx(
            rigid.path(), abs=1e-9
        )
        mechanism = convert(
            linkage, input_range_deg=linkage.input_range_deg, thickness_mm=params.thickness_mm
        )
        assert payload["feasibility"]["binding_joint"] == mechanism.feasibility.binding_joint
