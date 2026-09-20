"""Tests for the camera measurement pipeline, on synthetic scenes with known truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from cmtool.vision import (
    GoNoGoReport,
    HomographyError,
    MarkerLayout,
    PadSpec,
    detect,
    fit_circle,
    jitter,
    known_motion,
    marker_sheet,
    pad_centre_mm,
    solve_plane_map,
    track,
    write_marker_sheets,
)
from cmtool.vision.synthetic import SyntheticCamera, render, sweep_frames


@pytest.fixture(scope="module")
def layout() -> MarkerLayout:
    """Marker layout taken from a real exported mechanism, not invented."""
    data = json.loads(
        Path("examples/mechanisms/fb_02_0052_mechanism.json").read_text(encoding="utf-8")
    )["layout"]
    return MarkerLayout.for_mechanism(
        fiducial_pads_mm=[tuple(p) for p in data["fiducial_pads_mm"]],
        lever_pad_mm=tuple(data["lever_pad_mm"]),
        coupler_pad_mm=tuple(data["coupler_pad_mm"]),
    )


@pytest.fixture(scope="module")
def camera(layout: MarkerLayout) -> SyntheticCamera:
    points = np.vstack(list(layout.marker_corners_mm().values()))
    return SyntheticCamera.fitting(points, tilt=(1.2e-5, -8e-6), rotation_deg=2.5)


class TestLayout:
    def test_base_pads_are_the_fixed_ones(self, layout):
        assert {pad.name for pad in layout.fixed_pads()} == {"base_0", "base_1"}

    def test_marker_ids_are_unique_across_pads(self, layout):
        ids = [i for pad in layout.pads for i in pad.marker_ids]
        assert len(ids) == len(set(ids))

    def test_duplicate_ids_are_refused(self):
        with pytest.raises(ValueError, match="more than one pad"):
            MarkerLayout(
                pads=(
                    PadSpec("a", (0.0, 0.0), (0, 1, 2, 3)),
                    PadSpec("b", (50.0, 0.0), (3, 4, 5, 6)),
                )
            )

    def test_grid_size_must_match_the_id_count(self):
        with pytest.raises(ValueError, match="ids for a 2x2 grid"):
            PadSpec("a", (0.0, 0.0), (0, 1), grid=(2, 2))

    def test_marker_corners_are_in_detector_order(self, layout):
        corners = layout.pad("coupler").marker_corners_mm()
        quad = next(iter(corners.values()))
        # Top-left, top-right, bottom-right, bottom-left in a +y-up frame.
        assert quad[0][0] < quad[1][0]
        assert quad[0][1] > quad[3][1]

    def test_real_layout_has_no_overlapping_pads(self, layout):
        """Markers printed on top of each other are simply never detected."""
        assert layout.overlaps() == []

    def test_overlap_check_catches_a_collision(self):
        collisions = MarkerLayout(
            pads=(
                PadSpec("a", (0.0, 0.0), (0,), grid=(1, 1)),
                PadSpec("b", (4.0, 0.0), (1,), grid=(1, 1)),
            )
        ).overlaps()
        assert collisions and collisions[0][2] < 0.0


class TestSheets:
    def test_one_sheet_per_pad(self, layout):
        assert set(marker_sheet(layout)) == set(layout.pad_names)

    def test_sheet_is_scaled_to_the_stated_resolution(self, layout):
        sheet = marker_sheet(layout, pixels_per_mm=10.0, margin_mm=5.0)["base_0"]
        pad = layout.pad("base_0")
        assert sheet.shape[1] == pytest.approx((pad.size_mm[0] + 10.0) * 10.0, abs=1)

    def test_sheets_round_trip_through_detection(self, layout, tmp_path):
        paths = write_marker_sheets(layout, tmp_path)
        assert set(paths) == set(layout.pad_names)
        assert all(path.is_file() for path in paths.values())


class TestDetectionAndPlaneMap:
    def test_every_marker_is_found(self, layout, camera):
        detection = detect(render(layout, camera), layout)
        assert set(detection.ids) == set(layout.marker_corners_mm())

    def test_homography_uses_only_the_fixed_markers(self, layout, camera):
        detection = detect(render(layout, camera), layout)
        plane = solve_plane_map(detection, layout)
        fixed = {i for pad in layout.fixed_pads() for i in pad.marker_ids}
        assert set(plane.marker_ids) <= fixed
        assert plane.reprojection_rms_px < 1.0

    def test_moving_pads_are_recovered_in_millimetres(self, layout, camera):
        detection = detect(render(layout, camera), layout)
        plane = solve_plane_map(detection, layout)
        for name in ("coupler", "lever"):
            truth = np.asarray(layout.pad(name).centre_mm, dtype=float)
            got = pad_centre_mm(detection, layout, plane, name)
            # A millimetre, not a tenth: the synthetic harness carries a few tenths
            # of a percent of scale bias from the render-then-detect round trip,
            # which at ~120 mm from the origin is a few tenths of a millimetre. That
            # bias belongs to the renderer, not the pipeline -- see
            # cmtool.vision.synthetic. Shape fidelity is checked separately, and far
            # more tightly, by the circle tests.
            assert float(np.linalg.norm(got - truth)) < 1.0

    def test_a_moved_pad_is_followed(self, layout, camera):
        shifted = np.asarray(layout.pad("coupler").centre_mm) + np.array([12.0, -7.0])
        detection = detect(render(layout, camera, moved_pads={"coupler": shifted}), layout)
        plane = solve_plane_map(detection, layout)
        got = pad_centre_mm(detection, layout, plane, "coupler")
        assert float(np.linalg.norm(got - shifted)) < 1.0

    def test_missing_fiducials_raise_a_useful_error(self, layout):
        from cmtool.vision.markers import Detection

        with pytest.raises(HomographyError, match="fixed markers were detected"):
            solve_plane_map(Detection(corners={}), layout)

    def test_unseen_pad_returns_none(self, layout, camera):

        detection = detect(render(layout, camera), layout)
        plane = solve_plane_map(detection, layout)
        for marker_id in layout.pad("coupler").marker_ids:
            detection.corners.pop(marker_id, None)
        assert pad_centre_mm(detection, layout, plane, "coupler") is None

    def test_plane_map_round_trips(self, layout, camera):
        detection = detect(render(layout, camera), layout)
        plane = solve_plane_map(detection, layout)
        points = np.array([[100.0, 200.0], [800.0, 640.0]])
        np.testing.assert_allclose(plane.to_px(plane.to_mm(points)), points, atol=1e-6)


class TestTracking:
    def test_tracks_a_swept_path(self, layout, camera):
        centre = np.asarray(layout.pad("coupler").centre_mm) - np.array([25.0, 0.0])
        angles = np.radians(np.linspace(-20.0, 20.0, 15))
        truth = centre + 25.0 * np.column_stack([np.cos(angles), np.sin(angles)])
        # Frame the sweep, not just the static layout, or the pad leaves the image
        # at the ends of the arc -- which is a real framing mistake worth avoiding
        # on the bench too.
        points = np.vstack([np.vstack(list(layout.marker_corners_mm().values())), truth])
        wide = SyntheticCamera.fitting(points, tilt=camera.tilt, rotation_deg=camera.rotation_deg)
        frames = sweep_frames(layout, wide, {"coupler": truth})

        result = track(frames, layout, pivot_mm=(0.0, 0.0))
        assert result.n_tracked == len(truth)
        assert float(np.linalg.norm(result.path_mm() - truth, axis=1).max()) < 1.5

    def test_shape_of_a_swept_path_is_recovered_far_more_tightly(self, layout, camera):
        """Scale bias aside, the measured path has the right shape to microns."""
        centre = np.asarray(layout.pad("coupler").centre_mm) - np.array([25.0, 0.0])
        angles = np.radians(np.linspace(-20.0, 20.0, 15))
        truth = centre + 25.0 * np.column_stack([np.cos(angles), np.sin(angles)])
        points = np.vstack([np.vstack(list(layout.marker_corners_mm().values())), truth])
        wide = SyntheticCamera.fitting(points, tilt=camera.tilt, rotation_deg=camera.rotation_deg)

        result = track(sweep_frames(layout, wide, {"coupler": truth}), layout)
        estimate = known_motion(result.path_mm())
        assert estimate.detail["shape_residual_mm"] < 0.05

    def test_writes_csv_with_angle_and_position(self, layout, camera, tmp_path):
        result = track([render(layout, camera)], layout, pivot_mm=(0.0, 0.0))
        path = result.write_csv(tmp_path / "p.csv")
        header = path.read_text(encoding="utf-8").splitlines()[0]
        assert "input_angle_deg" in header
        assert "coupler_x_mm" in header

    def test_input_angle_comes_from_the_lever_pad(self, layout, camera):
        pivot = (0.0, 0.0)
        result = track([render(layout, camera)], layout, pivot_mm=pivot)
        lever = np.asarray(layout.pad("lever").centre_mm)
        expected = np.degrees(np.arctan2(lever[1], lever[0]))
        assert result.frames[0].input_angle_deg == pytest.approx(expected, abs=0.5)

    def test_blank_frames_are_reported_not_silently_dropped(self, layout, camera):
        blank = np.full((camera.image_size[1], camera.image_size[0]), 255, dtype=np.uint8)
        result = track([blank], layout)
        assert result.n_tracked == 0
        assert result.summary()["failures"]

    def test_summary_is_serialisable(self, layout, camera):
        result = track([render(layout, camera)], layout, pivot_mm=(0.0, 0.0))
        json.dumps(result.summary())


class TestUncertainty:
    def test_jitter_measures_scatter_of_a_stationary_point(self):
        rng = np.random.default_rng(0)
        points = np.array([50.0, 20.0]) + rng.normal(0.0, 0.05, (200, 2))
        estimate = jitter(points)
        assert estimate.sigma_mm == pytest.approx(0.05 * np.sqrt(2), rel=0.2)
        assert "detector noise only" in estimate.detail["caveat"]

    def test_jitter_needs_more_than_one_frame(self):
        with pytest.raises(ValueError, match="at least two frames"):
            jitter(np.array([[0.0, 0.0]]))

    def test_circle_fit_recovers_a_known_circle(self):
        angles = np.linspace(0.0, 1.2, 40)
        points = np.array([10.0, -5.0]) + 30.0 * np.column_stack([np.cos(angles), np.sin(angles)])
        centre, radius, residual = fit_circle(points)
        np.testing.assert_allclose(centre, [10.0, -5.0], atol=1e-6)
        assert radius == pytest.approx(30.0, abs=1e-6)
        assert float(np.max(np.abs(residual))) < 1e-6

    def test_known_motion_reports_scale_error_separately(self):
        angles = np.linspace(0.0, 1.0, 30)
        # A perfect circle of the wrong radius: shape is fine, scale is not.
        points = 30.3 * np.column_stack([np.cos(angles), np.sin(angles)])
        estimate = known_motion(points, expected_radius_mm=30.0)
        assert estimate.detail["shape_residual_mm"] < 1e-6
        assert estimate.detail["radius_error_mm"] == pytest.approx(0.3, abs=1e-3)
        assert estimate.sigma_mm == pytest.approx(0.3, abs=1e-3)

    def test_scale_error_is_invisible_without_a_measured_radius(self):
        angles = np.linspace(0.0, 1.0, 30)
        points = 30.3 * np.column_stack([np.cos(angles), np.sin(angles)])
        assert known_motion(points).sigma_mm < 1e-6


class TestGoNoGo:
    @staticmethod
    def _estimate(sigma: float, method: str):
        from cmtool.vision import UncertaintyEstimate

        return UncertaintyEstimate(
            method=method, sigma_mm=sigma, max_deviation_mm=sigma * 2, n_samples=20, detail={}
        )

    def test_takes_the_worst_estimate_not_the_most_complete(self):
        """Taking the smaller of two incomplete measures would flatter the rig."""
        report = GoNoGoReport(
            estimates=[
                self._estimate(0.20, "static_jitter"),
                self._estimate(0.02, "known_motion_circle"),
            ]
        )
        assert report.governing.method == "static_jitter"
        assert report.ratio == pytest.approx(0.46 / 0.20, rel=1e-6)

    @pytest.mark.parametrize(
        ("sigma", "verdict"),
        [(0.05, "go"), (0.12, "marginal"), (0.30, "no-go")],
    )
    def test_verdict_thresholds(self, sigma, verdict):
        report = GoNoGoReport(estimates=[self._estimate(sigma, "known_motion_circle")])
        assert report.verdict == verdict

    def test_explanation_names_the_signal_and_the_ratio(self):
        report = GoNoGoReport(estimates=[self._estimate(0.05, "known_motion_circle")])
        text = report.explain()
        assert "0.460 mm" in text
        assert "PRBM vs beam FEA" in text
        assert "to 1" in text

    def test_default_signal_is_the_measured_prbm_fea_gap(self):
        from cmtool.vision.uncertainty import DEFAULT_SIGNAL_MM

        assert pytest.approx(0.46) == DEFAULT_SIGNAL_MM

    def test_report_is_serialisable(self):
        report = GoNoGoReport(estimates=[self._estimate(0.05, "known_motion_circle")])
        payload = json.loads(json.dumps(report.to_dict()))
        assert payload["verdict"] == "go"
