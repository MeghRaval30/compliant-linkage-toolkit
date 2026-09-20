"""Tracking a printed mechanism from frames or video.

For every frame: detect markers, solve the image-to-millimetre homography from
the fixed base fiducials, then read the lever and coupler pads through it. The
input angle comes from the lever pad's position relative to the input pivot,
which is more robust than a single marker's own orientation -- the pivot is tens
of millimetres away, so the same corner noise subtends a much smaller angle.

The homography is re-solved **every frame** rather than once. The base pads do
not move relative to the mechanism, but the camera might: a nudged tripod
between takes would otherwise shift every subsequent measurement silently.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cmtool.core.units import FloatArray, ImageArray
from cmtool.vision.calibrate import CameraCalibration
from cmtool.vision.homography import HomographyError, pad_centre_mm, solve_plane_map
from cmtool.vision.markers import MarkerLayout, detect


@dataclass
class FrameResult:
    """What was measured in one frame."""

    index: int
    coupler_mm: FloatArray | None = None
    lever_mm: FloatArray | None = None
    input_angle_deg: float | None = None
    n_markers: int = 0
    homography_rms_px: float | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        """Whether both tracked pads were measured."""
        return self.error is None and self.coupler_mm is not None


@dataclass
class TrackingResult:
    """Measurements across a whole sequence."""

    frames: list[FrameResult] = field(default_factory=list)
    layout: MarkerLayout | None = None
    pivot_mm: tuple[float, float] | None = None

    @property
    def n_frames(self) -> int:
        """Total frames processed."""
        return len(self.frames)

    @property
    def n_tracked(self) -> int:
        """Frames in which the coupler was measured."""
        return sum(1 for frame in self.frames if frame.ok)

    def path_mm(self) -> FloatArray:
        """Measured coupler path, one row per successfully tracked frame."""
        points = [f.coupler_mm for f in self.frames if f.ok and f.coupler_mm is not None]
        return np.asarray(points, dtype=float) if points else np.empty((0, 2))

    def input_angles_deg(self) -> FloatArray:
        """Measured input angle for each successfully tracked frame."""
        values = [f.input_angle_deg for f in self.frames if f.ok and f.input_angle_deg is not None]
        return np.asarray(values, dtype=float) if values else np.empty(0)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        rms = [f.homography_rms_px for f in self.frames if f.homography_rms_px is not None]
        failures: dict[str, int] = {}
        for frame in self.frames:
            if frame.error:
                failures[frame.error.split(";")[0]] = failures.get(frame.error.split(";")[0], 0) + 1
        return {
            "n_frames": self.n_frames,
            "n_tracked": self.n_tracked,
            "tracked_fraction": self.n_tracked / self.n_frames if self.n_frames else 0.0,
            "homography_rms_px_mean": float(np.mean(rms)) if rms else None,
            "homography_rms_px_max": float(np.max(rms)) if rms else None,
            "failures": failures,
        }

    def write_csv(self, path: str | Path) -> Path:
        """Write the measured path against input angle."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "frame",
                    "input_angle_deg",
                    "coupler_x_mm",
                    "coupler_y_mm",
                    "n_markers",
                    "homography_rms_px",
                ]
            )
            for frame in self.frames:
                if not frame.ok or frame.coupler_mm is None:
                    continue
                writer.writerow(
                    [
                        frame.index,
                        "" if frame.input_angle_deg is None else f"{frame.input_angle_deg:.4f}",
                        f"{frame.coupler_mm[0]:.4f}",
                        f"{frame.coupler_mm[1]:.4f}",
                        frame.n_markers,
                        "" if frame.homography_rms_px is None else f"{frame.homography_rms_px:.4f}",
                    ]
                )
        return out


def frames_from_video(path: str | Path, *, stride: int = 1) -> Iterator[ImageArray]:
    """Yield frames from a video file, optionally taking every ``stride``-th."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"could not open video {path}")
    try:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index % stride == 0:
                yield np.asarray(frame)
            index += 1
    finally:
        capture.release()


def frames_from_directory(path: str | Path, *, pattern: str = "*.png") -> Iterator[ImageArray]:
    """Yield images from a directory, in sorted filename order."""
    for file in sorted(Path(path).glob(pattern)):
        image = cv2.imread(str(file))
        if image is not None:
            yield np.asarray(image)


def track_frame(
    image: ImageArray,
    layout: MarkerLayout,
    *,
    index: int = 0,
    pivot_mm: tuple[float, float] | None = None,
    calibration: CameraCalibration | None = None,
) -> FrameResult:
    """Measure one frame."""
    result = FrameResult(index=index)
    working = calibration.undistort(image) if calibration is not None else image

    detection = detect(working, layout)
    result.n_markers = len(detection.corners)
    if calibration is not None:
        for marker_id, corners in detection.corners.items():
            detection.corners[marker_id] = calibration.undistort_points(corners)

    try:
        plane = solve_plane_map(detection, layout)
    except HomographyError as exc:
        result.error = str(exc)
        return result
    result.homography_rms_px = plane.reprojection_rms_px

    if "coupler" in layout.pad_names:
        result.coupler_mm = pad_centre_mm(detection, layout, plane, "coupler")
    if "lever" in layout.pad_names:
        result.lever_mm = pad_centre_mm(detection, layout, plane, "lever")

    if result.coupler_mm is None:
        result.error = "coupler pad not detected"
        return result

    if result.lever_mm is not None and pivot_mm is not None:
        offset = result.lever_mm - np.asarray(pivot_mm, dtype=float)
        result.input_angle_deg = float(np.degrees(np.arctan2(offset[1], offset[0])))
    return result


def track(
    images: Iterator[ImageArray] | list[ImageArray],
    layout: MarkerLayout,
    *,
    pivot_mm: tuple[float, float] | None = None,
    calibration: CameraCalibration | None = None,
) -> TrackingResult:
    """Measure a whole sequence."""
    result = TrackingResult(layout=layout, pivot_mm=pivot_mm)
    for index, image in enumerate(images):
        result.frames.append(
            track_frame(image, layout, index=index, pivot_mm=pivot_mm, calibration=calibration)
        )
    return result
