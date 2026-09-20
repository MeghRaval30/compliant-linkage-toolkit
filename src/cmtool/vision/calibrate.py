"""Camera calibration from checkerboard images.

A homography maps a plane projectively. Lens distortion is not projective, so it
cannot be absorbed and has to be removed first. Skipping this step does not make
the measurement noisier -- it makes it *biased*, in a way that varies across the
frame and therefore looks exactly like a real path deviation. That is the worst
kind of error for this project, because it would be mistaken for the signal.

Shoot the checkerboard at the same focal length and focus distance as the
mechanism footage. Phone cameras change both when they refocus, so lock focus if
the app allows it, and re-calibrate if anything about the setup changes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cmtool.core.units import FloatArray, ImageArray

#: Inner-corner counts of the checkerboard, not square counts. A 10x7 board of
#: squares has 9x6 inner corners.
DEFAULT_PATTERN = (9, 6)


class CalibrationError(RuntimeError):
    """Raised when calibration cannot be completed."""


@dataclass(frozen=True)
class CameraCalibration:
    """Intrinsics and distortion coefficients for one camera setup."""

    camera_matrix: FloatArray
    distortion: FloatArray
    image_size: tuple[int, int]
    reprojection_rms_px: float
    n_images: int
    pattern: tuple[int, int]
    square_size_mm: float

    def undistort(self, image: ImageArray) -> ImageArray:
        """Remove lens distortion from an image."""
        return np.asarray(cv2.undistort(np.asarray(image), self.camera_matrix, self.distortion))

    def undistort_points(self, points_px: FloatArray) -> FloatArray:
        """Remove lens distortion from image points, keeping pixel units."""
        points = np.asarray(points_px, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.undistortPoints(points, self.camera_matrix, self.distortion, P=self.camera_matrix)
        return np.asarray(out, dtype=float).reshape(-1, 2)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "camera_matrix": [list(row) for row in self.camera_matrix],
            "distortion": list(np.asarray(self.distortion).ravel()),
            "image_size": list(self.image_size),
            "reprojection_rms_px": self.reprojection_rms_px,
            "n_images": self.n_images,
            "pattern": list(self.pattern),
            "square_size_mm": self.square_size_mm,
        }

    def save(self, path: str | Path) -> Path:
        """Write the calibration to JSON."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return out

    @classmethod
    def load(cls, path: str | Path) -> CameraCalibration:
        """Read a calibration back from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            camera_matrix=np.asarray(data["camera_matrix"], dtype=float),
            distortion=np.asarray(data["distortion"], dtype=float),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            reprojection_rms_px=float(data["reprojection_rms_px"]),
            n_images=int(data["n_images"]),
            pattern=(int(data["pattern"][0]), int(data["pattern"][1])),
            square_size_mm=float(data["square_size_mm"]),
        )


def _object_points(pattern: tuple[int, int], square_size_mm: float) -> FloatArray:
    grid = np.zeros((pattern[0] * pattern[1], 3), dtype=float)
    grid[:, :2] = np.mgrid[0 : pattern[0], 0 : pattern[1]].T.reshape(-1, 2)
    return grid * float(square_size_mm)


def find_corners(
    image: ImageArray, pattern: tuple[int, int] = DEFAULT_PATTERN
) -> FloatArray | None:
    """Find checkerboard inner corners, refined to sub-pixel, or ``None``."""
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        array,
        pattern,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-4)
    refined = cv2.cornerSubPix(array, corners, (7, 7), (-1, -1), criteria)
    return np.asarray(refined, dtype=float).reshape(-1, 2)


def calibrate(
    images: list[ImageArray],
    *,
    pattern: tuple[int, int] = DEFAULT_PATTERN,
    square_size_mm: float = 10.0,
    min_images: int = 6,
) -> CameraCalibration:
    """Calibrate from a set of checkerboard images.

    Parameters
    ----------
    min_images
        Fewest usable views to accept. Distortion coefficients are poorly
        determined from a handful of similar views; aim for a dozen or more, with
        the board tilted differently in each.

    Raises
    ------
    CalibrationError
        If too few views contain a complete board.
    """
    object_points: list[Any] = []
    image_points: list[Any] = []
    size: tuple[int, int] | None = None

    for image in images:
        array = np.asarray(image)
        grey: ImageArray = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY) if array.ndim == 3 else array
        corners = find_corners(grey, pattern)
        if corners is None:
            continue
        size = (int(grey.shape[1]), int(grey.shape[0]))
        object_points.append(np.asarray(_object_points(pattern, square_size_mm), dtype=np.float32))
        image_points.append(np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2))

    if size is None or len(image_points) < min_images:
        raise CalibrationError(
            f"found a complete {pattern[0]}x{pattern[1]} board in only "
            f"{len(image_points)} of {len(images)} images; at least {min_images} are "
            "needed. Check that the whole board is in frame and in focus."
        )

    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, size, None, None
    )
    return CameraCalibration(
        camera_matrix=np.asarray(matrix, dtype=float),
        distortion=np.asarray(distortion, dtype=float),
        image_size=size,
        reprojection_rms_px=float(rms),
        n_images=len(image_points),
        pattern=pattern,
        square_size_mm=float(square_size_mm),
    )
