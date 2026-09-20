"""Mapping the image to the mechanism plane, in millimetres.

The mechanism is planar and lies flat, so one homography takes image pixels to
millimetres for everything in that plane. It is solved from the **base fiducials
only**: they are bolted to ground, so their millimetre positions are known from
the CAD and do not change. Everything else -- lever, coupler -- is then read
through that mapping.

Two things this cannot absorb, and which therefore have to be controlled:

* **Lens distortion.** A homography is a projective map of a plane; barrel
  distortion is not. Calibrate and undistort first, or the error shows up as a
  position-dependent bias that looks exactly like a real path deviation.
* **Out-of-plane motion.** A marker that lifts out of the calibration plane
  appears displaced sideways. That is why every marker pad on the part is raised
  by the same amount, and why ``docs/physics.md`` section 7 bothers with the sag
  estimate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from cmtool.core.units import FloatArray
from cmtool.vision.markers import Detection, MarkerLayout


class HomographyError(RuntimeError):
    """Raised when the image-to-millimetre mapping cannot be established."""


@dataclass(frozen=True)
class PlaneMap:
    """A homography from image pixels to the mechanism plane in mm."""

    matrix: FloatArray
    n_correspondences: int
    reprojection_rms_px: float
    marker_ids: tuple[int, ...]

    def to_mm(self, points_px: FloatArray) -> FloatArray:
        """Map image points to millimetres."""
        points = np.asarray(points_px, dtype=float).reshape(-1, 1, 2)
        return np.asarray(cv2.perspectiveTransform(points, self.matrix), dtype=float).reshape(-1, 2)

    def to_px(self, points_mm: FloatArray) -> FloatArray:
        """Map millimetre points back to the image."""
        points = np.asarray(points_mm, dtype=float).reshape(-1, 1, 2)
        return np.asarray(
            cv2.perspectiveTransform(points, np.linalg.inv(self.matrix)), dtype=float
        ).reshape(-1, 2)

    def scale_mm_per_px(self) -> float:
        """Approximate local scale at the image centre, for sanity checks."""
        origin = self.to_mm(np.array([[0.0, 0.0]]))[0]
        unit_x = self.to_mm(np.array([[1.0, 0.0]]))[0]
        unit_y = self.to_mm(np.array([[0.0, 1.0]]))[0]
        return float(0.5 * (np.linalg.norm(unit_x - origin) + np.linalg.norm(unit_y - origin)))

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "n_correspondences": self.n_correspondences,
            "reprojection_rms_px": self.reprojection_rms_px,
            "marker_ids": list(self.marker_ids),
            "scale_mm_per_px": self.scale_mm_per_px(),
            "matrix": [list(row) for row in self.matrix],
        }


def solve_plane_map(
    detection: Detection,
    layout: MarkerLayout,
    *,
    min_markers: int = 2,
) -> PlaneMap:
    """Solve the image-to-mm homography from the fixed base fiducials.

    Parameters
    ----------
    min_markers
        Fewest fixed markers that may be used. Two 4-corner markers give eight
        correspondences for eight unknowns, which is the bare minimum; three or
        more is what makes the fit over-determined and the residual meaningful.

    Raises
    ------
    HomographyError
        If too few fixed markers were seen, or the fit does not converge.
    """
    fixed_ids: list[int] = []
    for pad in layout.fixed_pads():
        fixed_ids.extend(pad.marker_ids)
    corners_mm = layout.marker_corners_mm()

    image_points: list[FloatArray] = []
    plane_points: list[FloatArray] = []
    used: list[int] = []
    for marker_id in sorted(fixed_ids):
        if marker_id not in detection.corners:
            continue
        image_points.append(detection.corners[marker_id])
        plane_points.append(corners_mm[marker_id])
        used.append(marker_id)

    if len(used) < min_markers:
        raise HomographyError(
            f"only {len(used)} of {len(fixed_ids)} fixed markers were detected; "
            f"at least {min_markers} are needed. Check lighting, focus, and that the "
            "base fiducials are fully in frame."
        )

    source = np.vstack(image_points).astype(np.float64)
    target = np.vstack(plane_points).astype(np.float64)
    matrix, _ = cv2.findHomography(source, target, method=0)
    if matrix is None:
        raise HomographyError("homography fit failed; the marker correspondences are degenerate")

    reprojected = cv2.perspectiveTransform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residual_mm = np.linalg.norm(reprojected - target, axis=1)
    back = cv2.perspectiveTransform(target.reshape(-1, 1, 2), np.linalg.inv(matrix)).reshape(-1, 2)
    residual_px = float(np.sqrt(np.mean(np.sum((back - source) ** 2, axis=1))))

    _ = residual_mm  # kept for clarity; the pixel residual is the reported one
    return PlaneMap(
        matrix=np.asarray(matrix, dtype=float),
        n_correspondences=len(source),
        reprojection_rms_px=residual_px,
        marker_ids=tuple(used),
    )


def pad_centre_mm(
    detection: Detection, layout: MarkerLayout, plane: PlaneMap, pad_name: str
) -> FloatArray | None:
    """Millimetre position of a pad's centre, or ``None`` if it was not seen.

    Averaged over every marker corner on the pad, which is what the multi-marker
    pads are for: corner noise averages down.
    """
    pad = layout.pad(pad_name)
    points: list[FloatArray] = []
    for marker_id in pad.marker_ids:
        if marker_id in detection.corners:
            points.append(detection.corners[marker_id])
    if not points:
        return None

    corners_mm = layout.marker_corners_mm()
    measured = plane.to_mm(np.vstack(points))
    # Each marker's corners are offset from the pad centre by a known amount;
    # subtracting those offsets makes every corner an estimate of the centre.
    offsets = np.vstack(
        [
            corners_mm[marker_id] - np.asarray(pad.centre_mm, dtype=float)
            for marker_id in pad.marker_ids
            if marker_id in detection.corners
        ]
    )
    return np.asarray(np.mean(measured - offsets, axis=0), dtype=float)
