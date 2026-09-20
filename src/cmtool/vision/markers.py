"""ArUco marker layout, printable sheets, and detection.

Each tracked feature on the mechanism carries a **pad** of several markers rather
than one. A single tag's pose is noisy: its corners are only a few pixels apart
compared with the pad, so small corner errors turn into large angle errors. Four
markers spanning 30 mm give the homography four times as many correspondences
over a much longer baseline, for the cost of a slightly larger sticker.

Frames
------
Marker corners are listed in the order the ArUco detector reports them, which is
fixed relative to the marker's own pattern: top-left, top-right, bottom-right,
bottom-left, read with the pattern upright. The millimetre layout uses the same
order in a **+y-up** frame, so a marker printed upright on the pad corresponds
corner-for-corner with its detection. Nothing here depends on the camera's
orientation; the homography absorbs that.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from cmtool.core.units import FloatArray, ImageArray

#: ArUco dictionary used throughout. 4x4 patterns stay readable at small sizes,
#: and 50 ids is far more than this project needs.
DEFAULT_DICTIONARY = "DICT_4X4_50"


def dictionary(name: str = DEFAULT_DICTIONARY) -> Any:
    """Return a predefined ArUco dictionary by name."""
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"unknown ArUco dictionary {name!r}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


@dataclass(frozen=True)
class PadSpec:
    """A grid of markers stuck on one pad of the printed part."""

    name: str
    centre_mm: tuple[float, float]
    marker_ids: tuple[int, ...]
    grid: tuple[int, int] = (2, 2)
    marker_size_mm: float = 12.0
    gap_mm: float = 3.0
    fixed: bool = False

    def __post_init__(self) -> None:
        rows, cols = self.grid
        if len(self.marker_ids) != rows * cols:
            raise ValueError(
                f"pad {self.name!r} has {len(self.marker_ids)} ids for a {rows}x{cols} grid"
            )

    @property
    def pitch_mm(self) -> float:
        """Centre-to-centre spacing of adjacent markers."""
        return self.marker_size_mm + self.gap_mm

    @property
    def size_mm(self) -> tuple[float, float]:
        """Overall pad extent covered by markers."""
        rows, cols = self.grid
        return (
            cols * self.marker_size_mm + (cols - 1) * self.gap_mm,
            rows * self.marker_size_mm + (rows - 1) * self.gap_mm,
        )

    def marker_centres_mm(self) -> dict[int, FloatArray]:
        """Centre of each marker, in the mechanism's mm frame."""
        rows, cols = self.grid
        origin = np.asarray(self.centre_mm, dtype=float)
        out: dict[int, FloatArray] = {}
        for index, marker_id in enumerate(self.marker_ids):
            row, col = divmod(index, cols)
            offset = np.array(
                [
                    (col - (cols - 1) / 2.0) * self.pitch_mm,
                    ((rows - 1) / 2.0 - row) * self.pitch_mm,
                ]
            )
            out[marker_id] = origin + offset
        return out

    def marker_corners_mm(self) -> dict[int, FloatArray]:
        """Four corners of each marker, in detector order, in the mm frame."""
        half = self.marker_size_mm / 2.0
        pattern = np.array(
            [[-half, half], [half, half], [half, -half], [-half, -half]], dtype=float
        )
        return {
            marker_id: centre + pattern for marker_id, centre in self.marker_centres_mm().items()
        }


@dataclass(frozen=True)
class MarkerLayout:
    """Every marker on one mechanism, and where it sits in the mm frame."""

    pads: tuple[PadSpec, ...]
    dictionary_name: str = DEFAULT_DICTIONARY

    def __post_init__(self) -> None:
        seen: set[int] = set()
        for pad in self.pads:
            clash = seen & set(pad.marker_ids)
            if clash:
                raise ValueError(f"marker ids {sorted(clash)} used on more than one pad")
            seen |= set(pad.marker_ids)

    @property
    def pad_names(self) -> list[str]:
        """Names of every pad."""
        return [pad.name for pad in self.pads]

    def pad(self, name: str) -> PadSpec:
        """Look up a pad by name."""
        for pad in self.pads:
            if pad.name == name:
                return pad
        raise KeyError(f"no pad named {name!r}; have {self.pad_names}")

    def fixed_pads(self) -> tuple[PadSpec, ...]:
        """Return the pads rigidly attached to ground, which define the mm frame."""
        return tuple(pad for pad in self.pads if pad.fixed)

    def pad_of(self, marker_id: int) -> str | None:
        """Which pad a marker belongs to."""
        for pad in self.pads:
            if marker_id in pad.marker_ids:
                return pad.name
        return None

    def marker_corners_mm(self) -> dict[int, FloatArray]:
        """Corners of every marker, in the mm frame."""
        out: dict[int, FloatArray] = {}
        for pad in self.pads:
            out.update(pad.marker_corners_mm())
        return out

    def overlaps(self, clearance_mm: float = 2.0) -> list[tuple[str, str, float]]:
        """Return pads whose marker areas collide in the plane.

        Worth checking, because two markers printed on top of each other simply
        will not be detected -- and the failure looks like a lighting or focus
        problem rather than a layout one. Returns ``(pad, pad, gap_mm)`` for every
        pair closer than ``clearance_mm``, with a negative gap meaning genuine
        overlap.
        """
        found: list[tuple[str, str, float]] = []
        for index, first in enumerate(self.pads):
            for second in self.pads[index + 1 :]:
                gap = _rect_gap_mm(first, second)
                if gap < clearance_mm:
                    found.append((first.name, second.name, gap))
        return found

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description."""
        return {
            "dictionary": self.dictionary_name,
            "pads": [
                {
                    "name": pad.name,
                    "centre_mm": list(pad.centre_mm),
                    "marker_ids": list(pad.marker_ids),
                    "grid": list(pad.grid),
                    "marker_size_mm": pad.marker_size_mm,
                    "gap_mm": pad.gap_mm,
                    "fixed": pad.fixed,
                }
                for pad in self.pads
            ],
        }

    @classmethod
    def for_mechanism(
        cls,
        *,
        fiducial_pads_mm: list[tuple[float, float]],
        lever_pad_mm: tuple[float, float] | None,
        coupler_pad_mm: tuple[float, float] | None,
        marker_size_mm: float = 12.0,
        gap_mm: float = 3.0,
        dictionary_name: str = DEFAULT_DICTIONARY,
    ) -> MarkerLayout:
        """Build the layout from a :class:`~cmtool.cad.mechanism.MechanismLayout`.

        Base fiducials are marked ``fixed``: they are what the homography is
        solved from, because they do not move.
        """
        pads: list[PadSpec] = []
        next_id = 0

        def take(count: int) -> tuple[int, ...]:
            nonlocal next_id
            ids = tuple(range(next_id, next_id + count))
            next_id += count
            return ids

        for index, centre in enumerate(fiducial_pads_mm):
            pads.append(
                PadSpec(
                    name=f"base_{index}",
                    centre_mm=(float(centre[0]), float(centre[1])),
                    marker_ids=take(4),
                    grid=(2, 2),
                    marker_size_mm=marker_size_mm,
                    gap_mm=gap_mm,
                    fixed=True,
                )
            )
        if lever_pad_mm is not None:
            pads.append(
                PadSpec(
                    name="lever",
                    centre_mm=(float(lever_pad_mm[0]), float(lever_pad_mm[1])),
                    marker_ids=take(1),
                    grid=(1, 1),
                    marker_size_mm=marker_size_mm,
                    gap_mm=gap_mm,
                )
            )
        if coupler_pad_mm is not None:
            pads.append(
                PadSpec(
                    name="coupler",
                    centre_mm=(float(coupler_pad_mm[0]), float(coupler_pad_mm[1])),
                    marker_ids=take(1),
                    grid=(1, 1),
                    marker_size_mm=marker_size_mm,
                    gap_mm=gap_mm,
                )
            )
        return cls(pads=tuple(pads), dictionary_name=dictionary_name)


def _rect_gap_mm(first: PadSpec, second: PadSpec) -> float:
    """Smallest gap between two pads' marker footprints, negative if overlapping."""
    gaps = []
    for axis in (0, 1):
        half_a = first.size_mm[axis] / 2.0
        half_b = second.size_mm[axis] / 2.0
        distance = abs(first.centre_mm[axis] - second.centre_mm[axis])
        gaps.append(distance - (half_a + half_b))
    # Separated on either axis means separated; otherwise the deepest overlap.
    return float(max(gaps))


@dataclass
class Detection:
    """Markers found in one image."""

    corners: dict[int, FloatArray] = field(default_factory=dict)

    @property
    def ids(self) -> list[int]:
        """Ids detected, sorted."""
        return sorted(self.corners)

    def centre(self, marker_id: int) -> FloatArray:
        """Mean of a marker's four image corners."""
        return np.mean(self.corners[marker_id], axis=0)


def detect(image: ImageArray, layout: MarkerLayout) -> Detection:
    """Detect the layout's markers in an image.

    Accepts colour or greyscale; the detector wants greyscale, so colour is
    converted rather than refused.
    """
    array = np.asarray(image)
    if array.ndim == 3:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    detector = cv2.aruco.ArucoDetector(
        dictionary(layout.dictionary_name), cv2.aruco.DetectorParameters()
    )
    corners, ids, _ = detector.detectMarkers(array)

    found: dict[int, FloatArray] = {}
    if ids is not None:
        for quad, marker_id in zip(corners, ids.ravel(), strict=True):
            found[int(marker_id)] = np.asarray(quad, dtype=float).reshape(4, 2)
    return Detection(corners=found)


def marker_sheet(
    layout: MarkerLayout, *, pixels_per_mm: float = 12.0, margin_mm: float = 8.0
) -> dict[str, ImageArray]:
    """Render one printable image per pad, at a stated scale.

    Printing these at the right size matters: the mm layout assumes the marker
    is ``marker_size_mm`` across. Print at 100 percent scale and measure one
    marker with calipers before sticking anything down.
    """
    book = dictionary(layout.dictionary_name)
    sheets: dict[str, ImageArray] = {}

    for pad in layout.pads:
        width_mm, height_mm = pad.size_mm
        width = round((width_mm + 2 * margin_mm) * pixels_per_mm)
        height = round((height_mm + 2 * margin_mm) * pixels_per_mm)
        sheet: ImageArray = np.full((height, width), 255, dtype=np.uint8)
        size_px = round(pad.marker_size_mm * pixels_per_mm)

        centres = pad.marker_centres_mm()
        origin = np.asarray(pad.centre_mm, dtype=float)
        for marker_id, centre in centres.items():
            image = cv2.aruco.generateImageMarker(book, marker_id, size_px)
            offset = centre - origin
            # +y is up in mm, down in image rows.
            col = round((offset[0] + width_mm / 2.0 + margin_mm) * pixels_per_mm)
            row = round((height_mm / 2.0 - offset[1] + margin_mm) * pixels_per_mm)
            top, left = row - size_px // 2, col - size_px // 2
            sheet[top : top + size_px, left : left + size_px] = image
        sheets[pad.name] = sheet
    return sheets


def write_marker_sheets(
    layout: MarkerLayout, out_dir: str | Path, *, pixels_per_mm: float = 12.0
) -> dict[str, Path]:
    """Write the printable marker sheets as PNGs."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, sheet in marker_sheet(layout, pixels_per_mm=pixels_per_mm).items():
        path = directory / f"markers_{name}.png"
        cv2.imwrite(str(path), sheet)
        written[name] = path
    return written
