"""Synthetic scenes, so the vision pipeline can be validated without footage.

Every stage of the measurement chain -- detection, homography, pad tracking,
angle extraction -- can be checked against ground truth by rendering markers at
*known* millimetre positions through a *known* projective transform and asking
the pipeline to recover them.

This is not a substitute for the real go/no-go, which needs a real camera and a
real printed part. It is what makes the software side testable at all, and what
separates "the code is wrong" from "the rig is wrong" when the two disagree.

What these numbers are, and are not
-----------------------------------
The harness reproduces **shape** faithfully: a swept circle comes back with a
residual of about 14 micrometres regardless of tilt or resolution, which is what
validates detection, the homography and the pad tracking.

It carries its own **scale bias** of a few tenths of a percent, present even at
zero tilt, from the render-then-detect round trip: the detector's sub-pixel
corner estimate on a rasterised marker is not exactly the corner that was drawn.
That bias belongs to the harness, not to the pipeline.

So synthetic figures must not be quoted as the rig's accuracy. Use them to show
the software recovers what it should, and get the real uncertainty from real
footage with a caliper-measured reference radius -- which is precisely why the
known-motion test asks for one.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from cmtool.core.units import FloatArray, ImageArray
from cmtool.vision.markers import MarkerLayout, dictionary


@dataclass(frozen=True)
class SyntheticCamera:
    """A projective view of the mechanism plane.

    Attributes
    ----------
    image_size
        ``(width, height)`` in pixels.
    pixels_per_mm
        Scale at the plane origin.
    tilt
        Small out-of-plane perspective terms. Zero gives a pure scale and shift;
        non-zero produces genuine foreshortening, which is what a real tripod
        that is not perfectly square to the part produces.
    """

    image_size: tuple[int, int] = (1600, 1200)
    pixels_per_mm: float = 6.0
    origin_px: tuple[float, float] = (120.0, 1080.0)
    tilt: tuple[float, float] = (0.0, 0.0)
    rotation_deg: float = 0.0

    def matrix_mm_to_px(self) -> FloatArray:
        """Homography taking millimetres to pixels."""
        scale = self.pixels_per_mm
        angle = np.radians(self.rotation_deg)
        cos, sin = np.cos(angle), np.sin(angle)
        # +y is up in mm and down in image rows, hence the sign on the second row.
        base = np.array(
            [
                [scale * cos, -scale * sin, self.origin_px[0]],
                [-scale * sin, -scale * cos, self.origin_px[1]],
                [self.tilt[0], self.tilt[1], 1.0],
            ],
            dtype=float,
        )
        return np.asarray(base, dtype=float)

    def project(self, points_mm: FloatArray) -> FloatArray:
        """Project millimetre points into the image."""
        points = np.asarray(points_mm, dtype=float).reshape(-1, 1, 2)
        return np.asarray(
            cv2.perspectiveTransform(points, self.matrix_mm_to_px()), dtype=float
        ).reshape(-1, 2)

    @classmethod
    def fitting(
        cls,
        points_mm: FloatArray,
        *,
        image_size: tuple[int, int] = (1600, 1200),
        margin_px: float = 60.0,
        tilt: tuple[float, float] = (0.0, 0.0),
        rotation_deg: float = 0.0,
        iterations: int = 4,
    ) -> SyntheticCamera:
        """Build a camera that frames the given millimetre points.

        Solved iteratively because the perspective terms couple the scale to the
        position, which a single closed-form fit would get slightly wrong.
        """
        points = np.asarray(points_mm, dtype=float).reshape(-1, 2)
        width, height = image_size
        camera = cls(
            image_size=image_size,
            pixels_per_mm=1.0,
            origin_px=(0.0, 0.0),
            tilt=tilt,
            rotation_deg=rotation_deg,
        )
        for _ in range(iterations):
            projected = camera.project(points)
            lo, hi = projected.min(axis=0), projected.max(axis=0)
            span = np.maximum(hi - lo, 1e-9)
            factor = float(
                min((width - 2 * margin_px) / span[0], (height - 2 * margin_px) / span[1])
            )
            scale = camera.pixels_per_mm * factor
            trial = cls(
                image_size=image_size,
                pixels_per_mm=scale,
                origin_px=(0.0, 0.0),
                tilt=tilt,
                rotation_deg=rotation_deg,
            )
            shifted = trial.project(points)
            lo2, hi2 = shifted.min(axis=0), shifted.max(axis=0)
            origin = (
                float(margin_px - lo2[0] + (width - 2 * margin_px - (hi2[0] - lo2[0])) / 2.0),
                float(margin_px - lo2[1] + (height - 2 * margin_px - (hi2[1] - lo2[1])) / 2.0),
            )
            camera = cls(
                image_size=image_size,
                pixels_per_mm=scale,
                origin_px=origin,
                tilt=tilt,
                rotation_deg=rotation_deg,
            )
        return camera


def render(
    layout: MarkerLayout,
    camera: SyntheticCamera,
    *,
    moved_pads: dict[str, FloatArray] | None = None,
    marker_pixels: int | None = None,
    noise_sigma: float = 0.0,
    seed: int = 0,
) -> ImageArray:
    """Render a synthetic image of the layout.

    Parameters
    ----------
    moved_pads
        ``{pad_name: new_centre_mm}``, for pads that have moved since the layout
        was defined. This is how a swept mechanism is simulated: the base pads
        stay put and the lever and coupler pads move.
    marker_pixels
        Size of the source marker tile. Defaults to comfortably more than the
        marker's projected size, because **upsampling a tile biases the detected
        corners outward** and shows up as a scale error in the measurement --
        an artefact of the renderer that would otherwise be mistaken for one of
        the pipeline.
    noise_sigma
        Standard deviation of additive Gaussian pixel noise, for testing how the
        measurement degrades with image quality.
    """
    moved_pads = moved_pads or {}
    if marker_pixels is None:
        sizes = [
            float(np.linalg.norm(camera.project(corners)[1] - camera.project(corners)[0]))
            for pad in layout.pads
            for corners in pad.marker_corners_mm().values()
        ]
        marker_pixels = int(max(160, 2 * max(sizes) if sizes else 160))
    width, height = camera.image_size
    image: ImageArray = np.full((height, width), 235, dtype=np.uint8)
    book = dictionary(layout.dictionary_name)

    for pad in layout.pads:
        centre = np.asarray(moved_pads.get(pad.name, pad.centre_mm), dtype=float)
        shift = centre - np.asarray(pad.centre_mm, dtype=float)
        for marker_id, corners in pad.marker_corners_mm().items():
            target = camera.project(corners + shift).astype(np.float32)
            tile = cv2.aruco.generateImageMarker(book, marker_id, marker_pixels)
            tile = cv2.copyMakeBorder(tile, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
            size = tile.shape[0]
            # Pixel EDGES, not pixel centres. The marker's geometric corner is the
            # outer edge of its corner pixel, so using centres would render every
            # marker (n-1)/n of its intended size -- a 0.6 percent scale error at
            # 160 px, which would then show up as a bogus scale error in the
            # measurement and get mistaken for a real one.
            source = np.array(
                [
                    [7.5, 7.5],
                    [size - 8.5, 7.5],
                    [size - 8.5, size - 8.5],
                    [7.5, size - 8.5],
                ],
                dtype=np.float32,
            )
            warp = cv2.getPerspectiveTransform(source, target)
            patch = cv2.warpPerspective(
                tile,
                warp,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=255,
            )
            mask = cv2.warpPerspective(
                np.full_like(tile, 255),
                warp,
                (width, height),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            image = np.asarray(np.where(mask > 0, patch, image), dtype=np.uint8)

    if noise_sigma > 0.0:
        rng = np.random.default_rng(seed)
        noisy = image.astype(float) + rng.normal(0.0, noise_sigma, image.shape)
        image = np.asarray(np.clip(noisy, 0, 255), dtype=np.uint8)
    return image


def sweep_frames(
    layout: MarkerLayout,
    camera: SyntheticCamera,
    pad_paths: dict[str, FloatArray],
    **kwargs: object,
) -> list[ImageArray]:
    """Render a sequence of frames from per-pad millimetre paths.

    ``pad_paths`` maps a pad name to an ``(n, 2)`` array of centres, one per
    frame. Every path must have the same length.
    """
    lengths = {len(np.asarray(path)) for path in pad_paths.values()}
    if len(lengths) != 1:
        raise ValueError("every pad path must have the same number of frames")
    count = lengths.pop()

    options = dict(kwargs)
    base_seed = int(options.pop("seed", 0) or 0)  # type: ignore[call-overload]

    frames = []
    for index in range(count):
        moved = {name: np.asarray(path, dtype=float)[index] for name, path in pad_paths.items()}
        # A fresh seed per frame, so image noise is independent between frames as
        # it is in reality. Reusing one seed would correlate the noise and make
        # the measurement look far more repeatable than it is.
        frames.append(
            render(layout, camera, moved_pads=moved, seed=base_seed + index, **options)  # type: ignore[arg-type]
        )
    return frames
