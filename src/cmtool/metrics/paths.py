"""Comparing coupler paths: reading measured ones, and measuring the gap.

The project's headline number is a distance between two curves -- rigid against
PRBM, PRBM against beam FEA, all three against what the camera measured. This
module is where that distance is defined, so every figure, report and dataset
sample quotes the same one.

Three numbers, because they answer different questions
------------------------------------------------------
``mean_mm`` / ``max_mm``
    Pointwise, at **matched input angles**. This is the honest comparison when
    both curves are parameterised the same way, and it is what the sample schema
    stores. It needs the two curves sampled at the same angles; use
    :func:`resample_to_angles` when they are not.

``frechet_mm``
    Discrete Frechet distance: the shortest leash needed to walk both curves
    forward together. It compares curves as *shapes*, ignoring how each was
    sampled, which is what makes it the right measure against a measured path
    whose frames land wherever the camera happened to catch them.

A pointwise comparison of two curves sampled at different rates is a common way
to manufacture a discrepancy that is not there, which is why the two are kept
apart rather than averaged into one "error".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from cmtool.core.units import FloatArray


@dataclass(frozen=True)
class MeasuredPath:
    """A coupler path read back from a tracked CSV.

    Attributes
    ----------
    input_angles_deg
        Input-link orientation at each row. ``NaN`` where the tracker could not
        read the lever, which is a gap in the record, never a zero.
    points_mm
        ``(N, 2)`` coupler positions, in mm.
    source
        The file it came from, kept for provenance.
    """

    input_angles_deg: FloatArray
    points_mm: FloatArray
    source: str

    @property
    def n_points(self) -> int:
        """Number of tracked rows."""
        return int(self.points_mm.shape[0])

    @property
    def has_angles(self) -> bool:
        """Whether every row carries an input angle, so it can be matched by angle."""
        return bool(self.input_angles_deg.size) and not bool(
            np.any(np.isnan(self.input_angles_deg))
        )


def read_path_csv(
    path: str | Path,
    *,
    x_column: str = "coupler_x_mm",
    y_column: str = "coupler_y_mm",
    angle_column: str = "input_angle_deg",
) -> MeasuredPath:
    """Read a measured path written by ``cmtool track``.

    Parameters
    ----------
    path
        CSV with ``input_angle_deg``, ``coupler_x_mm`` and ``coupler_y_mm``
        columns, as :meth:`cmtool.vision.track.TrackingResult.write_csv` writes.
    x_column, y_column, angle_column
        Column names, for reading a file written by something else.

    Raises
    ------
    ValueError
        If the coordinate columns are missing, or no row parses. An empty
        measured file is an error rather than an empty path, because silently
        returning nothing would let a figure claim "no measurement yet" when
        what actually happened is that the file is broken.
    """
    source = Path(path)
    rows = list(csv.DictReader(source.read_text(encoding="utf-8").splitlines()))
    if rows and (x_column not in rows[0] or y_column not in rows[0]):
        raise ValueError(
            f"{source}: expected columns {x_column!r} and {y_column!r}, found {list(rows[0])}"
        )

    angles: list[float] = []
    points: list[list[float]] = []
    for row in rows:
        try:
            points.append([float(row[x_column]), float(row[y_column])])
        except (KeyError, TypeError, ValueError):
            continue
        raw = (row.get(angle_column) or "").strip()
        angles.append(float(raw) if raw else float("nan"))

    if not points:
        raise ValueError(f"{source}: no rows with usable coordinates")

    return MeasuredPath(
        input_angles_deg=np.asarray(angles, dtype=float),
        points_mm=np.asarray(points, dtype=float),
        source=str(source),
    )


def resample_to_angles(
    angles_deg: FloatArray, points_mm: FloatArray, at_deg: FloatArray
) -> FloatArray:
    """Linearly interpolate a path onto a different set of input angles.

    Parameters
    ----------
    angles_deg
        Monotonic input angles the path is sampled at. A decreasing sweep is
        accepted and reversed internally.
    points_mm
        ``(N, 2)`` points at those angles.
    at_deg
        Angles to interpolate onto. Angles outside the source range are clamped
        to its ends, so the result never extrapolates a curve the solver never
        computed.
    """
    angles = np.asarray(angles_deg, dtype=float)
    points = np.asarray(points_mm, dtype=float)
    if angles.size != points.shape[0]:
        raise ValueError(f"{angles.size} angles for {points.shape[0]} points")
    if angles.size < 2:
        raise ValueError("need at least two samples to interpolate")

    order = np.argsort(angles)
    angles = angles[order]
    points = points[order]
    target = np.clip(np.asarray(at_deg, dtype=float), angles[0], angles[-1])
    return np.column_stack(
        [np.interp(target, angles, points[:, axis]) for axis in range(points.shape[1])]
    )


@dataclass(frozen=True)
class PathComparison:
    """How far apart two coupler paths are.

    Attributes
    ----------
    mean_mm, max_mm, rms_mm
        Pointwise distances at matched samples.
    frechet_mm
        Discrete Frechet distance: a shape comparison, independent of sampling.
    n_points
        Number of matched samples the pointwise numbers came from.
    labels
        Which two series were compared, for figure captions and reports.
    """

    mean_mm: float
    max_mm: float
    rms_mm: float
    frechet_mm: float
    n_points: int
    labels: tuple[str, str] = ("a", "b")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "a": self.labels[0],
            "b": self.labels[1],
            "mean_mm": self.mean_mm,
            "max_mm": self.max_mm,
            "rms_mm": self.rms_mm,
            "frechet_mm": self.frechet_mm,
            "n_points": self.n_points,
        }


def compare_paths(
    first_mm: FloatArray,
    second_mm: FloatArray,
    *,
    labels: tuple[str, str] = ("a", "b"),
) -> PathComparison:
    """Compare two paths pointwise and by shape.

    The pointwise numbers require the two arrays to have the same length and to
    correspond sample for sample -- matched input angles. The Frechet distance
    does not, and is computed whatever the lengths are.

    Raises
    ------
    ValueError
        If either path is empty, or the lengths differ (which would make a
        pointwise comparison meaningless; resample first).
    """
    a = np.asarray(first_mm, dtype=float)
    b = np.asarray(second_mm, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("cannot compare an empty path")
    if a.shape != b.shape:
        raise ValueError(
            f"pointwise comparison needs matched samples, got {a.shape} and {b.shape}; "
            "resample_to_angles() first"
        )

    distances = np.linalg.norm(a - b, axis=1)
    return PathComparison(
        mean_mm=float(np.mean(distances)),
        max_mm=float(np.max(distances)),
        rms_mm=float(np.sqrt(np.mean(distances**2))),
        frechet_mm=discrete_frechet(a, b),
        n_points=int(distances.size),
        labels=labels,
    )


def discrete_frechet(first_mm: FloatArray, second_mm: FloatArray) -> float:
    """Return the discrete Frechet distance between two polylines, in mm.

    The classic coupled-walk recursion of Eiter and Mannila, iterated rather than
    recursed so a long measured take cannot blow the stack. Both walkers move
    forward only, so unlike a nearest-neighbour distance this cannot be fooled by
    a curve that doubles back on itself -- which a coupler path frequently does.

    Cost is ``O(n m)`` in time and ``O(m)`` in memory.

    The sampling caveat
    -------------------
    This is the **discrete** Frechet distance: the walkers stop only at vertices,
    never partway along an edge. It therefore bounds the continuous distance from
    above, by at most the longer of the two curves' vertex spacings -- two
    parallel lines 2 mm apart, one sampled every 1 mm and the other every 0.1 mm,
    measure 2.06 mm rather than 2.00 mm.

    That matters here because a measured take has far more frames than a solver
    sweep has states. Two curves sampled at *matched* input angles are unaffected,
    since walking them in step is an admissible coupling, so
    :func:`compare_paths` quotes a number no larger than its own pointwise
    maximum. When the sampling differs, resample onto common angles with
    :func:`resample_to_angles` first; when there are no angles to match on, the
    residual inflation is part of the reported figure and is roughly the coarser
    curve's point spacing.
    """
    a = np.asarray(first_mm, dtype=float)
    b = np.asarray(second_mm, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("cannot compare an empty path")

    previous = np.empty(b.shape[0])
    current = np.empty(b.shape[0])

    for i in range(a.shape[0]):
        row = np.linalg.norm(b - a[i], axis=1)
        for j in range(b.shape[0]):
            if i == 0 and j == 0:
                current[j] = row[0]
            elif i == 0:
                current[j] = max(current[j - 1], row[j])
            elif j == 0:
                current[j] = max(previous[0], row[0])
            else:
                current[j] = max(min(previous[j], previous[j - 1], current[j - 1]), row[j])
        previous, current = current, previous

    return float(previous[-1])
