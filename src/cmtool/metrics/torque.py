"""Measuring input torque, and comparing it against the models.

Why torque and not just the path
--------------------------------
With a prescribed input angle, one degree of freedom and no external load, the
coupler path is fixed by geometry: doubling every flexure's stiffness leaves the
path untouched. So the path, on its own, **cannot test the stiffness model at
all**. It tests the kinematics and the pivot placement, which matter, but nothing
about ``K``.

Input torque is directly proportional to stiffness. It is the measurement that
puts the PRBM's ``K = K_Theta EI/L``, the beam FEA's distributed compliance, and
the measured modulus on the line together. For the pilot mechanisms the two
models already disagree by about a third on peak torque, which is far larger than
a hand scale's resolution -- so this is a discriminating measurement, not a
formality.

The measurement
---------------
Hook a spring scale (kitchen or luggage) through the hole in the input lever, at
the known radius the CAD records, and pull **perpendicular to the lever**. Then

``T = F * r * sin(pull angle from the lever)``

with the pull angle 90 degrees in the ideal case. Record the angle anyway: a 10
degree error costs only 1.5 percent, but knowing it is what lets that claim be
checked rather than assumed.

Take readings **loading and unloading**. PLA is viscoelastic, so the two branches
will not coincide; the gap between them is hysteresis, and it is a result rather
than an error.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cmtool.core.units import FloatArray

#: Columns of the torque measurement template.
TEMPLATE_COLUMNS = (
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
)


def write_template(
    path: str | Path,
    *,
    lever_radius_mm: float,
    input_range_deg: tuple[float, float],
    n_points: int = 9,
    repeats: int = 3,
    printer: str = "",
) -> Path:
    """Write the angle-versus-force data-entry template.

    Rows are pre-filled with the angles to visit, going up and then back down, so
    the loading and unloading branches are recorded as a matched pair rather than
    two separate experiments.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(float(input_range_deg[0]), float(input_range_deg[1]), n_points)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(TEMPLATE_COLUMNS)
        for repeat in range(1, repeats + 1):
            for direction, sequence in (("loading", angles), ("unloading", angles[::-1])):
                for angle in sequence:
                    writer.writerow(
                        [
                            "",
                            printer,
                            "data",
                            direction,
                            f"{angle:.2f}",
                            "",
                            "90",
                            f"{lever_radius_mm:.2f}",
                            repeat,
                            "",
                        ]
                    )
        handle.write("\n")
        handle.write(
            "# T = F * r * sin(pull_angle_from_lever_deg); pull perpendicular if you can\n"
        )
        handle.write(
            "# record loading and unloading separately: the gap between them is hysteresis\n"
        )
        handle.write("# let the reading settle for a stated time at each angle -- PLA creeps\n")
        handle.write("# print_purpose: trial while settings are still moving, data once frozen\n")
    return out


@dataclass(frozen=True)
class TorqueReading:
    """One measured point."""

    input_angle_deg: float
    torque_nmm: float
    direction: str
    repeat: int = 1


def read_measurements(path: str | Path) -> list[TorqueReading]:
    """Read a filled-in torque template, converting force and radius to torque."""
    readings: list[TorqueReading] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(line for line in handle if not line.startswith("#")):
            if not (row.get("force_N") or "").strip():
                continue
            force = float(row["force_N"])
            radius = float(row["lever_radius_mm"])
            pull = float(row.get("pull_angle_from_lever_deg") or 90.0)
            readings.append(
                TorqueReading(
                    input_angle_deg=float(row["input_angle_deg"]),
                    # Force in newtons, radius in mm -> torque in N*mm.
                    torque_nmm=force * radius * float(np.sin(np.radians(pull))),
                    direction=(row.get("direction") or "loading").strip().lower(),
                    repeat=int(row.get("repeat") or 1),
                )
            )
    return readings


@dataclass
class TorqueComparison:
    """Measured torque against one or more model predictions."""

    readings: list[TorqueReading]
    model_errors: dict[str, dict[str, float]] = field(default_factory=dict)
    hysteresis_nmm: float | None = None
    hysteresis_fraction: float | None = None

    @property
    def n_readings(self) -> int:
        """How many measured points were used."""
        return len(self.readings)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "n_readings": self.n_readings,
            "model_errors": self.model_errors,
            "hysteresis_nmm": self.hysteresis_nmm,
            "hysteresis_fraction": self.hysteresis_fraction,
        }


def _interpolate(angles_deg: FloatArray, values: FloatArray, at_deg: FloatArray) -> FloatArray:
    order = np.argsort(angles_deg)
    return np.interp(at_deg, np.asarray(angles_deg)[order], np.asarray(values)[order])


def compare(
    readings: list[TorqueReading],
    predictions: dict[str, tuple[FloatArray, FloatArray]],
) -> TorqueComparison:
    """Compare measured torque against model predictions.

    Parameters
    ----------
    readings
        Measured points, from :func:`read_measurements`.
    predictions
        ``{model_name: (input_angles_deg, torque_nmm)}``, for example from
        ``simulate(cm, solver="prbm")`` and ``solver="beam_fea"``.

    Notes
    -----
    Loading and unloading are compared against the same prediction, because a
    quasi-static model has only one branch. The measured gap between them is
    reported separately as hysteresis -- a property of the material, not an error
    in the model.
    """
    comparison = TorqueComparison(readings=readings)
    if not readings:
        return comparison

    measured_angles = np.array([r.input_angle_deg for r in readings], dtype=float)
    measured_torque = np.array([r.torque_nmm for r in readings], dtype=float)

    for name, (angles, torque) in predictions.items():
        predicted = _interpolate(np.asarray(angles), np.asarray(torque), measured_angles)
        error = np.abs(predicted - measured_torque)
        scale = float(np.max(np.abs(measured_torque))) or 1.0
        comparison.model_errors[name] = {
            "mean_abs_error_nmm": float(np.mean(error)),
            "max_abs_error_nmm": float(np.max(error)),
            "mean_relative_error": float(np.mean(error) / scale),
            "bias_nmm": float(np.mean(predicted - measured_torque)),
        }

    loading = [r for r in readings if r.direction == "loading"]
    unloading = [r for r in readings if r.direction == "unloading"]
    if loading and unloading:
        common = np.array([r.input_angle_deg for r in loading], dtype=float)
        up = np.array([r.torque_nmm for r in loading], dtype=float)
        down = _interpolate(
            np.array([r.input_angle_deg for r in unloading], dtype=float),
            np.array([r.torque_nmm for r in unloading], dtype=float),
            common,
        )
        gap = float(np.mean(np.abs(up - down)))
        scale = float(np.max(np.abs(up))) or 1.0
        comparison.hysteresis_nmm = gap
        comparison.hysteresis_fraction = gap / scale

    return comparison
