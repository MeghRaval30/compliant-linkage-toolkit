"""Measuring input torque, and comparing it against the models.

Why torque and not just the path
--------------------------------
With a prescribed input angle, one degree of freedom and no external load, the
coupler path is fixed by geometry: doubling every flexure's stiffness leaves the
path untouched. So the path, on its own, **cannot test the stiffness model at
all**. It tests the kinematics and the pivot placement, which matter, but nothing
about ``K``.

Input torque is directly proportional to stiffness. It is the measurement that
puts the PRBM's ``K``, the beam FEA's distributed compliance and the measured
modulus on the line together. For the pilot mechanisms the two models disagree by
about a third on peak torque, which is far above the resolution of the rigs
below, so this discriminates between them rather than merely confirming both.

Two rigs
--------
**Mode A, dead weight over a pulley (primary).** The mechanism lies flat, so the
pull on the lever is *horizontal* -- which a kitchen or luggage scale, weighing
vertically, cannot read. Instead: tie a thread to the lever hole, run it
horizontally to a pulley at the table edge, and hang known weights. The
mechanism settles at an equilibrium angle, read from the same camera that
measures the path.

The subtlety is that the thread direction is **not** fixed. As the lever swings,
the hole moves and the thread swings with it, so the moment arm changes through
the sweep. The applied torque is

``T = W * cross(H - A, u)``

where ``H`` is the hole, ``A`` the input pivot, and ``u`` the unit vector from the
hole toward the pulley. The cross product *is* the effective moment arm; assuming
it stays at ``r`` would be wrong by however much the geometry rotates.

**Mode B, spring scale (backup).** Force read directly at a known angle. Kept
because it needs no pulley, but it is awkward in-plane and less repeatable.

Hysteresis
----------
Take readings loading and unloading. PLA is viscoelastic, so the branches will
not coincide. **The gap is hysteresis plus rig friction, not hysteresis alone** --
a pulley and a thread have their own stiction, and it acts in the same direction
as material hysteresis. Separating them needs a friction characterisation of the
rig on its own; until that exists the reported number is an upper bound on the
material's contribution, and it is labelled as such.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from cmtool.core.units import FloatArray, cross2

#: Standard gravity, m/s^2. Converts hung masses to force.
GRAVITY_M_PER_S2 = 9.80665

#: Columns of the dead-weight template (mode A).
WEIGHT_COLUMNS = (
    "specimen_id",
    "printer",
    "print_purpose",
    "sequence",
    "step",
    "hung_mass_g",
    "measured_input_angle_deg",
    "pulley_x_mm",
    "pulley_y_mm",
    "hole_radius_mm",
    "settle_time_s",
    "repeat",
    "notes",
)

#: Columns of the spring-scale template (mode B).
SCALE_COLUMNS = (
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

Mode = Literal["weight", "scale"]


def hole_position_mm(
    pivot_mm: FloatArray,
    hole_mm: FloatArray,
    input_angle_deg: float,
    reference_angle_deg: float,
) -> FloatArray:
    """Where the lever hole sits after the input link rotates.

    The hole is rigidly attached to the input link, so it swings on a circle
    about the input pivot.
    """
    delta = np.radians(float(input_angle_deg) - float(reference_angle_deg))
    offset = np.asarray(hole_mm, dtype=float) - np.asarray(pivot_mm, dtype=float)
    rotation = np.array([[np.cos(delta), -np.sin(delta)], [np.sin(delta), np.cos(delta)]])
    return np.asarray(pivot_mm, dtype=float) + rotation @ offset


def moment_arm_mm(pivot_mm: FloatArray, hole_mm: FloatArray, pulley_mm: FloatArray) -> float:
    """Effective moment arm of a thread from the hole to the pulley, in mm.

    ``cross(H - A, u)`` with ``u`` the unit vector toward the pulley. This is the
    quantity that changes as the lever rotates, and the reason the naive ``r``
    is not good enough.
    """
    hole = np.asarray(hole_mm, dtype=float)
    pivot = np.asarray(pivot_mm, dtype=float)
    along = np.asarray(pulley_mm, dtype=float) - hole
    distance = float(np.linalg.norm(along))
    if distance <= 0.0:
        raise ValueError("the pulley cannot sit on the lever hole")
    return cross2(hole - pivot, along / distance)


def weight_to_force_n(mass_g: float) -> float:
    """Convert a hung mass in grams to a thread tension in newtons."""
    return float(mass_g) / 1000.0 * GRAVITY_M_PER_S2


# ------------------------------------------------------------------ templates


def write_weight_template(
    path: str | Path,
    *,
    hole_radius_mm: float,
    masses_g: tuple[float, ...] = (0.0, 20.0, 50.0, 100.0, 200.0, 500.0),
    repeats: int = 3,
    printer: str = "",
    pulley_mm: tuple[float, float] | None = None,
) -> Path:
    """Write the dead-weight (mode A) data-entry template.

    Each repeat runs the masses up and then back down, so loading and unloading
    are recorded as a matched pair.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    pulley_x = f"{pulley_mm[0]:.2f}" if pulley_mm else ""
    pulley_y = f"{pulley_mm[1]:.2f}" if pulley_mm else ""

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(WEIGHT_COLUMNS)
        for repeat in range(1, repeats + 1):
            for sequence, order in (
                ("loading", list(masses_g)),
                ("unloading", list(reversed(masses_g))),
            ):
                for step, mass in enumerate(order, start=1):
                    writer.writerow(
                        [
                            "",
                            printer,
                            "data",
                            sequence,
                            step,
                            f"{mass:.1f}",
                            "",
                            pulley_x,
                            pulley_y,
                            f"{hole_radius_mm:.2f}",
                            "",
                            repeat,
                            "",
                        ]
                    )
        handle.write("\n")
        for line in (
            "# MODE A, dead weight over a pulley. This is the primary rig.",
            "# The mechanism lies flat, so the pull on the lever is horizontal and a",
            "# kitchen scale (which weighs vertically) cannot read it. A thread over a",
            "# pulley at the table edge converts a hanging weight into a horizontal pull.",
            "#",
            "# Setup:",
            "#   - tie thread through the lever hole; keep it at the MID-THICKNESS of the",
            "#     part so it applies no out-of-plane moment",
            "#   - run the thread horizontally to a pulley clamped at the table edge",
            "#   - set the pulley height to the same mid-plane",
            "#   - record the pulley position in the SAME mm frame as the mechanism",
            "#     (read it from the camera image, or measure it from the fixture)",
            "#",
            "# Procedure: hang each mass, let it settle for a stated time, photograph or",
            "# film the mechanism, read the input angle from the markers, write it here.",
            "#",
            "# The thread direction changes as the lever swings, so the moment arm is not",
            "# constant: the analysis computes cross(H - A, u) per reading rather than",
            "# assuming the hole radius.",
            "#",
            "# Record settle_time_s: PLA creeps, so the angle depends on how long you wait.",
            "# The loading/unloading gap is hysteresis PLUS pulley and thread friction.",
            "# print_purpose: trial while settings are still moving, data once frozen.",
        ):
            handle.write(line + "\n")
    return out


def write_scale_template(
    path: str | Path,
    *,
    lever_radius_mm: float,
    input_range_deg: tuple[float, float],
    n_points: int = 9,
    repeats: int = 3,
    printer: str = "",
) -> Path:
    """Write the spring-scale (mode B) data-entry template, kept as a backup rig."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    angles = np.linspace(float(input_range_deg[0]), float(input_range_deg[1]), n_points)

    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCALE_COLUMNS)
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
        for line in (
            "# MODE B, spring scale. Backup rig: no pulley needed, but the pull is",
            "# horizontal while the scale weighs vertically, so it is awkward in-plane",
            "# and less repeatable than the dead-weight rig. Prefer mode A.",
            "# T = F * r * sin(pull_angle_from_lever_deg); pull perpendicular if you can.",
            "# The loading/unloading gap is hysteresis plus any rig friction.",
        ):
            handle.write(line + "\n")
    return out


# ------------------------------------------------------------------- readings


@dataclass(frozen=True)
class TorqueReading:
    """One measured point, reduced to an applied torque at a measured angle."""

    input_angle_deg: float
    torque_nmm: float
    direction: str
    repeat: int = 1
    mode: Mode = "scale"
    moment_arm_mm: float | None = None
    hung_mass_g: float | None = None


def read_scale_measurements(path: str | Path) -> list[TorqueReading]:
    """Read a filled-in mode B template."""
    readings: list[TorqueReading] = []
    for row in _rows(path):
        if not (row.get("force_N") or "").strip():
            continue
        radius = float(row["lever_radius_mm"])
        pull = float(row.get("pull_angle_from_lever_deg") or 90.0)
        arm = radius * float(np.sin(np.radians(pull)))
        readings.append(
            TorqueReading(
                input_angle_deg=float(row["input_angle_deg"]),
                torque_nmm=float(row["force_N"]) * arm,
                direction=(row.get("direction") or "loading").strip().lower(),
                repeat=int(row.get("repeat") or 1),
                mode="scale",
                moment_arm_mm=arm,
            )
        )
    return readings


def read_weight_measurements(
    path: str | Path,
    *,
    pivot_mm: FloatArray,
    hole_mm: FloatArray,
    reference_angle_deg: float,
) -> list[TorqueReading]:
    """Read a filled-in mode A template, resolving the changing moment arm.

    Parameters
    ----------
    pivot_mm, hole_mm
        Input pivot and lever hole in the reference (as-printed) configuration,
        from the CAD layout.
    reference_angle_deg
        Input angle of that reference configuration.
    """
    readings: list[TorqueReading] = []
    for row in _rows(path):
        angle_text = (row.get("measured_input_angle_deg") or "").strip()
        if not angle_text:
            continue
        angle = float(angle_text)
        pulley = np.array([float(row["pulley_x_mm"]), float(row["pulley_y_mm"])], dtype=float)
        hole_now = hole_position_mm(pivot_mm, hole_mm, angle, reference_angle_deg)
        arm = moment_arm_mm(pivot_mm, hole_now, pulley)
        mass = float(row.get("hung_mass_g") or 0.0)
        readings.append(
            TorqueReading(
                input_angle_deg=angle,
                torque_nmm=weight_to_force_n(mass) * arm,
                direction=(row.get("sequence") or "loading").strip().lower(),
                repeat=int(row.get("repeat") or 1),
                mode="weight",
                moment_arm_mm=arm,
                hung_mass_g=mass,
            )
        )
    return readings


def _rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(
            csv.DictReader(line for line in handle if not line.startswith("#") and line.strip())
        )


def read_measurements(path: str | Path, **kwargs: Any) -> list[TorqueReading]:
    """Read either template, detecting the mode from its columns."""
    with Path(path).open(newline="", encoding="utf-8") as handle:
        header = next(
            csv.reader(line for line in handle if not line.startswith("#") and line.strip())
        )
    if "hung_mass_g" in header:
        return read_weight_measurements(path, **kwargs)
    return read_scale_measurements(path)


# ---------------------------------------------------------------- comparison


@dataclass
class TorqueComparison:
    """Measured torque against one or more model predictions."""

    readings: list[TorqueReading]
    model_errors: dict[str, dict[str, float]] = field(default_factory=dict)
    angle_errors: dict[str, dict[str, float]] = field(default_factory=dict)
    hysteresis_nmm: float | None = None
    hysteresis_fraction: float | None = None

    @property
    def n_readings(self) -> int:
        """How many measured points were used."""
        return len(self.readings)

    @property
    def mode(self) -> str | None:
        """Which rig produced these readings."""
        return self.readings[0].mode if self.readings else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "mode": self.mode,
            "n_readings": self.n_readings,
            "model_errors": self.model_errors,
            "angle_errors": self.angle_errors,
            "hysteresis_nmm": self.hysteresis_nmm,
            "hysteresis_fraction": self.hysteresis_fraction,
            "hysteresis_note": (
                "includes rig friction (pulley, thread) as well as material hysteresis; "
                "an upper bound on the material's contribution"
            ),
        }


def _interpolate(angles_deg: FloatArray, values: FloatArray, at_deg: FloatArray) -> FloatArray:
    order = np.argsort(angles_deg)
    return np.interp(at_deg, np.asarray(angles_deg)[order], np.asarray(values)[order])


def _predicted_angle(
    angles_deg: FloatArray, torque_nmm: FloatArray, applied_nmm: float
) -> float | None:
    """Angle at which a model's restoring torque balances an applied torque."""
    angles = np.asarray(angles_deg, dtype=float)
    torque = np.asarray(torque_nmm, dtype=float)
    order = np.argsort(torque)
    sorted_torque = torque[order]
    if applied_nmm < sorted_torque[0] or applied_nmm > sorted_torque[-1]:
        return None
    return float(np.interp(applied_nmm, sorted_torque, angles[order]))


def compare(
    readings: list[TorqueReading],
    predictions: dict[str, tuple[FloatArray, FloatArray]],
) -> TorqueComparison:
    """Compare measured torque against model predictions.

    Reports two views, because the dead-weight rig naturally produces both:

    * **torque residual** at the measured angle -- how far the model's restoring
      torque is from the torque actually applied there;
    * **angle residual** -- where the model says the mechanism *should* have
      settled under that weight, against where it did.

    Loading and unloading are compared against the same prediction, because a
    quasi-static model has only one branch. Their measured separation is reported
    apart from the model error, as hysteresis plus rig friction.
    """
    comparison = TorqueComparison(readings=readings)
    if not readings:
        return comparison

    measured_angles = np.array([r.input_angle_deg for r in readings], dtype=float)
    measured_torque = np.array([r.torque_nmm for r in readings], dtype=float)
    scale = float(np.max(np.abs(measured_torque))) or 1.0

    for name, (angles, torque) in predictions.items():
        predicted = _interpolate(np.asarray(angles), np.asarray(torque), measured_angles)
        error = np.abs(predicted - measured_torque)
        comparison.model_errors[name] = {
            "mean_abs_error_nmm": float(np.mean(error)),
            "max_abs_error_nmm": float(np.max(error)),
            "mean_relative_error": float(np.mean(error) / scale),
            "bias_nmm": float(np.mean(predicted - measured_torque)),
        }

        residuals = []
        for reading in readings:
            angle = _predicted_angle(angles, torque, reading.torque_nmm)
            if angle is not None:
                residuals.append(angle - reading.input_angle_deg)
        if residuals:
            values = np.asarray(residuals, dtype=float)
            comparison.angle_errors[name] = {
                "mean_abs_error_deg": float(np.mean(np.abs(values))),
                "max_abs_error_deg": float(np.max(np.abs(values))),
                "bias_deg": float(np.mean(values)),
                "n_in_range": len(residuals),
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
        peak = float(np.max(np.abs(up))) or 1.0
        comparison.hysteresis_nmm = gap
        comparison.hysteresis_fraction = gap / peak

    return comparison
