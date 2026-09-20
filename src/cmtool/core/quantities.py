"""Physical quantities with provenance, and the placeholder discipline.

The project rule is: **never present a simulated or guessed number as measured**.
This module enforces it mechanically.

Every physical quantity in ``configs/`` is a block of the form::

    youngs_modulus_MPa:
      value: null                 # null until we have measured it
      provisional_value: 3500.0   # exists ONLY so code can execute
      status: placeholder         # measured | vendor | design_choice | placeholder
      source: "..."
      note: "PLACEHOLDER: replace with measured value"

:func:`Quantity.get` returns ``value`` when the quantity is real. When it is a
placeholder it returns ``provisional_value``, records the quantity's name in the
active :class:`~cmtool.core.provenance.Provenance`, and warns once. In strict
mode (``CMTOOL_STRICT_DATA=1``) it raises instead. Results computed from a
placeholder therefore always carry a ``placeholders_used`` list, and a figure made
from guessed data is self-identifying.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Literal

Status = Literal[
    "measured",
    "vendor",
    "literature",
    "design_choice",
    "confirmed",
    "placeholder",
]

#: Statuses that represent a real, defensible number.
#:
#: ``literature`` covers published model constants such as the PRBM
#: characteristic radius factor. They are somebody's result, not ours and not a
#: guess, so they are usable and reportable -- but they must carry a citation, and
#: where they have not yet been checked against our own solver the config says so.
REAL_STATUSES: frozenset[str] = frozenset(
    {"measured", "vendor", "literature", "design_choice", "confirmed"}
)


class ProvisionalDataWarning(UserWarning):
    """Warns that a placeholder (not measured) value was used in a computation."""


class MissingMeasurementError(RuntimeError):
    """Raised in strict mode when a placeholder value would have been used."""


def strict_mode() -> bool:
    """Return whether strict data mode is enabled via ``CMTOOL_STRICT_DATA``."""
    return os.environ.get("CMTOOL_STRICT_DATA", "").strip().lower() in {"1", "true", "yes"}


@dataclass(frozen=True)
class Quantity:
    """A physical quantity together with where it came from.

    Attributes
    ----------
    name
        Dotted path of the quantity, e.g. ``"PLA.youngs_modulus_MPa"``.
    value
        The real value, or ``None`` if not yet measured.
    provisional_value
        A stand-in so that code can execute. Never a measurement.
    status
        One of :data:`REAL_STATUSES` or ``"placeholder"``.
    source
        Free text describing where the number came from.
    note
        Free text, typically the instruction for replacing a placeholder.
    extra
        Any further provenance keys present in the config block (measurement
        date, rate, orientation, ...).
    """

    name: str
    value: float | None
    provisional_value: float | None
    status: Status
    source: str | None = None
    note: str | None = None
    extra: dict[str, Any] | None = None

    @property
    def is_placeholder(self) -> bool:
        """Whether this quantity is a placeholder rather than a real number."""
        return self.status not in REAL_STATUSES

    @classmethod
    def from_config(cls, name: str, block: Any) -> Quantity:
        """Build a :class:`Quantity` from a config entry.

        A bare scalar is accepted and treated as a placeholder, because a number
        with no stated provenance is exactly what this module exists to catch.
        """
        if not isinstance(block, dict):
            return cls(
                name=name,
                value=None,
                provisional_value=None if block is None else float(block),
                status="placeholder",
                source=None,
                note=f"PLACEHOLDER: {name} has no provenance block in config",
            )

        known = {"value", "provisional_value", "status", "source", "note"}
        status: Status = block.get("status", "placeholder")
        value = block.get("value")
        return cls(
            name=name,
            value=None if value is None else float(value),
            provisional_value=(
                None
                if block.get("provisional_value") is None
                else float(block["provisional_value"])
            ),
            status=status,
            source=block.get("source"),
            note=block.get("note"),
            extra={k: v for k, v in block.items() if k not in known} or None,
        )

    def get(self, provenance: ProvenanceSink | None = None) -> float:
        """Return the numeric value, recording and warning if it is provisional.

        Raises
        ------
        MissingMeasurementError
            In strict mode when the quantity is a placeholder, or in any mode
            when there is no usable number at all.
        """
        if not self.is_placeholder:
            if self.value is None:
                raise MissingMeasurementError(
                    f"{self.name} has status {self.status!r} but no value; fix the config"
                )
            return self.value

        if strict_mode():
            raise MissingMeasurementError(
                f"{self.name} is a placeholder and CMTOOL_STRICT_DATA is set. "
                f"{self.note or 'Measure it and put the value in the config.'}"
            )
        if self.provisional_value is None:
            raise MissingMeasurementError(
                f"{self.name} is a placeholder with no provisional_value; nothing to compute with. "
                f"{self.note or ''}".strip()
            )
        if provenance is not None:
            provenance.record_placeholder(self.name)
        warnings.warn(
            f"{self.name} is a PLACEHOLDER, not a measurement "
            f"(using provisional value {self.provisional_value}). "
            f"Results must not be reported as physical predictions.",
            ProvisionalDataWarning,
            stacklevel=2,
        )
        return self.provisional_value


class ProvenanceSink:
    """Minimal protocol-ish base: anything that can record a placeholder use."""

    def record_placeholder(self, name: str) -> None:
        """Record that quantity ``name`` was used provisionally."""
        raise NotImplementedError
