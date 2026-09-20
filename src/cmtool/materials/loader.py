"""Materials and printers, loaded from config.

Neither class stores a number of its own. Every physical value is fetched through
a :class:`~cmtool.core.quantities.Quantity`, so using an unmeasured value records
it on the caller's :class:`~cmtool.core.provenance.Provenance` and marks the
result non-physical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from cmtool.core.config import ConfigError, dig, load_config, quantity
from cmtool.core.provenance import Provenance
from cmtool.core.quantities import Quantity


@dataclass(frozen=True)
class Material:
    """A material definition backed by ``configs/materials/<name>.yaml``."""

    name: str
    raw: dict[str, Any] = field(repr=False)
    path: Path | None = None

    @classmethod
    def load(cls, name: str) -> Material:
        """Load a material by config name (for example ``"PLA"``)."""
        data, path = load_config("materials", name.lower())
        return cls(name=data.get("name", name), raw=data, path=path)

    def quantity(self, key: str) -> Quantity:
        """Return one ``properties.<key>`` entry as a :class:`Quantity`."""
        return quantity(self.raw, f"properties.{key}", prefix=self.name)

    def youngs_modulus_mpa(self, provenance: Provenance | None = None) -> float:
        """Young's modulus in MPa (= N/mm^2)."""
        return self.quantity("youngs_modulus_MPa").get(provenance)

    def allowable_strain(self, provenance: Provenance | None = None) -> float:
        """Maximum allowable bending strain, dimensionless.

        This is the number that decides how far a flexure may bend, and therefore
        which designs are feasible at all.
        """
        return self.quantity("allowable_strain").get(provenance)

    def density_kg_per_m3(self, provenance: Provenance | None = None) -> float:
        """Density in kg/m^3."""
        return self.quantity("density_kg_per_m3").get(provenance)

    @property
    def supplier(self) -> dict[str, Any]:
        """Supplier block (brand, product line, colour, lot)."""
        value = self.raw.get("supplier", {})
        return value if isinstance(value, dict) else {}

    def unresolved(self) -> list[str]:
        """Names of properties that are still placeholders."""
        props = self.raw.get("properties", {})
        return [
            f"{self.name}.properties.{key}"
            for key in props
            if Quantity.from_config(key, props[key]).is_placeholder
        ]


@dataclass(frozen=True)
class Printer:
    """A printer and its process design rules, from ``configs/printer/<name>.yaml``."""

    name: str
    raw: dict[str, Any] = field(repr=False)
    path: Path | None = None

    @classmethod
    def load(cls, name: str) -> Printer:
        """Load a printer by config name (for example ``"bambu_a1"``)."""
        data, path = load_config("printer", name)
        return cls(name=data.get("name", name), raw=data, path=path)

    @property
    def model(self) -> str:
        """Manufacturer model string."""
        return str(self.raw.get("model", "unknown"))

    @property
    def role(self) -> str:
        """``"primary"`` or ``"secondary"``."""
        return str(self.raw.get("role", "primary"))

    def rule(self, key: str) -> Quantity:
        """Return one ``design_rules.<key>`` entry as a :class:`Quantity`."""
        return quantity(self.raw, f"design_rules.{key}", prefix=self.name)

    def nozzle_mm(self, provenance: Provenance | None = None) -> float:
        """Nozzle diameter in mm."""
        return quantity(self.raw, "nozzle_diameter_mm", prefix=self.name).get(provenance)

    def layer_height_mm(self, provenance: Provenance | None = None) -> float:
        """Layer height in mm."""
        return quantity(self.raw, "layer_height_mm", prefix=self.name).get(provenance)

    def min_flexure_thickness_mm(self, provenance: Provenance | None = None) -> float:
        """Thinnest flexure this process can produce reliably.

        Currently a placeholder on every printer: it is set from the A2 flexure
        test coupon, not guessed. See ``docs/walkthrough.md``.
        """
        return self.rule("min_flexure_thickness_mm").get(provenance)

    def part_thickness_mm(self, provenance: Provenance | None = None) -> float:
        """Out-of-plane depth of the printed part in mm."""
        return self.rule("part_thickness_mm").get(provenance)

    def min_link_width_mm(self, provenance: Provenance | None = None) -> float:
        """Minimum in-plane width of a rigid link in mm."""
        return self.rule("min_link_width_mm").get(provenance)

    def min_clearance_mm(self, provenance: Provenance | None = None) -> float:
        """Minimum gap between separate features in mm."""
        return self.rule("min_feature_clearance_mm").get(provenance)

    def design_envelope_mm(self) -> tuple[float, float]:
        """Usable design envelope ``(x, y)`` in mm.

        This is a team decision (parts must fit both printers), not a machine
        specification, so it is read directly rather than through a Quantity.
        """
        try:
            entry = dig(self.raw, "bed.design_envelope_mm")
        except ConfigError:
            raise ConfigError(f"printer {self.name!r} has no bed.design_envelope_mm") from None
        value = entry.get("value") if isinstance(entry, dict) else entry
        if value is None:
            raise ConfigError(f"printer {self.name!r} has no design envelope value")
        return (float(value[0]), float(value[1]))

    @property
    def slicer(self) -> dict[str, Any]:
        """Slicer guidance block."""
        value = self.raw.get("slicer", {})
        return value if isinstance(value, dict) else {}

    def unresolved(self) -> list[str]:
        """Names of design rules that are still placeholders."""
        rules = self.raw.get("design_rules", {})
        return [
            f"{self.name}.design_rules.{key}"
            for key in rules
            if Quantity.from_config(key, rules[key]).is_placeholder
        ]
