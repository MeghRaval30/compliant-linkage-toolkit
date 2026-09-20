"""Loading YAML configuration with provenance attached.

Configs are the only place physical numbers live. This module finds the config
tree, loads a file, and hands individual entries back as
:class:`~cmtool.core.quantities.Quantity` objects so that the placeholder
discipline (``docs/physics.md`` section 9) applies automatically wherever a number
is actually used.

Search order for the config root:

1. ``$CMTOOL_CONFIG_DIR``
2. a ``configs/`` directory in the current directory or any parent
3. a ``configs/`` directory above the installed package (the repo checkout)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from cmtool.core.provenance import canonical_hash
from cmtool.core.quantities import Quantity


class ConfigError(FileNotFoundError):
    """Raised when a configuration file or key cannot be found."""


def config_root() -> Path:
    """Return the directory holding the ``configs`` tree."""
    override = os.environ.get("CMTOOL_CONFIG_DIR")
    if override:
        path = Path(override)
        if not path.is_dir():
            raise ConfigError(f"CMTOOL_CONFIG_DIR={override!r} is not a directory")
        return path

    for start in (Path.cwd(), Path(__file__).resolve()):
        for parent in [start, *start.parents]:
            candidate = parent / "configs"
            if candidate.is_dir():
                return candidate
    raise ConfigError("could not find a 'configs' directory; set CMTOOL_CONFIG_DIR to point at one")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dictionary."""
    file = Path(path)
    if not file.is_file():
        raise ConfigError(f"no such config file: {file}")
    data = yaml.safe_load(file.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"config {file} did not parse to a mapping")
    return data


def load_config(kind: str, name: str) -> tuple[dict[str, Any], Path]:
    """Load ``configs/<kind>/<name>.yaml``; return the data and its path."""
    path = config_root() / kind / f"{name}.yaml"
    if not path.is_file():
        available = sorted(p.stem for p in (config_root() / kind).glob("*.yaml"))
        raise ConfigError(f"no {kind} config named {name!r}; available: {available}")
    return load_yaml(path), path


def dig(data: dict[str, Any], dotted: str) -> Any:
    """Return a nested value addressed by a dotted key, or raise :class:`ConfigError`."""
    node: Any = data
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            raise ConfigError(f"config key {dotted!r} not found")
        node = node[part]
    return node


def quantity(data: dict[str, Any], dotted: str, *, prefix: str) -> Quantity:
    """Return a config entry as a :class:`Quantity` named ``prefix.dotted``."""
    return Quantity.from_config(f"{prefix}.{dotted}", dig(data, dotted))


def config_hash(*configs: dict[str, Any]) -> str:
    """Return a stable digest over one or more resolved configuration mappings."""
    return canonical_hash(list(configs))
