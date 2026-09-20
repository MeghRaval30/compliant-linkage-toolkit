"""Reproducibility bookkeeping: config hash, code commit, seed, placeholders.

Every generated sample and every figure carries a :class:`Provenance` record, so
that any number in the paper can be traced back to the exact code and config that
produced it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cmtool.core.quantities import ProvenanceSink


def canonical_hash(obj: Any) -> str:
    """Return a stable SHA-256 hex digest of a JSON-serialisable object.

    Keys are sorted and floats are serialised by :mod:`json`, so the digest is
    stable across runs and platforms for the same logical content.
    """
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: str | Path | None = None) -> str:
    """Return the current commit hash, suffixed ``-dirty`` if the tree is modified.

    Returns ``"unknown"`` when git is unavailable or this is not a repository,
    so that provenance capture never crashes a long batch run.
    """
    cwd = str(repo) if repo is not None else None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def cmtool_version() -> str:
    """Return the installed cmtool version, or ``"unknown"``."""
    try:
        from importlib.metadata import version

        return version("cmtool")
    except Exception:
        return "unknown"


@dataclass
class Provenance(ProvenanceSink):
    """Everything needed to reproduce a result.

    Attributes
    ----------
    config_hash
        Digest of the resolved configuration dictionary.
    code_commit
        Git commit of the working tree that produced the result.
    seed
        Random seed, or ``None`` for deterministic computations.
    placeholders_used
        Names of quantities whose value was a placeholder rather than a
        measurement. **Non-empty means the result is not a physical prediction.**
    """

    config_hash: str | None = None
    code_commit: str = field(default_factory=git_commit)
    version: str = field(default_factory=cmtool_version)
    seed: int | None = None
    created_utc: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    placeholders_used: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    def record_placeholder(self, name: str) -> None:
        """Record that quantity ``name`` was used provisionally."""
        if name not in self.placeholders_used:
            self.placeholders_used.append(name)

    @property
    def is_physical(self) -> bool:
        """Whether every physical input was real (no placeholders were used)."""
        return not self.placeholders_used

    def merge(self, other: Provenance) -> None:
        """Absorb placeholder usage from another provenance record."""
        for name in other.placeholders_used:
            self.record_placeholder(name)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "version": self.version,
            "seed": self.seed,
            "created_utc": self.created_utc,
            "placeholders_used": list(self.placeholders_used),
            "is_physical": self.is_physical,
            "notes": dict(self.notes),
        }

    def caveat(self) -> str | None:
        """Return a one-line caveat for plots and reports, or ``None`` if clean."""
        if self.is_physical:
            return None
        names = ", ".join(self.placeholders_used)
        return f"NOT A PHYSICAL PREDICTION - placeholder inputs used: {names}"
