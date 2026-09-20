"""Rigid-to-compliant conversion strategies.

Importing this package registers the bundled strategies.
"""

from cmtool.convert import naive as _naive  # noqa: F401  (registers "naive")
from cmtool.convert.arc import ArcFit, ArcFitError, fit_input_arc
from cmtool.convert.base import (
    STRATEGIES,
    CompliantMechanism,
    DesignStrategy,
    FeasibilityReport,
    JointSizing,
)

__all__ = [
    "STRATEGIES",
    "ArcFit",
    "ArcFitError",
    "CompliantMechanism",
    "DesignStrategy",
    "FeasibilityReport",
    "JointSizing",
    "fit_input_arc",
]
