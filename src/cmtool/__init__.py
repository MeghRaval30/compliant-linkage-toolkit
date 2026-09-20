"""cmtool: rigid-to-compliant planar linkage toolkit.

Public API::

    from cmtool import Linkage, simulate

    mech = Linkage.from_json("examples/fourbar.json")
    res = simulate(mech, solver="rigid", input_range_deg=(30, 80))
    cm = convert(mech, material="PLA", printer="bambu_a1")

See ``docs/`` for the physics conventions and ``docs/extending.md`` for adding
new linkage types, flexures, solvers or design strategies.
"""

from cmtool.api import (
    available_flexures,
    available_solvers,
    available_strategies,
    convert,
    simulate,
)
from cmtool.convert import CompliantMechanism, fit_input_arc
from cmtool.core.graph import Body, Joint, Linkage, LinkageError, OutputPoint
from cmtool.core.provenance import Provenance
from cmtool.flexures import FlexureGeometry
from cmtool.materials import Material, Printer
from cmtool.solvers import SOLVERS, MechanismState, SimulationResult

try:  # pragma: no cover - trivial
    from importlib.metadata import version as _version

    __version__ = _version("cmtool")
except Exception:  # pragma: no cover
    __version__ = "0.0.0+unknown"

__all__ = [
    "SOLVERS",
    "Body",
    "CompliantMechanism",
    "FlexureGeometry",
    "Joint",
    "Linkage",
    "LinkageError",
    "Material",
    "MechanismState",
    "OutputPoint",
    "Printer",
    "Provenance",
    "SimulationResult",
    "__version__",
    "available_flexures",
    "available_solvers",
    "available_strategies",
    "convert",
    "fit_input_arc",
    "simulate",
]
