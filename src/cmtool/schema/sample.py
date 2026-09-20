"""The dataset sample schema.

These pydantic models **are** the schema: the JSON Schema files under
``src/cmtool/schema/json/`` are generated from them by
:mod:`cmtool.schema.export` and checked in CI, so the documented schema can
never drift from the code that writes the data.

Refinements to the starting point in the project brief, all deliberate:

* ``input_range_deg`` is stated to be the **absolute** orientation of the input
  link, matching the world ``xy_mm`` joint coordinates.
* ``peak_strain`` is per flexure, not a single number, and carries the name of
  the strain model used -- the model differs by flexure type, and a notch hinge
  cannot use the leaf-flexure formula (``docs/physics.md``).
* ``flexures`` record their **placement mode**. Pivot-matched placement puts the
  PRBM characteristic pivot on the original rigid joint; unmatched placement does
  not, and the resulting offset is a conversion artefact rather than a
  simulation-to-reality gap. Recording the mode keeps the two separable.
* A ``feasibility`` block records why a sample was kept or rejected.
* ``provenance`` carries ``placeholders_used``: a non-empty list means the
  sample's physical numbers came from placeholders, not measurements.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "0.1"

Mm = Annotated[float, Field(description="Length in millimetres")]
Deg = Annotated[float, Field(description="Angle in degrees")]


class Base(BaseModel):
    """Common model configuration: reject unknown keys, keep data honest."""

    model_config = ConfigDict(extra="forbid")


# ------------------------------------------------------------------- rigid


class JointSpec(Base):
    """One joint of the rigid linkage."""

    bodies: tuple[str, str] = Field(description="The two bodies this joint connects")
    xy_mm: tuple[Mm, Mm] = Field(description="Position in the reference configuration")
    kind: Literal["revolute", "prismatic", "flexure"] = "revolute"


class BodySpec(Base):
    """One body of the rigid linkage."""

    name: str
    is_ground: bool = False


class OutputSpec(Base):
    """A tracked point rigidly attached to a body."""

    body: str
    xy_mm: tuple[Mm, Mm]


class GrashofSpec(Base):
    """Grashof classification of the rigid linkage."""

    condition: Literal["grashof", "non_grashof", "change_point"]
    classification: str
    shortest: str
    s_plus_l: float
    p_plus_q: float
    input_fully_rotates: bool = Field(
        description=(
            "Whether the input link can fully rotate. A flexure cannot, so such a "
            "design has no direct compliant equivalent and must be driven over a "
            "limited arc."
        )
    )


class RigidSpec(Base):
    """The rigid linkage: graph, geometry and classification."""

    type: str = Field(description="Topology name, e.g. 'four_bar'")
    bodies: list[BodySpec]
    joints: dict[str, JointSpec]
    outputs: dict[str, OutputSpec]
    input_joint: str
    input_range_deg: tuple[Deg, Deg] = Field(
        description=(
            "Absolute orientation of the input link at the start and end of the arc, "
            "measured from the world +x axis. Always a limited arc."
        )
    )
    link_lengths_mm: dict[str, Mm] = Field(default_factory=dict)
    grashof: GrashofSpec | None = None
    transmission_angle_min_deg: Deg | None = None
    transmission_angle_max_deg: Deg | None = None


# --------------------------------------------------------------- compliant


class FlexureSpec(Base):
    """A flexure replacing one revolute joint."""

    type: str = Field(description="Registered flexure type, e.g. 'small_length_pivot'")
    t_mm: Mm = Field(description="Flexure thickness (in-plane, the bending dimension)")
    length_mm: Mm = Field(description="Flexure length along its neutral axis")
    width_mm: Mm = Field(description="Out-of-plane width (the printed depth)")
    placement: Literal["pivot_matched", "unmatched"] = Field(
        default="pivot_matched",
        description=(
            "'pivot_matched' shifts geometry so the PRBM characteristic pivot lands on "
            "the original rigid joint. 'unmatched' does not, which introduces a "
            "systematic path offset that is a conversion artefact, not a physical gap."
        ),
    )
    characteristic_pivot_mm: tuple[Mm, Mm] | None = Field(
        default=None, description="World position of the PRBM characteristic pivot"
    )
    pivot_offset_mm: Mm | None = Field(
        default=None,
        description="Distance from the characteristic pivot to the original rigid joint",
    )
    stiffness_nmm_per_rad: float | None = Field(
        default=None, description="PRBM torsional stiffness K, in N*mm/rad"
    )


class CadSpec(Base):
    """Exported CAD artefacts, as paths relative to the sample folder."""

    step: str | None = None
    stl: str | None = None
    part_thickness_mm: Mm | None = None
    printer_config: str | None = None


class CompliantSpec(Base):
    """The compliant counterpart of the rigid linkage."""

    flexures: dict[str, FlexureSpec]
    material: str = Field(description="Key into configs/materials/")
    strategy: str = Field(default="naive", description="Registered design strategy used")
    cad: CadSpec | None = None


# -------------------------------------------------------------- simulation


class StrainSpec(Base):
    """Peak bending strain in one flexure over the input arc."""

    peak_strain: float
    model: str = Field(
        description=(
            "Name of the strain model used, e.g. 'leaf_uniform_bending' "
            "(eps = t*theta/(2L)). Recorded because it differs by flexure type."
        )
    )
    bend_angle_deg: Deg
    includes_axial: bool = Field(
        default=False,
        description="Whether axial stress from link loads is included in this figure",
    )
    includes_stress_concentration: bool = Field(
        default=False,
        description=(
            "Whether a geometric stress-concentration factor is included. Required for "
            "notch hinges; not applicable to a prismatic leaf flexure."
        ),
    )


class SimulatedSpec(Base):
    """Simulated results, as paths to CSV files relative to the sample folder."""

    rigid_path: str | None = None
    prbm_path: str | None = None
    fea_beam_path: str | None = None
    fea_solid_path: str | None = None
    input_torque_curve: str | None = None
    strain: dict[str, StrainSpec] = Field(default_factory=dict)
    peak_strain_overall: float | None = None
    strain_margin: float | None = Field(
        default=None,
        description="allowable_strain / peak_strain_overall; < 1 means infeasible",
    )


# -------------------------------------------------------------- measurement


class PrintMetadata(Base):
    """Everything needed to reproduce a printed specimen."""

    printer: str | None = Field(
        default=None, description="Config name of the machine the part was printed on"
    )
    print_purpose: Literal["trial", "data"] | None = Field(
        default=None,
        description=(
            "'trial' while settings were still being adjusted, 'data' for a part printed "
            "under a frozen recipe. Only 'data' prints belong in the dataset."
        ),
    )
    material: str | None = None
    brand: str | None = None
    lot_id: str | None = None
    nozzle_mm: float | None = None
    layer_height_mm: float | None = None
    orientation: str | None = None
    infill_percent: float | None = None
    nozzle_temp_c: float | None = None
    bed_temp_c: float | None = None
    print_date: str | None = None
    test_date: str | None = None
    room_temp_c: float | None = None
    room_humidity_percent: float | None = None
    cycle_count: int | None = None
    measured_flexure_t_mm: dict[str, float] = Field(
        default_factory=dict,
        description="As-printed flexure thickness per joint, for the geometry-vs-material split",
    )


class MeasuredSpec(Base):
    """Measured results for one printed specimen."""

    specimen_id: str
    path_csv: str
    metadata: PrintMetadata = Field(default_factory=PrintMetadata)
    tracking_uncertainty_mm: float | None = Field(
        default=None,
        description="Measurement uncertainty from the A5 known-motion ground truth",
    )


# --------------------------------------------------- feasibility & metrics


class FeasibilitySpec(Base):
    """Why a candidate sample was kept or rejected."""

    assembles: bool
    branch_flip: bool
    transmission_angle_ok: bool | None = None
    strain_ok: bool | None = None
    printable: bool | None = None
    rejected_reasons: list[str] = Field(default_factory=list)

    @property
    def is_feasible(self) -> bool:
        """Whether no rejection reason was recorded."""
        return not self.rejected_reasons


class MetricsSpec(Base):
    """Comparison metrics between solvers and measurement."""

    model_config = ConfigDict(extra="allow")

    prbm_vs_fea_mean_mm: float | None = None
    prbm_vs_fea_max_mm: float | None = None
    prbm_vs_fea_frechet_mm: float | None = None
    rigid_vs_fea_mean_mm: float | None = None
    sim_vs_measured_mean_mm: float | None = None
    sim_vs_measured_max_mm: float | None = None
    sim_vs_measured_frechet_mm: float | None = None


class ProvenanceSpec(Base):
    """Reproducibility record for the sample."""

    config_hash: str | None = None
    code_commit: str
    version: str
    seed: int | None = None
    created_utc: str
    placeholders_used: list[str] = Field(default_factory=list)
    is_physical: bool = Field(
        description="False means placeholder inputs were used; not a physical prediction"
    )
    notes: dict[str, Any] = Field(default_factory=dict)


class Sample(Base):
    """One dataset sample: a rigid linkage and its compliant counterpart."""

    id: str = Field(description="Stable sample id, e.g. 'fb4_000123'")
    version: str = SCHEMA_VERSION
    rigid: RigidSpec
    compliant: CompliantSpec | None = None
    joint_excursion_deg: dict[str, Deg] = Field(
        default_factory=dict,
        description="Peak-to-peak relative rotation at each joint over the input arc",
    )
    simulated: SimulatedSpec = Field(default_factory=SimulatedSpec)
    measured: list[MeasuredSpec] = Field(
        default_factory=list, description="One entry per printed specimen; 3 per design in Phase B"
    )
    feasibility: FeasibilitySpec | None = None
    metrics: MetricsSpec = Field(default_factory=MetricsSpec)
    provenance: ProvenanceSpec


#: Models exported as standalone JSON Schema files.
EXPORTED_MODELS: dict[str, type[BaseModel]] = {
    "sample": Sample,
    "linkage": RigidSpec,
}
