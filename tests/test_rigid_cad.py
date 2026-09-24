"""Tests for the pin-jointed control part.

The expensive failure this guards against is a part that prints for an hour and
does not move, so most of these are about the joint actually being a joint:
clearances that exist, caps that retain without fusing, and the coupler never
swinging over a pin cap.
"""

from __future__ import annotations

import numpy as np
import pytest

from cmtool import Linkage
from cmtool.cad.export import bounding_box_mm
from cmtool.cad.rigid import (
    DEFAULT_CLEARANCE_MM,
    RigidCadSpec,
    build_rigid_mechanism,
    estimate_print,
)

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")


@pytest.fixture(scope="module")
def linkage():
    return Linkage.four_bar(
        ground_mm=40.0,
        input_mm=32.0,
        coupler_mm=44.0,
        output_mm=36.0,
        coupler_point_mm=(22.0, 38.0),
        input_angle_deg=90.0,
        input_range_deg=(79.013, 100.987),
        name="demo_pair",
    )


@pytest.fixture(scope="module")
def built(linkage):
    return build_rigid_mechanism(linkage)


class TestPrintInPlaceGeometry:
    def test_is_a_single_printed_body_needing_no_fasteners(self, built):
        _, layout = built
        assert layout.n_printed_bodies == 1
        assert layout.n_fasteners == 0
        assert layout.n_moving_joints == 4

    def test_levels_are_stacked_with_a_running_clearance(self, built):
        """Each level clears the one below, or the mechanism prints as a brick."""
        _, layout = built
        z = layout.level_z_mm
        assert z["lower_links"] > z["base_top"]
        assert z["coupler"] > z["lower_links"]
        assert z["coupler_top"] > z["coupler"]

    def test_every_joint_gets_a_pin(self, built, linkage):
        _, layout = built
        assert set(layout.pin_centres_mm) == set(linkage.joints)
        for name, centre in layout.pin_centres_mm.items():
            assert centre == pytest.approx(tuple(linkage.joints[name].position_mm), abs=1e-9)

    def test_the_pen_hole_is_at_the_coupler_point(self, built, linkage):
        """Both halves of the pair draw from the same coordinate, or the comparison is void."""
        _, layout = built
        output = next(iter(linkage.outputs.values()))
        assert layout.pen_hole_mm == pytest.approx(tuple(output.position_mm), abs=1e-9)

    def test_the_part_fits_the_demo_envelope(self, built):
        solid, _ = built
        extent = bounding_box_mm(solid)
        assert extent[0] <= 120.0 and extent[1] <= 120.0

    def test_a_bigger_clearance_removes_material(self, linkage):
        """The holes really are cut, and they really do track the clearance."""
        tight, _ = build_rigid_mechanism(linkage, spec=RigidCadSpec(clearance_mm=0.2))
        loose, _ = build_rigid_mechanism(linkage, spec=RigidCadSpec(clearance_mm=0.6))
        assert loose.val().Volume() < tight.val().Volume()


class TestCollisionCheck:
    def test_reports_the_clearance_it_found(self, built):
        _, layout = built
        assert any("coupler-to-cap clearance" in note for note in layout.warnings)
        assert not any(note.startswith("COLLISION") for note in layout.warnings)

    def test_catches_a_coupler_that_sweeps_over_a_pin_cap(self, linkage):
        """A part that prints as one solid is the failure worth an hour of print time.

        Wide links push the cap and the link half-width out until the coupler
        cannot clear the ground pin, which is exactly the geometry that fuses.
        """
        spec = RigidCadSpec(link_width_mm=64.0, pin_diameter_mm=5.0, pen_hole_diameter_mm=5.0)
        _, layout = build_rigid_mechanism(linkage, spec=spec)
        assert any(note.startswith("COLLISION") for note in layout.warnings)

    def test_says_so_when_there_is_no_arc_to_check(self, linkage):
        bare = Linkage.from_dict(linkage.to_dict())
        bare.input_range_deg = None
        _, layout = build_rigid_mechanism(bare)
        assert any("not checked" in note for note in layout.warnings)


class TestBoltVariant:
    def test_prints_as_separate_bodies_and_needs_hardware(self, linkage):
        _, layout = build_rigid_mechanism(linkage, spec=RigidCadSpec(joint_style="bolt"))
        assert layout.n_printed_bodies == 4  # base plate plus three moving links
        assert layout.n_fasteners == 4
        assert any("M3 bolts" in note for note in layout.warnings)

    def test_still_carries_a_pen_hole(self, linkage):
        _, layout = build_rigid_mechanism(linkage, spec=RigidCadSpec(joint_style="bolt"))
        assert layout.pen_hole_mm is not None


class TestSpecValidation:
    def test_rejects_an_unknown_joint_style(self, linkage):
        with pytest.raises(ValueError, match="unknown joint_style"):
            build_rigid_mechanism(linkage, spec=RigidCadSpec(joint_style="magnetic"))

    def test_rejects_a_zero_clearance(self, linkage):
        """A zero gap is not a tight joint, it is a solid block."""
        with pytest.raises(ValueError, match="fuses the joint"):
            build_rigid_mechanism(linkage, spec=RigidCadSpec(clearance_mm=0.0))

    def test_rejects_a_pen_hole_wider_than_the_link(self, linkage):
        with pytest.raises(ValueError, match="does not fit"):
            build_rigid_mechanism(
                linkage, spec=RigidCadSpec(link_width_mm=10.0, pen_hole_diameter_mm=12.0)
            )

    def test_rejects_a_pin_that_leaves_no_wall(self, linkage):
        with pytest.raises(ValueError, match="leaves under"):
            build_rigid_mechanism(
                linkage, spec=RigidCadSpec(link_width_mm=10.0, pin_diameter_mm=9.0)
            )

    def test_the_default_clearance_is_the_documented_one(self):
        assert RigidCadSpec().clearance_mm == DEFAULT_CLEARANCE_MM == 0.35


class TestEstimate:
    def test_mass_follows_volume_and_density(self, built):
        """An estimate, but an arithmetically correct one."""
        solid, _ = built
        estimate = estimate_print(solid, density_kg_per_m3=1240.0)
        assert estimate["mass_g"] == pytest.approx(estimate["volume_mm3"] * 1240.0 * 1e-6)
        assert estimate["estimated_minutes"] > 0.0

    def test_denser_material_weighs_more_for_the_same_part(self, built):
        solid, _ = built
        light = estimate_print(solid, density_kg_per_m3=1000.0)
        heavy = estimate_print(solid, density_kg_per_m3=1400.0)
        assert heavy["mass_g"] > light["mass_g"]
        assert heavy["estimated_minutes"] == pytest.approx(light["estimated_minutes"])


class TestPairsWithTheCompliantPart:
    def test_the_two_halves_share_their_mounting_holes(self, linkage):
        """The demo only works if one fixture position serves both parts."""
        from cmtool.api import convert
        from cmtool.cad.mechanism import MechanismCadSpec, build_mechanism

        base = {"base_depth_mm": 22.0, "base_margin_mm": 10.0, "bolt_inset_mm": 7.0}
        _, rigid_layout = build_rigid_mechanism(linkage, spec=RigidCadSpec(**base))

        mechanism = convert(linkage, input_range_deg=linkage.input_range_deg, thickness_mm=0.6)
        _, compliant_layout = build_mechanism(
            mechanism, spec=MechanismCadSpec(link_width_mm=10.0, **base)
        )

        rigid_holes = sorted(tuple(np.round(p, 6)) for p in rigid_layout.bolt_holes_mm)
        compliant_holes = sorted(tuple(np.round(p, 6)) for p in compliant_layout.bolt_holes_mm)
        assert rigid_holes == compliant_holes

    def test_both_halves_draw_from_the_same_point(self, linkage):
        from cmtool.api import convert
        from cmtool.cad.mechanism import MechanismCadSpec, build_mechanism

        _, rigid_layout = build_rigid_mechanism(linkage)
        mechanism = convert(linkage, input_range_deg=linkage.input_range_deg, thickness_mm=0.6)
        _, compliant_layout = build_mechanism(
            mechanism, spec=MechanismCadSpec(link_width_mm=10.0, pen_hole_diameter_mm=5.0)
        )
        assert rigid_layout.pen_hole_mm == pytest.approx(compliant_layout.coupler_pad_mm)

    def test_the_compliant_pen_hole_is_actually_cut(self, linkage):
        from cmtool.api import convert
        from cmtool.cad.mechanism import MechanismCadSpec, build_mechanism

        mechanism = convert(linkage, input_range_deg=linkage.input_range_deg, thickness_mm=0.6)
        solid_no_hole, _ = build_mechanism(
            mechanism, spec=MechanismCadSpec(link_width_mm=10.0, pen_hole_diameter_mm=0.0)
        )
        solid_hole, _ = build_mechanism(
            mechanism, spec=MechanismCadSpec(link_width_mm=10.0, pen_hole_diameter_mm=5.0)
        )
        assert solid_hole.val().Volume() < solid_no_hole.val().Volume()
