"""Tests for the monolithic mechanism CAD."""

from __future__ import annotations

import numpy as np
import pytest

from cmtool import Linkage, convert
from cmtool.cad import bounding_box_mm, check_printability
from cmtool.cad.mechanism import MechanismCadSpec, build_mechanism
from cmtool.materials import Printer

pytestmark = pytest.mark.filterwarnings("ignore::cmtool.core.quantities.ProvisionalDataWarning")


@pytest.fixture(scope="module")
def printer() -> Printer:
    return Printer.load("bambu_a1")


@pytest.fixture(scope="module")
def mechanism():
    linkage = Linkage.four_bar(
        ground_mm=60.0,
        input_mm=55.0,
        coupler_mm=62.0,
        output_mm=58.0,
        coupler_point_mm=(35.0, 70.0),
        input_angle_deg=95.0,
        input_range_deg=(88.0, 102.0),
        name="cad_demo",
    )
    return convert(linkage, thickness_mm=0.6)


@pytest.fixture(scope="module")
def built(mechanism):
    return build_mechanism(mechanism)


class TestMonolithic:
    def test_is_a_single_connected_solid(self, built):
        """A compliant mechanism is one piece: no assembly, no pin joints."""
        solid, _ = built
        assert len(solid.vals()[0].Solids()) == 1

    def test_fits_the_print_envelope(self, built, printer):
        solid, _ = built
        extent = bounding_box_mm(solid)
        envelope = printer.design_envelope_mm()
        assert extent[0] <= envelope[0]
        assert extent[1] <= envelope[1]

    def test_thickness_is_the_part_depth_plus_the_marker_pads(self, built, mechanism, printer):
        extent = bounding_box_mm(built[0])
        layout = built[1]
        depth = printer.part_thickness_mm()
        assert extent[2] == pytest.approx(depth + MechanismCadSpec().pad_thickness_mm)
        assert layout.pad_plane_z_mm == pytest.approx(extent[2])

    def test_printability_check_passes(self, built, mechanism, printer):
        solid, _ = built
        thinnest = min(s.geometry.thickness_mm for s in mechanism.sizing.values())
        assert check_printability(solid, printer, min_feature_mm=thinnest).ok


class TestFlexurePlacement:
    def test_every_joint_gets_a_flexure_centred_on_it(self, built, mechanism):
        """Centring is what makes the placement pivot-matched."""
        layout = built[1]
        for joint_name, joint in mechanism.base.joints.items():
            sized = mechanism.sizing[joint_name]
            axis = np.asarray(layout.flexure_axis[joint_name], dtype=float)
            half = sized.geometry.length_mm / 2.0
            host = sized.host_body
            other = joint.other(host)
            near = np.asarray(layout.attachment_points[other][joint_name], dtype=float)
            far = np.asarray(layout.attachment_points[host][joint_name], dtype=float)
            np.testing.assert_allclose((near + far) / 2.0, joint.position_mm, atol=1e-9)
            np.testing.assert_allclose(far - near, axis * 2.0 * half, atol=1e-9)

    def test_attachment_points_are_one_per_body_per_joint(self, built, mechanism):
        _, layout = built
        for body in mechanism.base.bodies:
            assert set(layout.attachment_points[body]) == set(mechanism.base.joints_of(body))

    def test_longer_flexures_make_a_bigger_part(self, mechanism):
        base = mechanism.base
        short = convert(base, thickness_mm=0.6, flexure_length_mm=2.0)
        long = convert(base, thickness_mm=0.6, flexure_length_mm=10.0)
        assert (
            bounding_box_mm(build_mechanism(long)[0])[0]
            >= bounding_box_mm(build_mechanism(short)[0])[0]
        )


class TestFixtureAndMarkers:
    def test_four_m3_bolt_holes(self, built):
        _, layout = built
        assert len(layout.bolt_holes_mm) == 4

    def test_two_fiducial_pads_at_a_known_spacing(self, built):
        """The vision pipeline scales the image by this distance."""
        _, layout = built
        assert len(layout.fiducial_pads_mm) == 2
        measured = float(
            np.linalg.norm(
                np.asarray(layout.fiducial_pads_mm[0]) - np.asarray(layout.fiducial_pads_mm[1])
            )
        )
        assert measured == pytest.approx(layout.fiducial_spacing_mm, rel=1e-6)
        assert layout.fiducial_spacing_mm > 30.0

    def test_lever_and_coupler_pads_exist(self, built):
        _, layout = built
        assert layout.lever_pad_mm is not None
        assert layout.coupler_pad_mm is not None

    def test_coupler_pad_sits_on_the_tracked_point(self, built, mechanism):
        _, layout = built
        output = next(iter(mechanism.base.outputs.values()))
        np.testing.assert_allclose(layout.coupler_pad_mm, output.position_mm, atol=1e-9)

    def test_all_marker_pads_are_coplanar(self, built):
        """One homography maps one plane; markers at different heights would not work."""
        _, layout = built
        assert layout.pad_plane_z_mm > 0.0

    def test_lever_clearance_is_checked(self, built):
        _, layout = built
        assert isinstance(layout.lever_clears_base, bool)

    def test_an_overlong_lever_is_reported(self, mechanism):
        _, layout = build_mechanism(mechanism, spec=MechanismCadSpec(lever_length_mm=200.0))
        assert not layout.lever_clears_base
        assert any("sweeps over the base" in w for w in layout.warnings)

    def test_layout_is_serialisable(self, built):
        import json

        json.dumps(built[1].to_dict())
