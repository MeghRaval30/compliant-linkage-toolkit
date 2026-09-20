"""Tests for the linkage graph: topology, frames, validation, serialisation."""

from __future__ import annotations

import numpy as np
import pytest

from cmtool.core.graph import Body, Joint, Linkage, LinkageError, OutputPoint


def make_four_bar() -> Linkage:
    """Build a four-bar by hand, so the graph code is tested without the helper."""
    return Linkage(
        bodies={
            "ground": Body("ground", is_ground=True),
            "input": Body("input"),
            "coupler": Body("coupler"),
            "output": Body("output"),
        },
        joints={
            "A": Joint("A", ("ground", "input"), (0.0, 0.0)),
            "B": Joint("B", ("input", "coupler"), (0.0, 30.0)),
            "C": Joint("C", ("coupler", "output"), (40.0, 30.0)),
            "D": Joint("D", ("output", "ground"), (40.0, 0.0)),
        },
        outputs={"P": OutputPoint("P", "coupler", (20.0, 45.0))},
        input_joint="A",
        input_range_deg=(60.0, 120.0),
        name="hand_built",
    )


class TestTopology:
    def test_four_bar_has_mobility_one(self):
        assert make_four_bar().mobility() == 1

    def test_four_bar_has_one_independent_loop(self):
        loops = make_four_bar().independent_loops()
        assert len(loops) == 1
        assert set(loops[0]) == {"A", "B", "C", "D"}

    def test_open_chain_has_no_loop_and_higher_mobility(self):
        chain = Linkage(
            bodies={
                "ground": Body("ground", is_ground=True),
                "l1": Body("l1"),
                "l2": Body("l2"),
            },
            joints={
                "A": Joint("A", ("ground", "l1"), (0.0, 0.0)),
                "B": Joint("B", ("l1", "l2"), (10.0, 0.0)),
            },
            input_joint="A",
            input_range_deg=(0.0, 10.0),
            name="chain",
        )
        chain.validate()
        assert chain.independent_loops() == []
        assert chain.mobility() == 2

    def test_five_bar_topology_is_representable(self):
        """The data model must already describe a 2-DOF five-bar (future work)."""
        five = Linkage(
            bodies={
                "ground": Body("ground", is_ground=True),
                "l1": Body("l1"),
                "l2": Body("l2"),
                "l3": Body("l3"),
                "l4": Body("l4"),
            },
            joints={
                "A": Joint("A", ("ground", "l1"), (0.0, 0.0)),
                "B": Joint("B", ("l1", "l2"), (20.0, 20.0)),
                "C": Joint("C", ("l2", "l3"), (50.0, 25.0)),
                "D": Joint("D", ("l3", "l4"), (70.0, 10.0)),
                "E": Joint("E", ("l4", "ground"), (60.0, 0.0)),
            },
            input_joint="A",
            input_range_deg=(0.0, 30.0),
            name="five_bar",
        )
        five.validate()
        assert five.mobility() == 2
        assert len(five.independent_loops()) == 1

    def test_joints_of_and_neighbours(self):
        mech = make_four_bar()
        assert mech.joints_of("input") == ["A", "B"]
        assert set(mech.neighbours("input")) == {"ground", "coupler"}

    def test_input_body_is_the_moving_side_of_the_input_joint(self):
        assert make_four_bar().input_body == "input"

    def test_joint_other_rejects_unrelated_body(self):
        joint = Joint("A", ("ground", "input"), (0.0, 0.0))
        with pytest.raises(LinkageError, match="does not touch"):
            joint.other("coupler")


class TestGeometry:
    def test_link_length(self):
        mech = make_four_bar()
        assert mech.link_length("input") == pytest.approx(30.0)
        assert mech.link_length("coupler") == pytest.approx(40.0)

    def test_body_frame_origin_and_direction(self):
        mech = make_four_bar()
        origin, angle = mech.body_frame("coupler")
        np.testing.assert_allclose(origin, [0.0, 30.0])
        assert angle == pytest.approx(0.0)

    def test_ground_frame_is_the_world_frame(self):
        origin, angle = make_four_bar().body_frame("ground")
        np.testing.assert_allclose(origin, [0.0, 0.0])
        assert angle == pytest.approx(0.0)

    def test_to_local_then_to_world_is_identity_at_reference(self):
        mech = make_four_bar()
        positions = {n: j.position_mm for n, j in mech.joints.items()}
        point = np.array([20.0, 45.0])
        local = mech.to_local("coupler", point)
        np.testing.assert_allclose(mech.to_world("coupler", local, positions), point, atol=1e-12)

    def test_rigid_attachment_follows_the_body(self):
        """A point fixed to a body must move rigidly with that body's joints."""
        mech = make_four_bar()
        local = mech.to_local("coupler", np.array([20.0, 45.0]))
        # Translate and rotate the coupler by moving both of its joints.
        moved = {
            "A": np.array([0.0, 0.0]),
            "B": np.array([5.0, 35.0]),
            "C": np.array([5.0, 75.0]),  # coupler now points along +y
            "D": np.array([40.0, 0.0]),
        }
        world = mech.to_world("coupler", local, moved)
        # Distances to both coupler joints are invariant under rigid motion.
        assert np.linalg.norm(world - moved["B"]) == pytest.approx(
            np.linalg.norm(np.array([20.0, 45.0]) - np.array([0.0, 30.0]))
        )
        assert np.linalg.norm(world - moved["C"]) == pytest.approx(
            np.linalg.norm(np.array([20.0, 45.0]) - np.array([40.0, 30.0]))
        )


class TestValidation:
    def test_valid_linkage_passes(self):
        make_four_bar().validate()

    def test_unknown_body_reference_is_caught(self):
        mech = make_four_bar()
        mech.joints["B"] = Joint("B", ("input", "ghost"), (0.0, 30.0))
        with pytest.raises(LinkageError, match="unknown body 'ghost'"):
            mech.validate()

    def test_two_ground_bodies_are_caught(self):
        mech = make_four_bar()
        mech.bodies["coupler"] = Body("coupler", is_ground=True)
        with pytest.raises(LinkageError, match="exactly one ground body"):
            mech.validate()

    def test_missing_input_joint_is_caught(self):
        mech = make_four_bar()
        mech.input_joint = ""
        with pytest.raises(LinkageError, match="input_joint is not set"):
            mech.validate()

    def test_input_joint_must_touch_ground(self):
        mech = make_four_bar()
        mech.input_joint = "B"
        with pytest.raises(LinkageError, match="must connect ground"):
            mech.validate()

    def test_empty_input_range_is_caught(self):
        mech = make_four_bar()
        mech.input_range_deg = (45.0, 45.0)
        with pytest.raises(LinkageError, match="input_range_deg is empty"):
            mech.validate()

    def test_body_without_joints_is_caught(self):
        mech = make_four_bar()
        mech.bodies["loose"] = Body("loose")
        with pytest.raises(LinkageError, match="body 'loose' has no joints"):
            mech.validate()

    def test_disconnected_component_is_caught(self):
        """Two bodies joined to each other but not to the main loop."""
        mech = make_four_bar()
        mech.bodies["x"] = Body("x")
        mech.bodies["y"] = Body("y")
        mech.joints["E"] = Joint("E", ("x", "y"), (200.0, 200.0))
        with pytest.raises(LinkageError, match="not connected"):
            mech.validate()

    def test_self_loop_is_caught(self):
        mech = make_four_bar()
        mech.joints["A"] = Joint("A", ("ground", "ground"), (0.0, 0.0))
        with pytest.raises(LinkageError, match="to itself"):
            mech.validate()


class TestSerialisation:
    def test_round_trip_preserves_geometry(self):
        original = make_four_bar()
        restored = Linkage.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.input_joint == original.input_joint
        assert restored.input_range_deg == original.input_range_deg
        for name, joint in original.joints.items():
            np.testing.assert_allclose(restored.joints[name].position_mm, joint.position_mm)
        np.testing.assert_allclose(
            restored.outputs["P"].position_mm, original.outputs["P"].position_mm
        )

    def test_json_file_round_trip(self, tmp_path):
        original = make_four_bar()
        path = tmp_path / "fourbar.json"
        original.to_json(path)
        restored = Linkage.from_json(path)
        assert restored.to_dict() == original.to_dict()

    def test_from_dict_validates(self):
        data = make_four_bar().to_dict()
        data["input_joint"] = "nope"
        with pytest.raises(LinkageError):
            Linkage.from_dict(data)
