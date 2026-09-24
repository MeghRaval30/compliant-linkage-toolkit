"""Tests for path comparison: reading measured paths, and measuring the gap.

The distance between two coupler paths is the project's headline number, so the
checks here are against geometry that can be worked out by hand -- two parallel
lines are exactly their separation apart, a curve is zero from itself -- rather
than against anything this code produced earlier.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cmtool.metrics.paths import (
    compare_paths,
    discrete_frechet,
    read_path_csv,
    resample_to_angles,
)


def line(start, end, n):
    """Return ``n`` points evenly spaced along a segment."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(start) + t * (np.asarray(end) - np.asarray(start))


class TestDiscreteFrechet:
    @pytest.mark.validation
    def test_curve_against_itself_is_zero(self):
        curve = np.column_stack([np.linspace(0, 10, 40), np.sin(np.linspace(0, 3, 40))])
        assert discrete_frechet(curve, curve) == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.validation
    @pytest.mark.parametrize("offset", [0.1, 1.0, 7.5])
    def test_parallel_lines_are_exactly_their_separation_apart(self, offset):
        """Two parallel segments: the leash never needs to be longer than the gap."""
        first = line((0.0, 0.0), (10.0, 0.0), 25)
        second = first + np.array([0.0, offset])
        assert discrete_frechet(first, second) == pytest.approx(offset, rel=1e-12)

    @pytest.mark.validation
    @pytest.mark.parametrize("n_coarse", [6, 11, 41, 201])
    def test_unequal_sampling_inflates_it_by_at_most_the_vertex_spacing(self, n_coarse):
        """The discrete measure stops only at vertices, so coarse sampling inflates it.

        The true (continuous) distance between these parallel lines is 2 mm.
        Bounding the discrete value between that and the continuous value plus
        the coarser curve's spacing pins down exactly how much sampling can cost
        -- which is why a measured take is resampled onto matched input angles
        before it is compared pointwise.
        """
        spacing = 10.0 / (n_coarse - 1)
        coarse = line((0.0, 0.0), (10.0, 0.0), n_coarse)
        dense = line((0.0, 2.0), (10.0, 2.0), 401)
        value = discrete_frechet(coarse, dense)
        assert value >= 2.0 - 1e-12
        assert value <= 2.0 + spacing + 1e-9

    @pytest.mark.validation
    def test_matched_sampling_is_exactly_the_continuous_distance(self):
        """Walking two equally sampled curves in step costs nothing extra."""
        first = line((0.0, 0.0), (10.0, 0.0), 41)
        second = line((0.0, 2.0), (10.0, 2.0), 41)
        assert discrete_frechet(first, second) == pytest.approx(2.0, rel=1e-12)

    @pytest.mark.validation
    def test_is_symmetric(self):
        first = line((0.0, 0.0), (5.0, 5.0), 17)
        second = line((1.0, 0.0), (6.0, 6.0), 23)
        assert discrete_frechet(first, second) == pytest.approx(
            discrete_frechet(second, first), rel=1e-12
        )

    @pytest.mark.validation
    def test_orientation_matters(self):
        """Both walkers move forward only, so a reversed curve is not the same curve.

        A nearest-neighbour distance would call these identical, which is why it
        is not what this project quotes.
        """
        curve = line((0.0, 0.0), (10.0, 0.0), 21)
        assert discrete_frechet(curve, curve[::-1]) == pytest.approx(10.0, rel=1e-12)

    @settings(max_examples=25, deadline=None)
    @given(
        shift=st.floats(min_value=-5.0, max_value=5.0),
        n=st.integers(min_value=2, max_value=40),
    )
    def test_never_exceeds_the_matched_pointwise_maximum(self, shift, n):
        """Walking both curves in step is one admissible coupling, so it bounds it."""
        first = line((0.0, 0.0), (10.0, 3.0), n)
        second = first + np.array([shift, 0.0])
        pointwise = float(np.max(np.linalg.norm(first - second, axis=1)))
        assert discrete_frechet(first, second) <= pointwise + 1e-9

    def test_rejects_an_empty_path(self):
        with pytest.raises(ValueError, match="empty path"):
            discrete_frechet(np.empty((0, 2)), line((0, 0), (1, 1), 3))


class TestComparePaths:
    @pytest.mark.validation
    def test_a_constant_offset_gives_that_offset_everywhere(self):
        first = line((0.0, 0.0), (20.0, 0.0), 31)
        second = first + np.array([0.0, 0.4])
        result = compare_paths(first, second, labels=("a", "b"))
        assert result.mean_mm == pytest.approx(0.4, rel=1e-12)
        assert result.max_mm == pytest.approx(0.4, rel=1e-12)
        assert result.rms_mm == pytest.approx(0.4, rel=1e-12)
        assert result.frechet_mm == pytest.approx(0.4, rel=1e-12)
        assert result.n_points == 31

    def test_rms_is_never_below_the_mean(self):
        first = line((0.0, 0.0), (10.0, 0.0), 40)
        second = first.copy()
        second[:, 1] += np.linspace(0.0, 2.0, 40)
        result = compare_paths(first, second)
        assert result.rms_mm >= result.mean_mm
        assert result.max_mm >= result.rms_mm

    def test_refuses_to_compare_mismatched_samples(self):
        """A pointwise comparison of differently sampled curves invents a discrepancy."""
        with pytest.raises(ValueError, match="matched samples"):
            compare_paths(line((0, 0), (1, 1), 10), line((0, 0), (1, 1), 11))

    def test_to_dict_carries_the_labels(self):
        result = compare_paths(
            line((0, 0), (1, 1), 5), line((0, 1), (1, 2), 5), labels=("rigid", "fea")
        )
        assert result.to_dict()["a"] == "rigid"
        assert result.to_dict()["b"] == "fea"


class TestResampleToAngles:
    @pytest.mark.validation
    def test_reproduces_a_linear_path_exactly(self):
        angles = np.linspace(10.0, 50.0, 9)
        points = np.column_stack([angles * 2.0, angles * -3.0 + 7.0])
        target = np.array([12.5, 33.3, 47.0])
        out = resample_to_angles(angles, points, target)
        assert out[:, 0] == pytest.approx(target * 2.0)
        assert out[:, 1] == pytest.approx(target * -3.0 + 7.0)

    def test_clamps_rather_than_extrapolating(self):
        """Beyond the solved arc there is no prediction, so the ends hold."""
        angles = np.linspace(0.0, 10.0, 11)
        points = np.column_stack([angles, angles])
        out = resample_to_angles(angles, points, np.array([-5.0, 15.0]))
        assert out[0] == pytest.approx([0.0, 0.0])
        assert out[1] == pytest.approx([10.0, 10.0])

    def test_handles_a_decreasing_sweep(self):
        angles = np.linspace(50.0, 10.0, 9)
        points = np.column_stack([angles, np.zeros_like(angles)])
        out = resample_to_angles(angles, points, np.array([25.0]))
        assert out[0, 0] == pytest.approx(25.0)

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError, match="angles for"):
            resample_to_angles(np.arange(5.0), np.zeros((4, 2)), np.array([1.0]))


HEADER = "frame,input_angle_deg,coupler_x_mm,coupler_y_mm,n_markers,homography_rms_px"


class TestReadPathCsv:
    def test_reads_a_tracked_file(self, tmp_path):
        path = tmp_path / "measured.csv"
        path.write_text(
            HEADER + "\n0,48.9,71.2,90.5,4,0.21\n1,52.0,74.0,89.1,4,0.19\n",
            encoding="utf-8",
        )
        measured = read_path_csv(path)
        assert measured.n_points == 2
        assert measured.has_angles
        assert measured.points_mm[1] == pytest.approx([74.0, 89.1])
        assert measured.source.endswith("measured.csv")

    def test_a_blank_angle_is_nan_not_zero(self, tmp_path):
        """A frame the tracker could not read the lever in is a gap, never a zero."""
        path = tmp_path / "gappy.csv"
        path.write_text(HEADER + "\n0,,71.2,90.5,4,0.21\n", encoding="utf-8")
        measured = read_path_csv(path)
        assert np.isnan(measured.input_angles_deg[0])
        assert not measured.has_angles

    def test_rejects_a_file_with_no_coordinate_columns(self, tmp_path):
        path = tmp_path / "wrong.csv"
        path.write_text("a,b\n1,2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected columns"):
            read_path_csv(path)

    def test_rejects_a_file_with_no_usable_rows(self, tmp_path):
        """An empty measured file is an error, not an empty measurement."""
        path = tmp_path / "empty.csv"
        path.write_text(HEADER + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="no rows with usable coordinates"):
            read_path_csv(path)
