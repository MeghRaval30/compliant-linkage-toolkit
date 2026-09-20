"""Tests for units, the plug-in registry and the placeholder discipline."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cmtool.core.provenance import Provenance, canonical_hash
from cmtool.core.quantities import (
    MissingMeasurementError,
    ProvisionalDataWarning,
    Quantity,
)
from cmtool.core.registry import Registry, RegistryError
from cmtool.core.units import cross2, rotation_matrix, wrap_to_pi


class TestUnits:
    def test_wrap_to_pi_keeps_small_angles(self):
        for angle in (-3.0, -1.0, 0.0, 1.0, 3.0):
            assert wrap_to_pi(angle) == pytest.approx(angle)

    def test_wrap_to_pi_removes_full_turns(self):
        assert wrap_to_pi(0.5 + 2 * np.pi) == pytest.approx(0.5)
        assert wrap_to_pi(0.5 - 4 * np.pi) == pytest.approx(0.5)

    def test_wrap_to_pi_is_half_open_at_pi(self):
        assert wrap_to_pi(np.pi) == pytest.approx(np.pi)
        assert wrap_to_pi(-np.pi) == pytest.approx(np.pi)

    def test_rotation_matrix_is_orthonormal(self):
        rot = rotation_matrix(0.7)
        np.testing.assert_allclose(rot @ rot.T, np.eye(2), atol=1e-12)
        assert np.linalg.det(rot) == pytest.approx(1.0)

    def test_rotation_by_ninety_degrees(self):
        np.testing.assert_allclose(
            rotation_matrix(np.pi / 2) @ np.array([1.0, 0.0]), [0.0, 1.0], atol=1e-12
        )

    def test_cross2_sign(self):
        assert cross2(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
        assert cross2(np.array([0.0, 1.0]), np.array([1.0, 0.0])) == pytest.approx(-1.0)


class TestRegistry:
    def test_register_and_get(self):
        reg: Registry[str] = Registry("thing")
        reg.add("a", "alpha")
        assert reg.get("a") == "alpha"
        assert "a" in reg
        assert reg.names() == ["a"]

    def test_decorator_registration(self):
        reg: Registry[object] = Registry("thing")

        @reg.register("beta")
        class Beta:
            pass

        assert reg.get("beta") is Beta

    def test_duplicate_registration_is_refused(self):
        reg: Registry[str] = Registry("thing")
        reg.add("a", "alpha")
        with pytest.raises(RegistryError, match="already registered"):
            reg.add("a", "other")

    def test_unknown_name_lists_alternatives(self):
        reg: Registry[str] = Registry("flexure type")
        reg.add("slfp", "x")
        with pytest.raises(RegistryError, match="available: slfp"):
            reg.get("notch")


class TestQuantity:
    def test_measured_value_is_returned_without_warning(self):
        q = Quantity.from_config(
            "PLA.E", {"value": 3210.0, "status": "measured", "source": "our test"}
        )
        assert not q.is_placeholder
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert q.get() == pytest.approx(3210.0)

    def test_placeholder_warns_and_is_recorded(self):
        prov = Provenance()
        q = Quantity.from_config(
            "PLA.E", {"value": None, "provisional_value": 3500.0, "status": "placeholder"}
        )
        assert q.is_placeholder
        with pytest.warns(ProvisionalDataWarning, match="PLACEHOLDER"):
            assert q.get(prov) == pytest.approx(3500.0)
        assert prov.placeholders_used == ["PLA.E"]
        assert prov.is_physical is False
        assert "PLA.E" in (prov.caveat() or "")

    def test_bare_scalar_has_no_provenance_so_counts_as_placeholder(self):
        q = Quantity.from_config("PLA.E", 3500.0)
        assert q.is_placeholder

    def test_strict_mode_refuses_placeholders(self, monkeypatch):
        monkeypatch.setenv("CMTOOL_STRICT_DATA", "1")
        q = Quantity.from_config(
            "PLA.E", {"value": None, "provisional_value": 3500.0, "status": "placeholder"}
        )
        with pytest.raises(MissingMeasurementError, match="CMTOOL_STRICT_DATA"):
            q.get()

    def test_placeholder_without_any_number_always_raises(self):
        q = Quantity.from_config("PLA.E", {"value": None, "status": "placeholder"})
        with pytest.raises(MissingMeasurementError, match="nothing to compute with"):
            q.get()

    def test_real_status_without_value_is_a_config_error(self):
        q = Quantity.from_config("PLA.E", {"value": None, "status": "measured"})
        with pytest.raises(MissingMeasurementError, match="fix the config"):
            q.get()


class TestProvenance:
    def test_clean_provenance_is_physical(self):
        assert Provenance().is_physical
        assert Provenance().caveat() is None

    def test_placeholders_are_deduplicated(self):
        prov = Provenance()
        prov.record_placeholder("a")
        prov.record_placeholder("a")
        assert prov.placeholders_used == ["a"]

    def test_merge_absorbs_placeholders(self):
        first, second = Provenance(), Provenance()
        second.record_placeholder("PLA.E")
        first.merge(second)
        assert first.placeholders_used == ["PLA.E"]

    def test_canonical_hash_is_key_order_independent(self):
        assert canonical_hash({"a": 1, "b": 2}) == canonical_hash({"b": 2, "a": 1})

    def test_canonical_hash_detects_change(self):
        assert canonical_hash({"a": 1}) != canonical_hash({"a": 2})

    def test_to_dict_round_trips_through_json(self):
        import json

        payload = json.loads(json.dumps(Provenance(seed=7).to_dict()))
        assert payload["seed"] == 7
        assert payload["is_physical"] is True
