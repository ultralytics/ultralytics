"""Unit tests for the model registry. Does NOT load weights — those are exercised
in Stage 1's sanity gate on westd.
"""
import pytest

import models as M


def test_registered_tags_match_spec():
    expected = {"champion", "mgn-t3", "mgn-t4", "t5fix", "solider"}
    assert expected.issubset(set(M.MODEL_REGISTRY.keys()))


def test_each_entry_has_required_fields():
    for tag, entry in M.MODEL_REGISTRY.items():
        assert "ckpt_env_var" in entry, f"{tag} missing ckpt_env_var"
        assert "kind" in entry, f"{tag} missing kind"
        assert "tap_p4" in entry, f"{tag} missing tap_p4"
        assert "tap_p5" in entry, f"{tag} missing tap_p5"
        assert "imgsz" in entry, f"{tag} missing imgsz"


def test_solider_kind_is_swin():
    assert M.MODEL_REGISTRY["solider"]["kind"] == "swin"


def test_champion_kind_is_yolo_reid():
    assert M.MODEL_REGISTRY["champion"]["kind"] == "yolo_reid"


def test_unknown_tag_raises():
    with pytest.raises(KeyError):
        M.get_model_entry("does-not-exist")
