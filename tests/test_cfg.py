# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest

from ultralytics.cfg import check_cfg


def test_check_cfg_str_keys_reject_non_str():
    """A string-only arg like `split` passed a non-str (e.g. the fuzzed `split={}`) must raise a clean TypeError.

    This guards issue ultralytics/ultralytics#25056: an invalid CLI value such as `split={}` previously crashed deep
    inside the validator with a raw `TypeError`, instead of failing fast at the cfg layer.
    """
    with pytest.raises(TypeError, match="'split='"):
        check_cfg({"split": {}}, hard=True)
    with pytest.raises(TypeError, match="'split='"):
        check_cfg({"split": ["val"]}, hard=True)
    # None is allowed (optional arg) and a valid str passes through untouched.
    cfg = {"split": None}
    check_cfg(cfg, hard=True)
    assert cfg["split"] is None
    check_cfg({"split": "val"}, hard=True)


def test_check_cfg_str_keys_soft_convert():
    """With hard=False a non-str `split` is coerced to str rather than raising."""
    cfg = {"split": 123}
    check_cfg(cfg, hard=False)
    assert cfg["split"] == "123"
