# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import pytest

from ultralytics.cfg import check_cfg


def test_check_cfg_bool_or_str_keys_reject_containers():
    """A bool-or-str arg like `compile` passed a container (e.g. the fuzzed `compile={}`) must raise a clean TypeError.

    This guards issue ultralytics/ultralytics#25056: an invalid CLI value such as `compile={}` previously crashed deep
    inside torch's DistributedSampler with a raw error, instead of failing fast at the cfg layer.
    """
    with pytest.raises(TypeError, match=r"'compile=\{\}' is of invalid type dict"):
        check_cfg({"compile": {}}, hard=True)
    with pytest.raises(TypeError, match=r"'compile=\['inductor'\]' is of invalid type list"):
        check_cfg({"compile": ["inductor"]}, hard=True)
    # Valid bool and str forms pass through untouched; None is invalid because the default is False.
    check_cfg({"compile": True}, hard=True)
    check_cfg({"compile": "inductor"}, hard=True)
    with pytest.raises(TypeError, match="'compile=None' is invalid"):
        check_cfg({"compile": None}, hard=True)


def test_check_cfg_bool_or_str_keys_soft_convert():
    """With hard=False a non-bool/non-str `compile` is coerced rather than raising."""
    cfg = {"compile": 1}
    check_cfg(cfg, hard=False)
    assert cfg["compile"] is True  # 1 -> bool True, mirroring CFG_BOOL_KEYS behavior
    cfg = {"compile": 0}
    check_cfg(cfg, hard=False)
    assert cfg["compile"] is False
