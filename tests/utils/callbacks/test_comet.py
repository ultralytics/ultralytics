# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ultralytics.utils.callbacks import comet as cb


@pytest.mark.parametrize("should_log_model", [True, False, None, "neither_true_or_false"])
def test_comet_respects_disable_model_logging(monkeypatch, tmp_path, should_log_model):
    """Test that Comet model logging respects the COMET_ULTRALYTICS_SHOULD_LOG_MODEL environment variable."""
    exp = MagicMock()
    trainer = SimpleNamespace(best=tmp_path / "best.pt")

    # Set environment variable BEFORE calling _log_model
    if should_log_model is None:
        # Don't set env var, should default to True
        pass
    elif should_log_model == "neither_true_or_false":
        monkeypatch.setenv("COMET_ULTRALYTICS_SHOULD_LOG_MODEL", "neither_true_or_false")
    elif should_log_model:
        monkeypatch.setenv("COMET_ULTRALYTICS_SHOULD_LOG_MODEL", "true")
    else:
        monkeypatch.setenv("COMET_ULTRALYTICS_SHOULD_LOG_MODEL", "false")

    cb._log_model(exp, trainer)

    # Assert expected behavior
    if should_log_model is False:
        exp.log_model.assert_not_called()
    else:
        # For None, True, or invalid values, should log model (default behavior)
        exp.log_model.assert_called_once()
