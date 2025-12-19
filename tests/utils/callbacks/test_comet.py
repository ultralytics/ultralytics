# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from ultralytics.utils.callbacks import comet as cb

@pytest.mark.parametrize("should_log_model", [True, False])
def test_comet_respects_disable_model_logging(monkeypatch, tmp_path, should_log_model):
    """Test that Comet model logging respects the COMET_ULTRALYTICS_SHOULD_LOG_MODEL environment variable."""
    exp = MagicMock()
    trainer = SimpleNamespace(best=tmp_path / "best.pt")
    monkeypatch.setenv("COMET_ULTRALYTICS_SHOULD_LOG_MODEL", "true" if should_log_model else "false")

    cb._log_model(exp, trainer)

    if should_log_model:
        exp.log_model.assert_called_once()
    else:
        exp.log_model.assert_not_called()