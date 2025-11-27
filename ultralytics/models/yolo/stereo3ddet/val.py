# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.engine.validator import BaseValidator


class Stereo3DDetValidator(BaseValidator):
    """Minimal validator placeholder for stereo3ddet to keep training loop happy.

    This skips detection-specific evaluation that expects detection dataset internals.
    """

    class _DummyMetrics:
        keys: list[str] = []

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "stereo3ddet"
        self.metrics = self._DummyMetrics()

    def __call__(self, trainer=None, model=None):
        # No validation metrics for now; return empty dict so fitness uses loss
        return {}
