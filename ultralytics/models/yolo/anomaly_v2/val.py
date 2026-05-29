# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly v2 validator — runs val twice per epoch.

Phase 0 wants to measure two numbers each val epoch:

  * **B-on**:  forward with ``mask`` rendered from ``batch["bboxes"]``
              -> upper bound when prior is perfect.
  * **B-off**: forward with mask disabled (``2*sigmoid(0)=1`` passthrough)
              -> what the model does without any anomaly prior (≈ vanilla YOLO).

Both are computed in a single ``__call__`` by running the standard
``DetectionValidator`` pipeline twice with different mask state, then merging
the metric dicts. The mask-off pass's metrics are prefixed ``mask_off/``.

Mask injection point: the overridden ``preprocess`` method talks to the model
through ``set_mask_input`` / ``disable_mask_once``. This sits on the v2 model
and only affects the next forward, so it composes cleanly with the standard
DetectionValidator loop in ``ultralytics/engine/validator.py``.
"""

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER

from ._util import resolve_v2_model


class AnomalyV2Validator(DetectionValidator):
    """Detection validator that evaluates the v2 model in both mask-on and mask-off modes."""

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "anomaly_v2"
        # State carried across calls
        self._mask_mode = "on"  # "on" or "off"; controls mask injection in preprocess
        self._model_ref = None  # captured in init_metrics so preprocess can reach the model

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------
    def init_metrics(self, model) -> None:
        """Capture model reference for use in preprocess (mask injection)."""
        super().init_metrics(model)
        self._model_ref = model

    def preprocess(self, batch):
        """Set v2 mask state on the model based on current ``_mask_mode``.

        Runs BEFORE the per-batch ``model(batch["img"])`` forward, so the
        mask injection only affects that single forward.
        """
        batch = super().preprocess(batch)
        model = resolve_v2_model(self._model_ref)
        if model is None or not hasattr(model, "set_mask_input"):
            # Not a v2 model (shouldn't happen in our pipeline, but stay safe).
            return batch
        if self._mask_mode == "on":
            bb = batch.get("bboxes")
            bi = batch.get("batch_idx")
            if bb is not None and bi is not None:
                model.set_mask_input(bb, bi)
            else:
                # Defensive: no labels -> nothing to render. Disable for this forward.
                model.disable_mask_once()
        else:  # "off"
            model.disable_mask_once()
        return batch

    # ------------------------------------------------------------------
    # Two-pass __call__
    # ------------------------------------------------------------------
    def __call__(self, trainer=None, model=None):
        """Run validation twice (mask-on and mask-off), merge metric dicts.

        Pass 1 (mask-on) carries the canonical loss; the loss reported back to
        the trainer comes from this pass. Pass 2 (mask-off) re-runs everything
        but its loss is intentionally not propagated back -- the dataloader is
        consumed twice, which adds val time but no training noise.
        """
        # -------- Pass 1: mask-on --------
        self._mask_mode = "on"
        stats_on = super().__call__(trainer=trainer, model=model)

        # -------- Pass 2: mask-off --------
        self._mask_mode = "off"
        try:
            stats_off = super().__call__(trainer=trainer, model=model)
        except Exception as e:
            # If pass 2 fails we still want pass 1's results to flow to the trainer.
            LOGGER.warning(f"AnomalyV2Validator: mask-off pass failed: {e}; falling back to mask-on only.")
            stats_off = {}

        # Trainer expects a dict-of-floats. Reset mode so a subsequent call starts clean.
        self._mask_mode = "on"

        if not isinstance(stats_on, dict):
            return stats_on  # e.g. RANK > 0 returns may be different; pass through.
        merged = dict(stats_on)
        if isinstance(stats_off, dict):
            for k, v in stats_off.items():
                merged[f"mask_off/{k}"] = v
        return merged
