# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLOA — training-free anomaly detection (v2) wrapper.

Fit a memory bank on normal images ("fit" = build the bank, no gradient training), then predict /
val with the heatmap prior. All bank-build and heatmap-processing knobs are baked into the model
and are not configurable per call.

Examples:
    >>> m = YOLOA("best.pt")
    >>> m.fit("bottle/train/good", name="bottle")
    >>> m.predict("test.png")
    >>> m.val(data="bottle.yaml")
    >>> m.save("bottle_fitted.pt")  # carries bank + fit_data
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import LOGGER

# Extra kwargs accepted by predict()/val(). Only ``hm_gate_blend`` remains as a per-call head override.
_INFER_KEYS = {"hm_gate_blend"}


class YOLOA(Model):
    """Training-free anomaly v2: fit a memory bank on normals, then predict / val with heatmap prior."""

    def __init__(self, model: str | Path = "yolo26m-anomaly-v2.yaml", verbose: bool = False) -> None:
        """Load a YOLOA model. A v2 ckpt routes by its baked ``task``; a yaml is forced to anomaly_v2."""
        super().__init__(model=model, task="anomaly_v2", verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map the anomaly_v2 task to its model, trainer, validator and predictor."""
        return {
            "anomaly_v2": {
                "model": YOLOAnomalyV2Model,
                "trainer": yolo.anomaly_v2.AnomalyV2Trainer,
                "validator": yolo.anomaly_v2.YOLOAnomalyValidator,
                "predictor": yolo.anomaly_v2.YOLOAnomalyPredictor,
            }
        }

    def fit(
        self,
        source: str | Path | list[str],
        name: str | None = None,
        cache: str | Path | None = None,
        refit: bool = False,
        device: Any = None,
        batch: int = 8,
    ) -> "YOLOA":
        """Build (or load from cache) the memory bank from normal images.

        Args:
            source: Directory of normal images, or a list of image paths.
            name: Friendly label for provenance + cache key (default: derived from ``source``).
            cache: Directory to cache/reuse the built bank on disk; None disables caching.
            refit: Force a rebuild even if a cache file exists.
            device: Build device (defaults to the model device); not part of the fit identity.
            batch: Mini-batch size for feature extraction; not part of the fit identity.

        Returns:
            (YOLOA): self, now fitted.
        """
        self._check_is_pytorch_model()
        m = self.model
        mb = getattr(m, "memory_bank", None)
        if mb is None or getattr(m, "bb_layers", None) is None:
            raise RuntimeError("model has no BackboneMemoryBank (no bb_layers in the model yaml) — cannot fit().")

        data_name = name or self._derive_name(source)
        device = device if device is not None else next(m.parameters()).device
        cache_path = (Path(cache) / f"{data_name}.pt") if cache is not None else None
        if cache_path is not None and cache_path.exists() and not refit:
            d = torch.load(cache_path, map_location="cpu")
            if not d.get("_calibrated"):
                LOGGER.warning(f"bank cache is old format (no calibration state); delete {cache_path} to rebuild")
            mb.load_bank(d["memory_bank"])
            mb.temperature = d["temperature"]
            mb.update = False
            if d.get("_calibrated"):
                mb._threshold = d["_threshold"]
                mb._compactness = d["_compactness"]
                mb._calibrated = True
            LOGGER.info(f"YOLOA.fit: loaded cached bank ({mb.memory_bank.shape[0]} vecs) <- {cache_path}")
        else:
            n = m.build_memory_bank(
                source,
                imgsz=640,
                device=device,
                batch=batch,
                max_bank_size=10000,
                max_images=0,
                verbose=True,
            )
            if cache_path is not None and n:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                entry = {
                    "memory_bank": mb.memory_bank.detach().cpu(),
                    "feature_dim": mb.feature_dim,
                    "temperature": float(mb.temperature),
                }
                if getattr(mb, "_calibrated", False):
                    entry["_threshold"] = mb._threshold
                    entry["_compactness"] = mb._compactness
                    entry["_calibrated"] = True
                torch.save(entry, cache_path)
                LOGGER.info(f"YOLOA.fit: cached bank ({mb.memory_bank.shape[0]} vecs) -> {cache_path}")

        m.fit_data = data_name
        m._heatmap_bank_warned = False
        return self

    def predict(self, source=None, stream: bool = False, **kwargs: Any):
        """Predict with the heatmap prior when a fitted bank is available.

        Args:
            source: Image source (path / array / list), as in :meth:`Model.predict`.
            stream: Stream the source.
            **kwargs: Standard predict args plus ``hm_gate_blend``.

        Returns:
            (list[Results]): Prediction results.
        """
        self._apply_infer_overrides(kwargs)
        return super().predict(source=source, stream=stream, **kwargs)

    def val(self, validator=None, **kwargs: Any):
        """Validate with the heatmap prior when a fitted bank is available.

        Args:
            validator: Optional custom validator.
            **kwargs: Standard val args plus ``end2end`` and ``hm_gate_blend``.
                ``end2end`` is applied to the detection head for this call only.

        Returns:
            Validation metrics.
        """
        # Optional head mutations for this val call only.
        e2e = kwargs.pop("end2end", None)
        head = self.model.model[-1]
        old_e2e = None
        if e2e is not None:
            old_e2e = getattr(head, "end2end", None)
            head.end2end = e2e
            self.model.end2end = e2e

        self._apply_infer_overrides(kwargs)
        try:
            return super().val(validator=validator, **kwargs)
        finally:
            if e2e is not None:
                if old_e2e is None:
                    head.__dict__.pop("end2end", None)
                else:
                    head.end2end = old_e2e

    # ---- internals ----------------------------------------------------------------------------

    def _apply_infer_overrides(self, kwargs: dict) -> None:
        """Pop the optional ``hm_gate_blend`` head override."""
        if "hm_gate_blend" in kwargs:
            self.model.model[-1].hm_gate_blend = float(kwargs.pop("hm_gate_blend"))

    @staticmethod
    def _derive_name(data) -> str:
        """Derive a short label from a data path (".../bottle/train/good" -> "bottle")."""
        if isinstance(data, (list, tuple)):
            return "imgset"
        p = Path(data)
        for cand in reversed(p.parts[:-1] if p.name in ("good", "train", "test") else p.parts):
            if cand not in ("good", "train", "test"):
                return cand
        return p.stem or p.name
