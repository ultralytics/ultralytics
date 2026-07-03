# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLOA — training-free anomaly detection (v2) wrapper.

Fit a memory bank on normal images ("fit" = build the bank, no gradient training), then predict /
val with a chosen prior. The fitted weight self-describes: the bank, the fit config (``fit_args``)
and the data label (``fit_data``) all round-trip in the saved ``*.pt``.

Three config layers, each with its own home:
  - architecture + training knobs -> baked in the model yaml / ckpt (frozen after training)
  - bank-build knobs (imgsz / max_images / bb_*) -> the fit config (yaml or kwargs, overridable)
  - prior-shaping knobs (heatmap_norm / heat_edge* / ...) -> predict()/val() kwargs (per-call)

Examples:
    >>> m = YOLOA("best.pt")
    >>> m.fit("bottle/train/good", name="bottle", cfg="yoloa_fit.yaml", bb_K=5)
    >>> m.predict("test.png", prior="heatmap", heat_edge=True)
    >>> m.val(data="bottle.yaml", prior="heatmap")
    >>> m.save("bottle_fitted.pt")  # carries bank + fit_args + fit_data
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import LOGGER, YAML
from ultralytics.utils.anomaly_v2 import FIT_KEYS, apply_bb_overrides

# Public prior selection. Internal model uses ``prior_mode``; these map 1:1 except
# "anomaly_model", which runs an external model to produce a heatmap and injects it through the
# existing external-mask seam (prior_mode="mask").
PRIOR_MODES = ("none", "heatmap", "mask", "anomaly_model")

# Prior-shaping knobs set on the model before forward. Accepts canonical names and short aliases.
_INFER_SET = {
    "heatmap_norm": "heatmap_norm",
    "heatmap_smooth_kernel": "heatmap_smooth_kernel",
    "heatmap_edge_weight": "heatmap_edge_weight",
    "heatmap_edge_p": "heatmap_edge_p",
    "heatmap_edge_m": "heatmap_edge_m",
    "heatmap_edge_sigma": "heatmap_edge_sigma",
    # short aliases
    "heat_norm": "heatmap_norm",
    "heat_smooth_kernel": "heatmap_smooth_kernel",
    "heat_edge": "heatmap_edge_weight",
    "heat_edge_p": "heatmap_edge_p",
    "heat_edge_m": "heatmap_edge_m",
    "heat_edge_sigma": "heatmap_edge_sigma",
}


class YOLOA(Model):
    """Training-free anomaly v2: fit a memory bank on normals, then predict / val with a prior."""

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
        cfg: str | Path | dict | None = None,
        name: str | None = None,
        cache: str | Path | None = None,
        refit: bool = False,
        device: Any = None,
        batch: int = 8,
    ) -> "YOLOA":
        """Build (or load from cache) the memory bank from normal images.

        Args:
            source: Directory of normal images, or a list of image paths.
            cfg: Fit-config yaml path or dict (imgsz / max_images / bb_*); ``kw`` overrides it,
                which overrides the model yaml's v2_cfg defaults.
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
        if mb is None or getattr(m, "_bb_layers", None) is None:
            raise RuntimeError("model has no BackboneMemoryBank (no bb_layers in the model yaml) — cannot fit().")

        fit_args = cfg if isinstance(cfg, dict) else YAML.load(cfg)
        data_name = name or self._derive_name(source)
        self._apply_bb_overrides(fit_args)

        device = device if device is not None else next(m.parameters()).device
        cache_path = (Path(cache) / f"{self._cache_key(fit_args, data_name)}.pt") if cache is not None else None
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
                imgsz=int(fit_args["imgsz"]),
                device=device,
                batch=batch,
                max_bank_size=fit_args.get("bb_max_bank_size"),
                max_images=int(fit_args["max_images"] or 0),
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

        m.fit_args = dict(fit_args)  # provenance — plain attrs, ride along on save()
        m.fit_data = data_name
        m._heatmap_bank_warned = False
        return self

    def update_memory_bank(
        self,
        source: str | Path | list[str] | None = None,
        remove_indices: torch.Tensor | list[int] | None = None,
        reset: bool = False,
        *,
        cfg: str | Path | dict | None = None,
        device: Any = None,
        batch: int = 8,
    ) -> "YOLOA":
        """Add, remove, or reset the memory bank.

        Exactly one of ``source``, ``remove_indices``, or ``reset`` must be supplied.

        Args:
            source: Directory of normal images or list of paths to add.
            remove_indices: Flat indices of bank entries to remove.
            reset: If True, clear the bank.
            cfg: Fit-config yaml/dict for add (uses ``imgsz``, ``bb_max_bank_size``, ``max_images``).
            device: Device for feature extraction.
            batch: Mini-batch size for feature extraction.

        Returns:
            (YOLOA): self.
        """
        self._check_is_pytorch_model()
        m = self.model
        mb = getattr(m, "memory_bank", None)
        if mb is None or getattr(m, "_bb_layers", None) is None:
            raise RuntimeError("model has no BackboneMemoryBank — cannot update_memory_bank().")

        n_ops = sum([source is not None, remove_indices is not None, reset])
        if n_ops != 1:
            raise ValueError("Exactly one of source, remove_indices, or reset must be provided.")

        if reset:
            mb.reset_memory_bank()
            m.fit_args = None
            m.fit_data = None
            m._heatmap_bank_warned = False
            return self

        if remove_indices is not None:
            mb.remove_features(remove_indices)
            m._heatmap_bank_warned = False
            return self

        # source is not None
        fit_args = cfg if isinstance(cfg, dict) else YAML.load(cfg) if cfg else getattr(m, "fit_args", {})
        if not fit_args:
            raise ValueError("update_memory_bank(add) requires cfg or a previous fit() to get imgsz/bank knobs.")
        # Never retap different backbone layers; existing bank dimensions must stay valid.
        fit_args.pop("bb_layers", None)
        self._apply_bb_overrides(fit_args)
        device = device if device is not None else next(m.parameters()).device
        features = m.extract_bank_features(
            source,
            imgsz=int(fit_args.get("imgsz", 640)),
            device=device,
            batch=batch,
            max_images=int(fit_args.get("max_images") or 0),
        )
        if features.shape[0] == 0:
            LOGGER.warning("update_memory_bank(add): no features extracted; bank unchanged.")
            return self
        mb.add_features(features)
        m._heatmap_bank_warned = False
        LOGGER.info(f"YOLOA.update_memory_bank: bank now has {mb.memory_bank.shape[0]} vectors")
        return self

    def predict(
        self, source=None, stream: bool = False, prior: str = "heatmap", anomaly_model: Any = None, **kwargs: Any
    ):
        """Predict with an optional prior.

        Args:
            source: Image source (path / array / list), as in :meth:`Model.predict`.
            stream: Stream the source.
            prior: One of "none" / "heatmap" / "mask" / "anomaly_model". Default is
                "heatmap"; if the memory bank is empty a warning is emitted and the
                model falls back to vanilla detection. Pass an explicit ``prior_mask``
                for an external mask.
            anomaly_model: Object exposing ``heatmap(img) -> Tensor[H, W] in [0, 1]``, required
                when ``prior="anomaly_model"``; its heatmap is injected as ``prior_mask``.
            **kwargs: Standard predict args plus prior-shaping knobs (heat_norm / heat_edge / ...).
                A raw ``prior_mask`` can also be passed directly.

        Returns:
            (list[Results]): Prediction results.
        """
        prior_mode, prior_mask = self._resolve_prior(prior, anomaly_model, source, kwargs)
        self._apply_infer_overrides(kwargs)
        return super().predict(
            source=source, stream=stream, prior_mode=prior_mode, prior_mask=prior_mask, **kwargs
        )

    def val(self, validator=None, prior: str = "heatmap", **kwargs: Any):
        """Validate with an optional prior (YOLOAnomalyValidator adds image/pixel AUROC).

        Args:
            validator: Optional custom validator.
            prior: One of "none" / "heatmap" / "mask" (advanced pass-through allowed).
                Default is "heatmap"; if the memory bank is empty a warning is emitted
                and the model falls back to vanilla detection. "anomaly_model" is
                predict-only for now.
            **kwargs: Standard val args plus prior-shaping knobs (heat_norm / heat_edge / ...).
                ``end2end`` and ``hm_gate_blend`` are also accepted and applied to the
                detection head for this call only.

        Returns:
            Validation metrics.
        """
        prior_mode = None if prior is None else str(prior).lower()
        if prior_mode == "anomaly_model":
            raise NotImplementedError(
                "prior='anomaly_model' is predict-only for now (needs per-image "
                "external heatmaps through the val dataloader)."
            )

        # Optional head mutations for this val call only (mirror run_yoloa.py --mode val).
        e2e = kwargs.pop("end2end", None)
        gate_blend = kwargs.pop("hm_gate_blend", None)
        heads = [self.model.model[-1]]
        if getattr(self.model, "two_head", False):
            heads.append(self.model.head_b)
        old = {}
        if e2e is not None:
            old["model_e2e"] = getattr(self.model, "end2end", None)
            old["head_e2e"] = [getattr(h, "end2end", None) for h in heads]
            self.model.end2end = e2e
            for h in heads:
                h.end2end = e2e
        if gate_blend is not None:
            old["head_gate"] = [getattr(h, "hm_gate_blend", None) for h in heads]
            for h in heads:
                h.hm_gate_blend = gate_blend

        self._apply_infer_overrides(kwargs)
        try:
            return super().val(validator=validator, prior_mode=prior_mode, **kwargs)
        finally:
            if e2e is not None:
                self.model.end2end = old["model_e2e"]
                for h, v in zip(heads, old["head_e2e"]):
                    if v is None:
                        h.__dict__.pop("end2end", None)
                    else:
                        h.end2end = v
            if gate_blend is not None:
                for h, v in zip(heads, old["head_gate"]):
                    if v is None:
                        h.__dict__.pop("hm_gate_blend", None)
                    else:
                        h.hm_gate_blend = v

    # ---- internals ----------------------------------------------------------------------------

    def _resolve_prior(self, prior, anomaly_model, source, kwargs):
        """Map the public ``prior`` to (prior_mode, prior_mask).

        ``prior_mode`` is only used to enable the internal memory-bank heatmap path
        (``"heatmap"``). Explicit masks are always carried by ``prior_mask``.
        """
        explicit_mask = kwargs.pop("prior_mask", None)

        if prior is None:
            return None, explicit_mask
        prior = str(prior).lower()
        if prior not in PRIOR_MODES:
            raise ValueError(f"prior={prior!r} not in {PRIOR_MODES}")
        if prior == "anomaly_model":
            if anomaly_model is None or not hasattr(anomaly_model, "heatmap"):
                raise ValueError("prior='anomaly_model' requires anomaly_model with a .heatmap(img) method")
            return None, self._external_heatmap(anomaly_model, source)
        if prior == "mask":
            if explicit_mask is None:
                raise ValueError("prior='mask' requires an explicit prior_mask")
            return None, explicit_mask
        if prior == "none":
            if explicit_mask is not None:
                LOGGER.warning("prior='none' ignores the provided prior_mask")
            return None, None
        return prior, None

    @staticmethod
    def _external_heatmap(anomaly_model, source) -> torch.Tensor:
        """Run an external anomaly model -> (1, 1, H, W) float heatmap to pass as ``prior_mask``."""
        hm = anomaly_model.heatmap(source)
        hm = hm if torch.is_tensor(hm) else torch.as_tensor(hm)
        hm = hm.float()
        while hm.dim() < 4:
            hm = hm.unsqueeze(0)
        return hm

    def _apply_infer_overrides(self, kwargs: dict) -> None:
        """Pop prior-shaping kwargs and set them on the model (read during forward)."""
        for key in list(kwargs):
            if key in _INFER_SET:
                setattr(self.model, _INFER_SET[key], kwargs.pop(key))

    def _apply_bb_overrides(self, fit_args: dict) -> None:
        """Apply bb_* overrides onto the model/bank before building."""
        apply_bb_overrides(self.model, fit_args)

    @staticmethod
    def _fit_hash(fit_args: dict) -> str:
        """8-char hash of the resolved fit config (imgsz + bank knobs) — the fit identity."""
        blob = json.dumps({k: fit_args.get(k) for k in FIT_KEYS}, sort_keys=True, default=str)
        return hashlib.md5(blob.encode()).hexdigest()[:8]

    @staticmethod
    def _cache_key(fit_args: dict, data_name: str) -> str:
        """Bank cache filename = data label + fit hash (different categories / fit configs never collide)."""
        return f"{data_name}_{YOLOA._fit_hash(fit_args)}"

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
