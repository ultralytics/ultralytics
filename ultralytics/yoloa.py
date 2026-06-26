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

# Public prior selection. Internal model uses ``prior_mode``; these map 1:1 except
# "anomaly_model", which runs an external model to produce a heatmap and injects it through the
# existing external-mask seam (prior_mode="mask"). "segment"/"seg_heatmap"/"cached" remain usable
# as advanced pass-through values.
PRIOR_MODES = ("none", "heatmap", "heatmap_learned", "heatmap_fused", "mask", "anomaly_model")
_ADVANCED_PRIORS = ("segment", "seg_heatmap", "cached")

# Bank-build knobs that live in the fit config. bb_* override the model yaml's v2_cfg defaults;
# imgsz / max_images are build inputs (not part of the model yaml).
FIT_KEYS = (
    "imgsz", "max_images", "bb_layers", "bb_max_bank_size", "bb_K", "bb_proj_dim",
    "bb_temperature",
    "bb_calibration_target_score", "bb_calibration_target_quantile",
    "bb_hmap_stretch_strength",
    "bb_holdout_max",
)

# fit-key -> BackboneMemoryBank attribute it sets (bb_layers handled separately: it re-taps).
_BB_TO_MB = {
    "bb_max_bank_size": "max_bank_size", "bb_K": "K",
    "bb_temperature": "temperature",
    "bb_calibration_target_score": "calibration_target_score",
    "bb_calibration_target_quantile": "calibration_target_quantile",
    "bb_proj_dim": "proj_dim",
    "bb_hmap_stretch_strength": "hmap_stretch_strength",
    "bb_holdout_max": "holdout_max",
}

# Prior-shaping knobs set on the model before forward. Accepts canonical names and short aliases.
_INFER_SET = {
    "heatmap_norm": "heatmap_norm", "heatmap_smooth_kernel": "heatmap_smooth_kernel",
    "heatmap_edge_weight": "heatmap_edge_weight", "heatmap_edge_p": "heatmap_edge_p",
    "heatmap_edge_m": "heatmap_edge_m", "heatmap_edge_sigma": "heatmap_edge_sigma",
    "spatial_softmax": "spatial_softmax", "softmax_temperature": "softmax_temperature",
    "mask_remap_mode": "mask_remap_mode", "mask_remap_kwargs": "mask_remap_kwargs",
    # short aliases
    "heat_norm": "heatmap_norm", "heat_smooth_kernel": "heatmap_smooth_kernel",
    "heat_edge": "heatmap_edge_weight", "heat_edge_p": "heatmap_edge_p",
    "heat_edge_m": "heatmap_edge_m", "heat_edge_sigma": "heatmap_edge_sigma",
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
                "validator": yolo.anomaly_v2.AnomalyV2Validator,
                "predictor": yolo.anomaly_v2.AnomalyV2Predictor,
            }
        }

    @property
    def is_fitted(self) -> bool:
        """True if a non-empty memory bank is loaded (ready for prior='heatmap')."""
        mb = getattr(self.model, "memory_bank", None)
        bank = getattr(mb, "memory_bank", None) if mb is not None else None
        return bank is not None and bank.numel() > 0 and bank.shape[0] > 0

    def fit(
        self,
        data: str | Path | list[str],
        cfg: str | Path | dict | None = None,
        name: str | None = None,
        cache: str | Path | None = None,
        refit: bool = False,
        device: Any = None,
        batch: int = 8,
        fit_disc: bool | dict = False,
        **kw: Any,
    ) -> "YOLOA":
        """Build (or load from cache) the memory bank from normal images.

        Args:
            data: Directory of normal images, or a list of image paths.
            cfg: Fit-config yaml path or dict (imgsz / max_images / bb_*); ``kw`` overrides it,
                which overrides the model yaml's v2_cfg defaults.
            name: Friendly label for provenance + cache key (default: derived from ``data``).
            cache: Directory to cache/reuse the built bank on disk; None disables caching.
            refit: Force a rebuild even if a cache file exists.
            device: Build device (defaults to the model device); not part of the fit identity.
            batch: Mini-batch size for feature extraction; not part of the fit identity.
            fit_disc: If True or a dict, also fit a FeatureDiscriminatorScorer on the bank's
                normal features (for prior="heatmap_learned" / "heatmap_fused"). A dict forwards
                as kwargs (noise_std, steps, hidden, ...).
            **kw: bb_* / imgsz / max_images overrides (highest priority).

        Returns:
            (YOLOA): self, now fitted.
        """
        self._check_is_pytorch_model()
        m = self.model
        mb = getattr(m, "memory_bank", None)
        if mb is None or getattr(m, "_bb_layers", None) is None:
            raise RuntimeError("model has no BackboneMemoryBank (no bb_layers in the model yaml) — cannot fit().")

        fit_args = self._resolve_fit_args(self._load_cfg(cfg), kw)
        data_name = name or self._derive_name(data)
        self._apply_bb_overrides(fit_args)

        device = device if device is not None else next(m.parameters()).device
        cache_path = (Path(cache) / f"{self._cache_key(fit_args, data_name)}.pt") if cache is not None else None
        if cache_path is not None and cache_path.exists() and not refit:
            d = torch.load(cache_path, map_location="cpu")
            mb.load_bank(d["memory_bank"])
            mb.temperature = d["temperature"]
            mb.update = False
            if d.get("_calibrated"):
                mb._threshold = d["_threshold"]
                mb._compactness = d["_compactness"]
                mb._calibrated = True
            LOGGER.info(f"YOLOA.fit: loaded cached bank ({mb.memory_bank.shape[0]} vecs) <- {cache_path}")
        else:
            n = m.load_support_set(
                data, imgsz=int(fit_args["imgsz"]), device=device, batch=batch,
                max_bank_size=fit_args.get("bb_max_bank_size"),
                max_images=int(fit_args["max_images"] or 0), verbose=True,
                fit_disc=fit_disc,
            )
            if cache_path is not None and n:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                entry = {"memory_bank": mb.memory_bank.detach().cpu(),
                         "feature_dim": mb.feature_dim, "temperature": float(mb.temperature)}
                if getattr(mb, "_calibrated", False):
                    entry["_threshold"] = mb._threshold
                    entry["_compactness"] = mb._compactness
                    entry["_calibrated"] = True
                torch.save(entry, cache_path)
                LOGGER.info(f"YOLOA.fit: cached bank ({mb.memory_bank.shape[0]} vecs) -> {cache_path}")

        m.fit_args = dict(fit_args)  # provenance — plain attrs, ride along on save()
        m.fit_data = data_name
        return self

    def predict(self, source=None, stream: bool = False, prior: str | None = None,
                anomaly_model: Any = None, **kwargs: Any):
        """Predict with an optional prior.

        Args:
            source: Image source (path / array / list), as in :meth:`Model.predict`.
            stream: Stream the source.
            prior: One of "none" / "heatmap" / "mask" / "anomaly_model" (advanced:
                "segment" / "seg_heatmap" / "cached"). None keeps legacy behavior.
            anomaly_model: Object exposing ``heatmap(img) -> Tensor[H, W] in [0, 1]``, required
                when ``prior="anomaly_model"``; its heatmap is injected via the external-mask seam.
            **kwargs: Standard predict args plus prior-shaping knobs (heat_norm / heat_edge / ...).

        Returns:
            (list[Results]): Prediction results.
        """
        prior_mode, external_mask = self._resolve_prior(prior, anomaly_model, source, kwargs)
        self._apply_infer_overrides(kwargs)
        return super().predict(source=source, stream=stream, prior_mode=prior_mode,
                               external_mask=external_mask, **kwargs)

    def val(self, validator=None, prior: str | None = None, **kwargs: Any):
        """Validate with an optional prior (AnomalyV2Validator adds image/pixel AUROC).

        Args:
            validator: Optional custom validator.
            prior: One of "none" / "heatmap" / "mask" (advanced pass-through allowed).
                "anomaly_model" is predict-only for now.
            **kwargs: Standard val args plus prior-shaping knobs (heat_norm / heat_edge / ...).

        Returns:
            Validation metrics.
        """
        prior_mode = None if prior is None else str(prior).lower()
        if prior_mode == "anomaly_model":
            raise NotImplementedError("prior='anomaly_model' is predict-only for now (needs per-image "
                                      "external heatmaps through the val dataloader).")
        self._apply_infer_overrides(kwargs)
        return super().val(validator=validator, prior_mode=prior_mode, **kwargs)

    # ---- internals ----------------------------------------------------------------------------

    def _resolve_prior(self, prior, anomaly_model, source, kwargs):
        """Map the public ``prior`` to (prior_mode, external_mask)."""
        if prior is None:
            return None, kwargs.pop("external_mask", None)
        prior = str(prior).lower()
        if prior not in PRIOR_MODES and prior not in _ADVANCED_PRIORS:
            raise ValueError(f"prior={prior!r} not in {PRIOR_MODES} (advanced: {_ADVANCED_PRIORS})")
        if prior == "anomaly_model":
            if anomaly_model is None or not hasattr(anomaly_model, "heatmap"):
                raise ValueError("prior='anomaly_model' requires anomaly_model with a .heatmap(img) method")
            return "mask", self._external_heatmap(anomaly_model, source)
        return prior, kwargs.pop("external_mask", None)

    @staticmethod
    def _external_heatmap(anomaly_model, source) -> torch.Tensor:
        """Run an external anomaly model -> (1, 1, H, W) float heatmap for the external-mask seam."""
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
        m, mb = self.model, self.model.memory_bank
        new_layers = fit_args.get("bb_layers")
        if new_layers is not None and list(new_layers) != list(m._bb_layers or []):
            new_layers = list(new_layers)
            LOGGER.warning(f"YOLOA.fit: bb_layers {m._bb_layers} -> {new_layers} (rebuilds bank; "
                           f"deviating from the training layer set may shift fusion behavior)")
            for h in getattr(m, "_bb_hook_handles", []):
                h.remove()
            m._bb_hook_handles = []
            m._bb_layers = new_layers
            mb._bb_layer_indices = new_layers
            m._install_backbone_taps(new_layers)
        for fk, mbk in _BB_TO_MB.items():
            if fit_args.get(fk) is not None:
                setattr(mb, mbk, fit_args[fk])

    def resolve_fit_args(self, cfg: str | Path | dict | None = None, **kw) -> dict:
        """Public: the fully resolved fit config (model yaml <- fit cfg <- kwargs)."""
        return self._resolve_fit_args(self._load_cfg(cfg), kw)

    def fit_id(self, cfg: str | Path | dict | None = None, **kw) -> str:
        """8-char hash of the resolved fit config — the fit identity (same hash as the bank cache).

        One fit config -> one id; any change (imgsz / bb_layers / bb_K / calibrate / ...) -> a new id.
        Resolve with the same (cfg, kwargs) you pass to fit().
        """
        return self._fit_hash(self.resolve_fit_args(cfg, **kw))

    def _resolve_fit_args(self, cfg_d: dict, kw: dict) -> dict:
        """Resolve fit args with precedence: kwargs > fit cfg > model yaml v2_cfg defaults."""
        v2 = self.model.yaml.get("anomaly_v2", {}) if isinstance(getattr(self.model, "yaml", None), dict) else {}
        out = {}
        for k in FIT_KEYS:
            if kw.get(k) is not None:
                out[k] = kw[k]
            elif cfg_d.get(k) is not None:
                out[k] = cfg_d[k]
            elif k == "imgsz":
                out[k] = v2.get("imgsz", 640)
            elif k == "max_images":
                out[k] = 0
            else:
                out[k] = v2.get(k)  # bb_* default from the model yaml (may be None)
        return out

    @staticmethod
    def _load_cfg(cfg) -> dict:
        """Load a fit config from a yaml path or dict (None -> {})."""
        if cfg is None:
            return {}
        return dict(cfg) if isinstance(cfg, dict) else dict(YAML.load(cfg))

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
