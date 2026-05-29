"""Model registry for h8: tag -> checkpoint + topology + feature taps.

Checkpoint paths come from environment variables so the registry is identical
across westd / seetacloud / any future box. Set these in your shell before
running extract.py:

    export H8_CHAMPION_CKPT=/path/to/champion/weights/best.pt
    export H8_MGN_T3_CKPT=...
    export H8_MGN_T4_CKPT=...
    export H8_T5FIX_CKPT=...
    export H8_SOLIDER_CKPT=...
    export H8_SOLIDER_DIR=/path/to/SOLIDER-REID  # repo root for SOLIDER's `model.py`

The tap strings are forward-pass module names; the loader registers forward hooks
on those modules. For yolo26l-2psa, P4 is `model.6` and P5 is `model.10` (the layer
right before the head). The Stage 1 sanity gate confirms tap shape matches expectation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "champion": {
        "ckpt_env_var": "H8_CHAMPION_CKPT",
        "kind": "yolo_reid",
        "model_yaml": "ultralytics/cfg/models/26/yolo26l-reid-2psa.yaml",
        "tap_p4": "model.6",
        "tap_p5": "model.10",
        "imgsz": 384,
    },
    "mgn-t3": {
        "ckpt_env_var": "H8_MGN_T3_CKPT",
        "kind": "yolo_reid_mgn",
        "model_yaml": "ultralytics/cfg/models/26/yolo26l-reid-2psa-mgn.yaml",
        "tap_p4": "model.6",
        "tap_p5": "model.10",
        "imgsz": 384,
    },
    "mgn-t4": {
        "ckpt_env_var": "H8_MGN_T4_CKPT",
        "kind": "yolo_reid_mgn",
        "model_yaml": "ultralytics/cfg/models/26/yolo26l-reid-2psa-mgn.yaml",
        "tap_p4": "model.6",
        "tap_p5": "model.10",
        "imgsz": 384,
    },
    "t5fix": {
        "ckpt_env_var": "H8_T5FIX_CKPT",
        "kind": "yolo_reid",
        "model_yaml": "ultralytics/cfg/models/26/yolo26l-reid-2psa.yaml",
        "tap_p4": "model.6",
        "tap_p5": "model.10",
        "imgsz": 384,
    },
    "solider": {
        "ckpt_env_var": "H8_SOLIDER_CKPT",
        "kind": "swin",
        "model_yaml": None,
        # SOLIDER-REID's SwinTransformer uses .stages (ModuleList), not .layers.
        "tap_p4": "base.stages.2",  # stage-3 output (stride-16 equiv)
        "tap_p5": "base.stages.3",  # stage-4 output (stride-32 equiv)
        "imgsz": (384, 128),
    },
}


@dataclass
class ModelHandle:
    """Opaque per-model bundle returned by `load_model`."""

    tag: str
    model: Any
    device: str
    embed_fn: Any  # callable: BCHW preprocessed tensor -> L2-normed (B, D)
    taps: dict[str, Any]
    imgsz: Any


def get_model_entry(tag: str) -> dict[str, Any]:
    """Return the registry entry for `tag`, raising KeyError on miss."""
    if tag not in MODEL_REGISTRY:
        raise KeyError(f"unknown model tag {tag!r}; valid: {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[tag]


def load_model(tag: str, device: str = "cuda:0") -> ModelHandle:
    """Load the model for `tag` in eval mode with feature taps registered.

    For `kind == 'yolo_reid'` and `'yolo_reid_mgn'`: loads via Ultralytics YOLO API.
    For `kind == 'swin'`: imports SOLIDER's `make_model` from H8_SOLIDER_DIR.
    """
    entry = get_model_entry(tag)
    ckpt_path = os.environ.get(entry["ckpt_env_var"])
    if not ckpt_path:
        raise EnvironmentError(
            f"set {entry['ckpt_env_var']} to the .pt path for model tag {tag!r}"
        )

    if entry["kind"] in {"yolo_reid", "yolo_reid_mgn"}:
        return _load_yolo_reid(tag, entry, ckpt_path, device)
    if entry["kind"] == "swin":
        return _load_solider(tag, entry, ckpt_path, device)
    raise ValueError(f"unknown kind {entry['kind']!r}")


def _load_yolo_reid(tag: str, entry: dict, ckpt_path: str, device: str) -> ModelHandle:
    import torch
    import torch.nn.functional as F
    from ultralytics import YOLO

    yolo = YOLO(entry["model_yaml"], task="reid")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Ultralytics strip_optimizers() nulls ckpt["model"]; the real weights live in ckpt["ema"].
    if isinstance(ckpt, dict):
        src = ckpt.get("ema") or ckpt.get("model") or ckpt
    else:
        src = ckpt
    src_sd = src.state_dict() if hasattr(src, "state_dict") else src
    dst_sd = yolo.model.state_dict()
    transfer = {k: v for k, v in src_sd.items() if k in dst_sd and v.shape == dst_sd[k].shape}
    missing = set(dst_sd) - set(transfer)
    if missing:
        print(f"[load_model:{tag}] {len(missing)} keys not transferred from ckpt (head re-init?)")
    yolo.model.load_state_dict(transfer, strict=False)
    model = yolo.model.to(device).eval()

    def embed_fn(img: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            out = model(img)
            emb = out[0] if isinstance(out, (list, tuple)) else out
            return F.normalize(emb, dim=-1)

    taps = {
        "p4": _resolve_module(model, entry["tap_p4"]),
        "p5": _resolve_module(model, entry["tap_p5"]),
    }
    return ModelHandle(tag=tag, model=model, device=device, embed_fn=embed_fn, taps=taps, imgsz=entry["imgsz"])


def _load_solider(tag: str, entry: dict, ckpt_path: str, device: str) -> ModelHandle:
    import sys
    import torch
    import torch.nn.functional as F

    solider_dir = os.environ.get("H8_SOLIDER_DIR")
    if not solider_dir:
        raise EnvironmentError("set H8_SOLIDER_DIR to the SOLIDER-REID repo root")
    sys.path.insert(0, solider_dir)
    from config import cfg as solider_cfg
    from model import make_model as solider_make_model

    solider_cfg.merge_from_file(f"{solider_dir}/configs/market/swin_base.yml")
    solider_cfg.MODEL.SEMANTIC_WEIGHT = 0.2
    solider_cfg.freeze()
    model = solider_make_model(
        solider_cfg,
        num_class=751,
        camera_num=6,
        view_num=0,
        semantic_weight=solider_cfg.MODEL.SEMANTIC_WEIGHT,
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()

    def embed_fn(img: "torch.Tensor") -> "torch.Tensor":
        with torch.no_grad():
            feat = model(img)
            if isinstance(feat, (list, tuple)):
                feat = feat[0]
            return F.normalize(feat, dim=-1)

    taps = {
        "p4": _resolve_module(model, entry["tap_p4"]),
        "p5": _resolve_module(model, entry["tap_p5"]),
    }
    return ModelHandle(tag=tag, model=model, device=device, embed_fn=embed_fn, taps=taps, imgsz=entry["imgsz"])


def _resolve_module(model, dotted: str):
    """Resolve a dotted module path like 'model.6' or 'base.layers.2'."""
    parts = dotted.split(".")
    obj = model
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return obj
