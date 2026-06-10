"""Convert a pickled detr_decoder YOLODETR checkpoint to a clean YOLODETR-compatible .pt.

Loads a source checkpoint that was trained on the ``detr_decoder`` branch (which references
removed symbols like ``deformable_attention_core_func_v2``, ``MSDeformAttnFunction``,
``dab_sine_embedding``), rebuilds the model under the clean ``detr_decoder_clean`` module
graph, transfers the state-dict 1:1, preserves training artifacts
(``train_args``/``train_metrics``/``train_results``/``git``), optionally patches headline mAP
fields with pycocotools eval numbers, and saves a portable checkpoint that loads with no
shims.

For non-COCO source checkpoints (e.g. Objects365 with 365 classes), pass ``--data`` pointing to
the dataset YAML. The conversion will read ``nc`` and ``names`` from it and pass them through
to the rebuilt model so the head dimensions match the source.

Examples:
    Basic COCO conversion (writes <stem>_clean.pt next to the source)
    >>> python yolodetr_convert.py --src ~/deimv2XL_coco.pt

    Obj365 conversion preserving 365 classes
    >>> python yolodetr_convert.py --src ~/yolo27x-detr-objv2.pt \
    ...     --data working_dir/datasets/Objects365v2.yaml --verify

    Patch headline metrics with pycocotools eval values via inline JSON
    >>> python yolodetr_convert.py --src ~/deimv2XL_coco.pt --coco-eval \
    ...     '{"mAP50_95": 0.599, "mAP50": 0.776, "mAP75": 0.653, \
    ...       "AP_small": 0.432, "AP_medium": 0.648, "AP_large": 0.774}'
"""

from __future__ import annotations

import argparse
import json
import shutil
import warnings
from pathlib import Path

import torch
import yaml

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parents[2] / "ultralytics2"
DEFAULT_YAML = REPO_ROOT / "ultralytics/cfg/models/27/yolo27x-detr.yaml"
DEFAULT_BUS = REPO_ROOT / "ultralytics/assets/bus.jpg"


def install_pickle_shims() -> None:
    """Register dummy stubs at import paths the source pickle references but the clean code no longer defines."""
    import ultralytics.nn.modules.utils as utm

    def _stub(*a, **kw):
        raise NotImplementedError("unpickle-time stub")

    for name in ("deformable_attention_core_func_v2", "MSDeformAttnFunction", "dab_sine_embedding"):
        if not hasattr(utm, name):
            setattr(utm, name, _stub)


def load_source(src: Path) -> dict:
    """Unpickle the source checkpoint with shims in place; returns the full ckpt dict."""
    install_pickle_shims()
    return torch.load(src, map_location="cpu", weights_only=False)


def load_dataset_meta(data_yaml: Path) -> tuple[int, dict]:
    """Read class count and names mapping from an Ultralytics dataset YAML.

    Args:
        data_yaml (Path): Dataset YAML with ``names`` (dict or list) and optionally ``nc``.

    Returns:
        nc (int): Number of classes derived from ``nc`` or ``len(names)``.
        names (dict): Integer-indexed class-name mapping.
    """
    cfg = yaml.safe_load(Path(data_yaml).read_text())
    names = cfg.get("names")
    if isinstance(names, list):
        names = {i: n for i, n in enumerate(names)}
    nc = int(cfg.get("nc") or (len(names) if names else 0))
    if not nc:
        raise ValueError(f"could not determine nc from {data_yaml}")
    return nc, names


def build_clean_model(yaml_path: Path, source_model, nc: int | None, dtype: torch.dtype):
    """Construct a clean-branch ``YOLODETRDetectionModel``, load source state_dict, and propagate metadata.

    Args:
        yaml_path (Path): Model YAML config matching the source architecture.
        source_model (nn.Module): Pickled source model carrying state_dict + class metadata.
        nc (int, optional): Class count to override the YAML default. None preserves YAML default.
        dtype (torch.dtype): Target parameter dtype.

    Returns:
        (YOLODETRDetectionModel): Loaded, eval-mode model in target dtype.
    """
    from ultralytics.nn.tasks import YOLODETRDetectionModel

    m = YOLODETRDetectionModel(str(yaml_path), ch=3, nc=nc, verbose=False)
    missing, unexpected = m.load_state_dict(source_model.state_dict(), strict=True)
    assert not missing and not unexpected, f"key mismatch missing={missing[:5]} unexpected={unexpected[:5]}"
    m.to(dtype).eval()
    m.task = "detect"
    m.yaml_file = Path(yaml_path).name
    for attr in ("names", "nc", "stride", "args"):
        if hasattr(source_model, attr):
            setattr(m, attr, getattr(source_model, attr))
    return m


def apply_coco_metrics(train_metrics: dict, coco: dict) -> None:
    """In-place patch of headline metrics with pycocotools AP values."""
    key_map = {
        "mAP50_95": "metrics/mAP50-95(B)",
        "mAP50": "metrics/mAP50(B)",
        "mAP75": "metrics/mAP75(B)",
        "AP_small": "metrics/mAP_small",
        "AP_medium": "metrics/mAP_medium",
        "AP_large": "metrics/mAP_large",
    }
    for k, v in coco.items():
        if k not in key_map:
            raise KeyError(f"unknown coco metric key {k!r}; expected one of {list(key_map)}")
        train_metrics[key_map[k]] = float(v)
    if "mAP50_95" in coco:
        train_metrics["fitness"] = float(coco["mAP50_95"])


def save_clean(src_ckpt: dict, model, out: Path) -> None:
    """Save the rebuilt model in a checkpoint dict that mirrors the source shape."""
    new_ckpt = {**src_ckpt, "model": model}
    torch.save(new_ckpt, out)


def verify_inference(out: Path, bus_path: Path) -> None:
    """Reload the saved checkpoint with no shims and run a sanity inference."""
    import numpy as np
    from PIL import Image

    from ultralytics import YOLODETR

    m = YOLODETR(str(out))
    mdt = next(m.model.parameters()).dtype
    img = Image.open(bus_path).convert("RGB").resize((640, 640))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(mdt)
    m.model.eval()
    with torch.no_grad():
        out_t = m.model(x)
    y = out_t[0] if isinstance(out_t, tuple) else out_t
    top = y[0, y[0, :, 4].argsort(descending=True)][:6]
    names = getattr(m.model, "names", {}) or {}
    print(f"verify: loaded class={type(m).__name__} dtype={mdt} top-6 detections on {bus_path.name}:")
    for d in top:
        cls = int(d[5])
        bb = [round(b, 3) for b in d[:4].tolist()]
        print(f"  cls={cls:3d} ({names.get(cls, '?')}) conf={d[4].item():.3f} bbox={bb}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True, help="source .pt path (detr_decoder branch checkpoint)")
    p.add_argument("--data", type=Path, default=None, help="dataset YAML for non-COCO nc/names (e.g. Objects365v2.yaml)")
    p.add_argument("--out", type=Path, default=None, help="output path (default: <src_stem>_clean.pt)")
    p.add_argument("--yaml", type=Path, default=DEFAULT_YAML, help=f"model YAML (default: {DEFAULT_YAML})")
    p.add_argument("--dtype", choices=["fp16", "fp32"], default="fp32", help="saved model dtype")
    p.add_argument("--copy-to", type=Path, default=None, help="extra destination path to copy the result to")
    p.add_argument(
        "--coco-eval",
        type=str,
        default=None,
        help='JSON dict of pycocotools metrics, e.g. \'{"mAP50_95":0.599,"mAP50":0.776}\'',
    )
    p.add_argument("--verify", action="store_true", help="reload saved ckpt and run bus.jpg sanity inference")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    out = args.out or args.src.with_name(f"{args.src.stem}_clean.pt")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32

    nc, names = (None, None)
    if args.data is not None:
        nc, names = load_dataset_meta(args.data)
        print(f"dataset {args.data.name}: nc={nc}, names_count={len(names) if names else 0}")

    print(f"loading source: {args.src}")
    src = load_source(args.src)
    src_model = src["model"]
    n_params = sum(p.numel() for p in src_model.parameters())
    src_dtype = next(src_model.parameters()).dtype
    src_names = getattr(src_model, "names", {}) or {}
    src_nc = int(getattr(src_model, "nc", len(src_names)) or len(src_names))
    print(f"  source dtype={src_dtype}, params={n_params:,}, nc={src_nc}, names={len(src_names)}")

    if nc is not None and src_nc and src_nc != nc:
        raise ValueError(
            f"source checkpoint nc={src_nc} does not match dataset nc={nc}; "
            f"pass the correct --data yaml (or omit --data for COCO)."
        )

    print(f"rebuilding under clean graph from: {args.yaml} (nc={nc if nc is not None else 'YAML default'})")
    model = build_clean_model(args.yaml, src_model, nc, dtype)

    if not src_names and names:
        model.names = names

    if args.coco_eval:
        coco = json.loads(args.coco_eval)
        print(f"patching pycocotools metrics: {coco}")
        apply_coco_metrics(src["train_metrics"], coco)

    print(f"saving: {out}")
    save_clean(src, model, out)
    print(f"  size: {out.stat().st_size / 1e6:.1f} MB")

    if args.copy_to:
        shutil.copy(out, args.copy_to)
        print(f"copied to: {args.copy_to}")

    if args.verify:
        verify_inference(out, DEFAULT_BUS)


if __name__ == "__main__":
    main()
