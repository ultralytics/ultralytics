#!/usr/bin/env python
"""Test script: load_support_set → prior_mode=\"heatmap\" predict + val.

Builds the BackboneMemoryBank from MVTec leather training (normal) images,
then evaluates image/pixel AUROC on the test set.

Usage:
    python test_heatmap_prior.py
    python test_heatmap_prior.py --category leather --imgsz 320 --max_bank 5000
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import LOGGER
from ultra_ext.im import concat_samh


def _report_load(matched: dict, ckpt_state: dict, model_state: dict) -> None:
    """Log which keys matched, which are missing in ckpt (random init), and
    which are missing in model (YAML mismatch / removed modules)."""
    ckpt_keys = set(ckpt_state.keys())
    model_keys = set(model_state.keys())
    matched_keys = set(matched.keys())
    missing_in_ckpt = model_keys - matched_keys
    missing_in_model = ckpt_keys - matched_keys

    LOGGER.info(
        f"Weight load: {len(matched)}/{len(ckpt_keys)} ckpt keys matched "
        f"(model has {len(model_keys)} total)"
    )
    if missing_in_model:
        grouped = _group_keys(missing_in_model)
        LOGGER.warning(
            f"  {len(missing_in_model)} keys in ckpt but NOT in model:\n"
            + "\n".join(f"    [{p}] {', '.join(sorted(ks)[:6])}{'...' if len(ks) > 6 else ''}"
                        for p, ks in grouped)
        )
    if missing_in_ckpt:
        grouped = _group_keys(missing_in_ckpt)
        LOGGER.warning(
            f"  {len(missing_in_ckpt)} keys in model but NOT in ckpt (RANDOM INIT):\n"
            + "\n".join(f"    [{p}] {', '.join(sorted(ks)[:6])}{'...' if len(ks) > 6 else ''}"
                        for p, ks in grouped)
        )


def _group_keys(keys: set[str]) -> list[tuple[str, list[str]]]:
    """Group state-dict keys by top-level prefix (e.g. 'model.2', 'seg_branch')."""
    groups: dict[str, list[str]] = {}
    for k in sorted(keys):
        prefix = k.split(".")[0]
        groups.setdefault(prefix, []).append(k)
    return sorted(groups.items())


def main():
    parser = argparse.ArgumentParser(description="Test heatmap prior mode end-to-end")
    parser.add_argument("--ckpt", type=str,
                        default="/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa_v2/"
                                "26m_yoloav2_softhint_rect_pd50_seg_a1_v1/weights/best.pt")
    parser.add_argument("--yaml", type=str,
                        default="yolo26m-anomaly-v2.yaml")
    parser.add_argument("--category", type=str, default="carpet")
    parser.add_argument("--imgsz", type=int, default=448)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_bank", type=int, default=5000)
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Cap on normal images (0=all). Set to 100 for smoke tests.")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device or "cpu"
    LOGGER.info(f"Using device: {device}")

    # Paths
    mvtec_root = Path("/Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO")
    train_dir = mvtec_root / args.category / "train" / "good"
    if not train_dir.is_dir():
        # Flat structure — train images directly in train/
        train_dir = mvtec_root / args.category / "train"
    if not train_dir.is_dir():
        LOGGER.error(f"Train directory not found: {train_dir}")
        sys.exit(1)

    data_yaml = mvtec_root / args.category / f"{args.category}_binary.yaml"
    if not data_yaml.is_file():
        LOGGER.error(f"Data YAML not found: {data_yaml}")
        sys.exit(1)

    LOGGER.info(f"Train images from: {train_dir}")
    LOGGER.info(f"Data YAML: {data_yaml}")

    # ------------------------------------------------------------------
    # 1. Build model from YAML (ensures memory_bank and bb_layers exist)
    # ------------------------------------------------------------------
    LOGGER.info("Creating model from YAML config...")
    model = YOLOAnomalyV2Model(args.yaml, nc=1, verbose=False)
    LOGGER.info(f"Model has memory_bank: {model.memory_bank is not None}")
    LOGGER.info(f"bb_layers: {model._bb_layers}")

    # Load trained weights on top
    LOGGER.info(f"Loading weights from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        inner = ckpt.get("ema") or ckpt.get("model")
        ckpt_state = inner.state_dict() if hasattr(inner, "state_dict") else inner
    else:
        ckpt_state = ckpt
    # intersect
    model_state = model.state_dict()
    matched = {k: v for k, v in ckpt_state.items()
               if k in model_state and model_state[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)

    # ---- Report loading status ----
    _report_load(matched, ckpt_state, model_state)

    model.to(device)
    model.eval()

    # ------------------------------------------------------------------
    # 2. Build memory bank from training images
    # ------------------------------------------------------------------
    LOGGER.info("Building memory bank...")
    n_bank = model.load_support_set(
        str(train_dir),
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        max_bank_size=args.max_bank,
        max_images=args.max_images,
        verbose=True,
    )
    LOGGER.info(f"Bank size: {n_bank}")

    # ------------------------------------------------------------------
    # 3. Predict with heatmap prior
    # ------------------------------------------------------------------
    # Find a test image (prefer an anomalous one in a subfolder like test/glue/)
    test_root = mvtec_root / args.category / "test"
    anomaly_img = None
    # Look for images in subdirs (different defect types)
    for subdir in sorted(test_root.iterdir()):
        if subdir.is_dir():
            imgs = sorted(subdir.glob("*.png"))
            if imgs:
                anomaly_img = str(imgs[0])
                break
    if anomaly_img is None:
        # Fallback: any test image
        all_test = sorted(test_root.rglob("*.png"))
        anomaly_img = str(all_test[0]) if all_test else None

    # Use YAML (not .pt) so YOLO picks up anomaly_v2 task → AnomalyV2Predictor/Validator
    y = YOLO(args.yaml)
    y.model = model  # replace random-weight model with our ckpt-loaded + bank-built one

    LOGGER.info(f"Testing predict with prior_mode='heatmap' on {anomaly_img}...")
    res = y.predict(anomaly_img, imgsz=args.imgsz, prior_mode="heatmap", verbose=False)
    n_boxes = res[0].boxes.shape[0] if res[0].boxes is not None else 0
    hmap = model._last_heatmap
    LOGGER.info(f"  Heatmap max: {hmap.max().item():.4f}" if hmap is not None else "  Heatmap: None")

    LOGGER.info(f"Testing predict with prior_mode='segment' on {anomaly_img}...")
    res_seg = y.predict(anomaly_img, imgsz=args.imgsz, prior_mode="segment", verbose=False)
    n_boxes_seg = res_seg[0].boxes.shape[0] if res_seg[0].boxes is not None else 0
    hmap_seg = model._last_heatmap
    LOGGER.info(f"  Boxes: {n_boxes_seg}, Heatmap max: {hmap_seg.max().item():.4f}" if hmap_seg is not None else f"  Boxes: {n_boxes_seg}, Heatmap: None")

    LOGGER.info(f"Testing predict with prior_mode='seg_heatmap' on {anomaly_img}...")
    res_sh = y.predict(anomaly_img, imgsz=args.imgsz, prior_mode="seg_heatmap", verbose=False)
    n_boxes_sh = res_sh[0].boxes.shape[0] if res_sh[0].boxes is not None else 0
    hmap_sh = model._last_heatmap
    LOGGER.info(f"  Boxes: {n_boxes_sh}, Heatmap max: {hmap_sh.max().item():.4f}" if hmap_sh is not None else f"  Boxes: {n_boxes_sh}, Heatmap: None")

    res_none = y.predict(anomaly_img, imgsz=args.imgsz, prior_mode="none", verbose=False)
    n_boxes_none = res_none[0].boxes.shape[0] if res_none[0].boxes is not None else 0
    LOGGER.info(f"  (none mode) Boxes: {n_boxes_none}")

    # ------------------------------------------------------------------
    # 4. Validate: none vs heatmap vs segment vs seg_heatmap vs mask comparison
    # ------------------------------------------------------------------
    LOGGER.info("Running validation with prior_mode='none'...")
    stats_none = y.val(data=str(data_yaml), imgsz=args.imgsz, batch=4, single_cls=True, conf=0.1,
                       prior_mode="none", save=False, plots=False, verbose=False)

    LOGGER.info("Running validation with prior_mode='heatmap'...")
    stats_heat = y.val(data=str(data_yaml), imgsz=args.imgsz, batch=4, single_cls=True, conf=0.1,
                       prior_mode="heatmap", save=False, plots=False, verbose=False)

    LOGGER.info("Running validation with prior_mode='segment'...")
    stats_seg = y.val(data=str(data_yaml), imgsz=args.imgsz, batch=4, single_cls=True, conf=0.1,
                      prior_mode="segment", save=False, plots=False, verbose=False)

    LOGGER.info("Running validation with prior_mode='seg_heatmap'...")
    stats_sh = y.val(data=str(data_yaml), imgsz=args.imgsz, batch=4, single_cls=True, conf=0.1,
                     prior_mode="seg_heatmap", save=False, plots=False, verbose=False)

    LOGGER.info("Running validation with prior_mode='mask' (GT upper-bound)...")
    stats_mask = y.val(data=str(data_yaml), imgsz=args.imgsz, batch=4, single_cls=True, conf=0.1,
                       prior_mode="mask", save=False, plots=False, verbose=False)

    LOGGER.info("=== Comparison: none vs heatmap vs segment vs seg_heatmap vs mask (GT) ===")
    LOGGER.info(f"  {'Metric':<20} {'none':>10} {'heatmap':>10} {'segment':>10} {'seg_hmap':>10} {'mask':>10}")
    LOGGER.info(f"  {'mAP50':<20} {stats_none.box.map50:>10.4f} {stats_heat.box.map50:>10.4f} {stats_seg.box.map50:>10.4f} {stats_sh.box.map50:>10.4f} {stats_mask.box.map50:>10.4f}")
    LOGGER.info(f"  {'mAP50-95':<20} {stats_none.box.map:>10.4f} {stats_heat.box.map:>10.4f} {stats_seg.box.map:>10.4f} {stats_sh.box.map:>10.4f} {stats_mask.box.map:>10.4f}")
    LOGGER.info(f"  {'image_auroc':<20} {stats_none.image_auroc:>10.4f} {stats_heat.image_auroc:>10.4f} {stats_seg.image_auroc:>10.4f} {stats_sh.image_auroc:>10.4f} {stats_mask.image_auroc:>10.4f}")
    LOGGER.info(f"  {'pixel_auroc':<20} {stats_none.pixel_auroc:>10.4f} {stats_heat.pixel_auroc:>10.4f} {stats_seg.pixel_auroc:>10.4f} {stats_sh.pixel_auroc:>10.4f} {stats_mask.pixel_auroc:>10.4f}")
    LOGGER.info("=== Test complete ===")


if __name__ == "__main__":
    main()
