#!/usr/bin/env python
"""QueryFiLM query-knockout test — does ONE query carry the detection?

Observation to validate: across images the defect attention concentrates on a single query
(e.g. q11) while the rest look like background noise. This script masks the QueryFiLM write-back
(eval-only hook ``queryfilm_fusion._keep_queries``) to run the memory-bank-prior detection three
ways per image and compare:

  original | all queries | only q<keep> | all-except q<keep>   (det count burned into each panel)

If "only q<keep>" ~= "all queries" and "all-except" collapses to noise, the multi-query design is
collapsed onto one query. Needs the softmax-bbeval YAML (bb_layers) for the heatmap prior.

Usage:
  python test_query_knockout.py \
      --ckpt .../26m_yoloav2_queryfilm_k16_softmax_v1/weights/best.pt \
      --yaml yolo26m-anomaly-v2-queryfilm-softmax-bbeval.yaml \
      --device mps --category zipper --n-per-category 3 --keep 11
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.nn.tasks import YOLOAnomalyV2Model
from ultralytics.utils import LOGGER
from ultralytics.utils.torch_utils import select_device
from ultra_ext.im import concat_samh

from compare_grid import CompareGrid
from test_predict_visual import (
    MVTEC_ROOT,
    _report_load,
    collect_test_images,
    restore_bank,
    save_bank,
)


def predict_panel(y, model, img_path, keep, imgsz, conf, iou, device, panel, title):
    """Run a heatmap-prior predict with a given _keep_queries setting; return a titled RGB panel."""
    model.queryfilm_fusion._keep_queries = keep
    res = y.predict(img_path, imgsz=imgsz, prior_mode="heatmap", conf=conf, iou=iou,
                    device=device, verbose=False)
    model.queryfilm_fusion._keep_queries = None
    r = res[0]
    n = r.boxes.shape[0] if r.boxes is not None else 0
    img = cv2.resize(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), (panel, panel))
    return CompareGrid()._title(img, f"{title} ({n} det)"), n


def main():
    p = argparse.ArgumentParser(description="QueryFiLM query-knockout detection test")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--yaml", required=True, help="softmax-bbeval YAML (bb_layers for heatmap prior)")
    p.add_argument("--category", default=None, help="single category, or 'all'")
    p.add_argument("--n-per-category", type=int, default=3)
    p.add_argument("--keep", type=str, default="11", help="query index/indices to keep (comma-sep)")
    p.add_argument("--conf", type=float, default=0.4)
    p.add_argument("--iou", type=float, default=0.1)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--max_bank", type=int, default=10000)
    p.add_argument("--max_images", type=int, default=1000)
    p.add_argument("--device", default=None)
    p.add_argument("--panel", type=int, default=360)
    p.add_argument("--heat-norm", default="minmax", choices=["none", "minmax"])
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    device = select_device(args.device or "cpu")
    keep = [int(x) for x in args.keep.split(",") if x.strip() != ""]

    ckpt_path = Path(args.ckpt).resolve()
    run_id = ckpt_path.parents[1].name if ckpt_path.parent.name == "weights" else ckpt_path.stem
    out_root = Path(args.out) if args.out else Path(f"runs/temp/query_knockout/{run_id}")
    banks_dir = out_root / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Archive dir: {out_root}  | keep queries = {keep}")

    model = YOLOAnomalyV2Model(args.yaml, nc=1, verbose=False)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    inner = ckpt.get("ema") or ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    ckpt_state = inner.state_dict() if hasattr(inner, "state_dict") else inner
    ms = model.state_dict()
    matched = {k: v for k, v in ckpt_state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    _report_load(matched, ckpt_state, ms)
    model.to(device).eval()
    model.heatmap_norm = args.heat_norm
    if model.fusion_mode != "queryfilm":
        raise SystemExit(f"--yaml is not queryfilm (fusion_mode={model.fusion_mode!r})")
    if model.memory_bank is None:
        raise SystemExit("No memory bank (bb_layers) — use the bbeval YAML.")
    qf = model.queryfilm_fusion
    all_but_keep = [k for k in range(qf.k) if k not in keep]
    LOGGER.info(f"QueryFiLM K={qf.k}; only={keep}; all-except={all_but_keep}")

    y = YOLO(args.yaml)
    y.model = model

    cats = [args.category] if args.category and args.category.lower() != "all" else \
        sorted(d.name for d in MVTEC_ROOT.iterdir() if d.is_dir())

    keep_lbl = "q" + "+".join(map(str, keep))
    total = 0
    for ci, cat in enumerate(cats, 1):
        test_root = MVTEC_ROOT / cat / "test"
        if not test_root.is_dir():
            continue
        mb = model.memory_bank
        bank_path = banks_dir / f"{cat}.pt"
        if bank_path.exists():
            restore_bank(mb, bank_path)
        else:
            train_dir = MVTEC_ROOT / cat / "train" / "good"
            train_dir = train_dir if train_dir.is_dir() else MVTEC_ROOT / cat / "train"
            mb.reset_memory_bank()
            model.load_support_set(str(train_dir), imgsz=args.imgsz, device=device, batch=args.batch,
                                   max_bank_size=args.max_bank, max_images=args.max_images, verbose=False)
            save_bank(mb, bank_path)
        LOGGER.info(f"[{ci}/{len(cats)}] {cat}: bank {mb.memory_bank.shape[0]} vecs")

        for img_path, label in collect_test_images(test_root, args.n_per_category):
            orig = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB), (args.panel, args.panel))
            p_all, _ = predict_panel(y, model, img_path, None, args.imgsz, args.conf, args.iou, device, args.panel, "all queries")
            p_keep, _ = predict_panel(y, model, img_path, keep, args.imgsz, args.conf, args.iou, device, args.panel, f"only {keep_lbl}")
            p_drop, _ = predict_panel(y, model, img_path, all_but_keep, args.imgsz, args.conf, args.iou, device, args.panel, f"without {keep_lbl}")
            fig = concat_samh([CompareGrid()._title(orig, "original"), p_all, p_keep, p_drop],
                              gap=12, gap_color=(255, 255, 255), cols=4)
            out_path = out_root / cat / f"{label}__{Path(img_path).stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
            total += 1
        LOGGER.info(f"[{ci}/{len(cats)}] {cat}: done")
    LOGGER.info(f"Done. {total} comparisons -> {out_root}")


if __name__ == "__main__":
    main()
