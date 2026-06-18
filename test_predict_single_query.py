#!/usr/bin/env python
"""Single-query QueryFiLM predictor — run detection with ONLY one query's modulation active.

The query-knockout test showed detection collapses onto a single query (default q11): only that
query's write-back produces clean boxes on the true defect, the other queries inject background
noise. This script runs the "only q<query>" model across all categories/samples so the q11-only
detector can be eyeballed at scale.

Per image, three panels:  original | q<query> attention | q<query>-only detection (N det)

The other queries are silenced via the eval-only hook ``queryfilm_fusion._keep_queries``. Needs the
softmax-bbeval YAML (bb_layers) for the memory-bank heatmap prior. Memory banks are cached per ckpt
under ``--banks-dir`` (shared across runs), so re-runs skip the slow feature extraction.

Usage:
  python test_predict_single_query.py \
      --ckpt .../26m_yoloav2_queryfilm_k16_softmax_v1/weights/best.pt \
      --yaml yolo26m-anomaly-v2-queryfilm-softmax-bbeval.yaml \
      --device mps --query 11                      # all 15 categories, all samples
  python test_predict_single_query.py --ckpt ... --yaml ... --category grid --n-per-category 10
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


def main():
    p = argparse.ArgumentParser(description="Single-query (q-only) QueryFiLM predictor over MVTec")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--yaml", required=True, help="softmax-bbeval YAML (bb_layers for heatmap prior)")
    p.add_argument("--query", type=int, default=11, help="the only query whose write-back stays active")
    p.add_argument("--category", default=None, help="single category, or 'all' (default: all 15)")
    p.add_argument("--n-per-category", type=int, default=0, help="max samples/category (0 = all)")
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
    p.add_argument("--banks-dir", default=None, help="shared bank cache (default runs/temp/qf_banks/<run_id>)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    device = select_device(args.device or "cpu")

    ckpt_path = Path(args.ckpt).resolve()
    run_id = ckpt_path.parents[1].name if ckpt_path.parent.name == "weights" else ckpt_path.stem
    out_root = Path(args.out) if args.out else Path(f"runs/temp/predict_single_query/{run_id}_q{args.query}")
    banks_dir = Path(args.banks_dir) if args.banks_dir else Path(f"runs/temp/qf_banks/{run_id}")
    banks_dir.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Archive dir: {out_root}  | only query = q{args.query}  | banks: {banks_dir}")

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
    if not (0 <= args.query < qf.k):
        raise SystemExit(f"--query {args.query} out of range [0, {qf.k})")
    qf._keep_queries = [args.query]  # silence all other queries' write-back
    model._qf_capture = True         # so we can read the kept query's attention map
    LOGGER.info(f"QueryFiLM K={qf.k}; keeping only q{args.query}")

    y = YOLO(args.yaml)
    y.model = model

    cats = [args.category] if args.category and args.category.lower() != "all" else \
        sorted(d.name for d in MVTEC_ROOT.iterdir() if d.is_dir())

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

        samples = collect_test_images(test_root, args.n_per_category)
        for img_path, label in samples:
            orig = cv2.resize(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB),
                              (args.panel, args.panel))
            res = y.predict(img_path, imgsz=args.imgsz, prior_mode="heatmap",
                            conf=args.conf, iou=args.iou, device=device, verbose=False)
            r = res[0]
            n = r.boxes.shape[0] if r.boxes is not None else 0
            pred = cv2.resize(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), (args.panel, args.panel))
            aux = getattr(model, "_qf_aux_buf", None)
            attn = aux["A"][0, args.query].detach().cpu().float().numpy() if aux is not None else None
            attn_panel = CompareGrid.heatmap_overlay(orig, attn)
            cg = CompareGrid()
            fig = concat_samh(
                [cg._title(orig, "original"),
                 cg._title(attn_panel, f"q{args.query} attn"),
                 cg._title(pred, f"q{args.query} only ({n} det)")],
                gap=12, gap_color=(255, 255, 255), cols=3,
            )
            out_path = out_root / cat / f"{label}__{Path(img_path).stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
            total += 1
        LOGGER.info(f"[{ci}/{len(cats)}] {cat}: {len(samples)} -> {out_root / cat}")
    LOGGER.info(f"Done. {total} images (only q{args.query}) -> {out_root}")


if __name__ == "__main__":
    main()
