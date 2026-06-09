#!/usr/bin/env python
"""QueryFiLM attention visualizer — per-image grid of all K query attention maps.

A sibling of ``test_predict_visual.py`` specialized for ``fusion_mode: queryfilm`` checkpoints.
For every MVTec test image it runs predict with the memory-bank heatmap prior
(``prior_mode="heatmap"``) and captures the K query attention maps the QueryFiLM module produces
(``model._qf_aux_buf["A"]``, shape ``(1, K, h, w)``), then saves one figure per image:

  Top row : original | mb heatmap (prior) | heatmap-prior detection
  Below   : 4-col grid of all K query attention maps overlaid on the image, sorted by query
            objectness (desc), each labeled ``q<idx> obj=<sigmoid>``.

Attention only exists when the prior is non-empty (the model skips fusion when ``mask is None``),
so the honest, deployable memory-bank heatmap drives it here — not the leaky GT mask. The memory
bank is built once per category and cached (``banks/<category>.pt``), like ``test_predict_visual.py``.
Requires the bbeval YAML (``bb_layers`` re-added) so a bank exists for ``prior_mode="heatmap"``.

Usage:
  python test_predict_visual_qf.py \
      --ckpt runs/yoloa_v2/26m_yoloav2_queryfilm_k16_gauss_v2/weights/best.pt \
      --yaml yolo26m-anomaly-v2-queryfilm-bbeval.yaml \
      --device mps --out runs/temp/predict_visual_qf/queryfilm-k16-gauss-bbeval_v2
  python test_predict_visual_qf.py --ckpt ... --yaml ... --category zipper --n-per-category 3
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


def query_attention_grid(original_rgb, a_khw, obj_k, panel: int, cols: int = 4) -> np.ndarray:
    """Overlay every query attention map on the image, sorted by objectness, into a grid.

    Args:
        original_rgb: RGB uint8 image, already resized to ``(panel, panel)``.
        a_khw: ``(K, h, w)`` float tensor of sigmoid attention maps.
        obj_k: ``(K,)`` float tensor of sigmoid objectness.
        panel: per-panel side length in pixels.
        cols: columns per row (K/cols rows).

    Returns:
        RGB grid (uint8) of all K overlays.
    """
    order = obj_k.argsort(descending=True).tolist()  # active queries first
    panels = []
    for qi in order:
        heat = a_khw[qi].detach().cpu().float().numpy()
        ov = CompareGrid.heatmap_overlay(original_rgb, heat)  # resizes heat -> panel
        amax = float(a_khw[qi].max())
        cv2.putText(ov, f"q{qi} obj={float(obj_k[qi]):.2f} a<={amax:.2f}", (6, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        panels.append(ov)
    return concat_samh(panels, gap=8, gap_color=(255, 255, 255), cols=cols)


# High-contrast colors (RGB), assigned to active queries by objectness rank so the few
# active queries are always maximally distinct (independent of their raw index).
_DISTINCT = np.array([
    [230, 30, 30], [30, 200, 30], [50, 90, 240], [245, 160, 20], [220, 40, 220],
    [30, 200, 200], [235, 225, 30], [150, 40, 220], [240, 110, 160], [120, 200, 40],
    [40, 140, 120], [180, 110, 40],
], dtype=np.float32)


def query_ownership_panel(original_rgb, a_khw, obj_k, obj_thresh: float, a_thresh: float) -> np.ndarray:
    """Color each pixel by the query that owns it (argmax over active queries).

    Owner ``= argmax_k A_k * sigmoid(obj_k)`` restricted to active queries
    (``sigmoid(obj_k) > obj_thresh``); pixels whose winning attention is below ``a_thresh``
    keep the original image (unclaimed background). Active queries get high-contrast colors by
    objectness rank, and each owned region is labeled ``q<idx>`` at its centroid — so whether
    distinct queries partition distinct regions (vs one greedy query claiming all) is obvious.
    """
    out = original_rgb.copy()
    panel = out.shape[0]
    a = a_khw.detach().cpu().float()
    obj = obj_k.detach().cpu().float()
    active = (obj > obj_thresh).nonzero(as_tuple=True)[0].tolist()
    if not active:
        cv2.putText(out, "no active query", (8, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return out
    active.sort(key=lambda k: float(obj[k]), reverse=True)  # strongest query -> first color
    color_of = {k: _DISTINCT[i % len(_DISTINCT)] for i, k in enumerate(active)}

    weighted = a * obj[:, None, None]
    mask = torch.full_like(obj, False, dtype=torch.bool)
    mask[active] = True
    weighted[~mask] = -1.0
    owner = weighted.argmax(0).numpy()  # (h, w), index of winning active query
    a_owner = np.take_along_axis(a.numpy(), owner[None], axis=0)[0]  # (h, w)
    valid = a_owner > a_thresh

    owner_up = cv2.resize(owner.astype(np.int32), (panel, panel), interpolation=cv2.INTER_NEAREST)
    valid_up = cv2.resize(valid.astype(np.uint8), (panel, panel),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    min_area = panel * panel * 0.002
    for k in active:
        sel = valid_up & (owner_up == k)
        if sel.sum() < min_area:
            continue
        out[sel] = (0.6 * color_of[k] + 0.4 * out[sel]).astype(np.uint8)
        ys, xs = np.where(sel)
        cv2.putText(out, f"q{k}", (int(xs.mean()) - 10, int(ys.mean())),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def _vstack_pad(rows: list[np.ndarray], gap: int = 12, color: int = 255) -> np.ndarray:
    """Center-pad rows to a common width, then vstack with a white spacer."""
    w = max(r.shape[1] for r in rows)
    out = []
    for r in rows:
        if r.shape[1] < w:
            extra = w - r.shape[1]
            r = cv2.copyMakeBorder(r, 0, 0, extra // 2, extra - extra // 2,
                                   cv2.BORDER_CONSTANT, value=(color, color, color))
        out.append(r)
    sep = np.full((gap, w, 3), color, dtype=np.uint8)
    stacked = out[0]
    for r in out[1:]:
        stacked = np.vstack([stacked, sep, r])
    return stacked


def main():
    parser = argparse.ArgumentParser(description="QueryFiLM per-image attention-map grid archive")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--yaml", type=str, required=True,
                        help="bbeval YAML (bb_layers re-added so prior_mode='heatmap' has a bank)")
    parser.add_argument("--category", type=str, default=None,
                        help="Single category (default: all 15). 'all' = all.")
    parser.add_argument("--n-per-category", type=int, default=20,
                        help="Max test images per category (0 = all)")
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.1, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max_bank", type=int, default=10000)
    parser.add_argument("--max_images", type=int, default=1000,
                        help="Cap on normal images for bank (0=all)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--panel", type=int, default=320, help="Per-panel side length in pixels")
    parser.add_argument("--obj-thresh", type=float, default=0.3,
                        help="sigmoid(obj) above which a query counts as active (ownership map)")
    parser.add_argument("--a-thresh", type=float, default=0.3,
                        help="winning attention below which a pixel stays unclaimed (ownership map)")
    parser.add_argument("--heat-norm", type=str, default="minmax", choices=["none", "minmax"],
                        help="Per-image prior normalization before fusion (minmax stretches to [0,1])")
    parser.add_argument("--out", type=str, default=None,
                        help="Output dir (default: runs/temp/predict_visual_qf/<run_id>)")
    parser.add_argument("--seed", type=int, default=0,
                        help="RNG seed for per-category sampling (same images across models)")
    args = parser.parse_args()

    random.seed(args.seed)
    device = select_device(args.device or "cpu")

    ckpt_path = Path(args.ckpt).resolve()
    run_id = ckpt_path.parents[1].name if ckpt_path.parent.name == "weights" else ckpt_path.stem
    out_root = Path(args.out) if args.out else Path(f"runs/temp/predict_visual_qf/{run_id}")
    banks_dir = out_root / "banks"
    banks_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(f"Archive dir: {out_root}")

    # ---- Build model ONCE (weights shared across categories) ----
    LOGGER.info("Building model...")
    model = YOLOAnomalyV2Model(args.yaml, nc=1, verbose=False)
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        inner = ckpt.get("ema") or ckpt.get("model")
        ckpt_state = inner.state_dict() if hasattr(inner, "state_dict") else inner
    else:
        ckpt_state = ckpt
    ms = model.state_dict()
    matched = {k: v for k, v in ckpt_state.items() if k in ms and ms[k].shape == v.shape}
    model.load_state_dict(matched, strict=False)
    _report_load(matched, ckpt_state, ms)
    model.to(device)
    model.eval()
    model.heatmap_norm = args.heat_norm
    model._qf_capture = True  # force QueryFiLM aux capture in eval (read _qf_aux_buf after predict)
    LOGGER.info(f"heatmap_norm = {model.heatmap_norm}")

    if model.fusion_mode != "queryfilm":
        raise SystemExit(f"--yaml is not a queryfilm model (fusion_mode={model.fusion_mode!r})")
    if model.memory_bank is None:
        raise SystemExit("Model has no memory bank (no bb_layers) — prior_mode='heatmap' is unavailable. "
                         "Use the bbeval YAML that re-adds bb_layers.")
    qf = model.queryfilm_fusion
    LOGGER.info(f"QueryFiLM: K={qf.k}, D={qf.d}, G={qf.g}, P3 channels={qf.c}")

    y = YOLO(args.yaml)
    y.model = model
    cg = CompareGrid()

    if args.category and args.category.lower() != "all":
        cats = [args.category]
    else:
        cats = sorted(d.name for d in MVTEC_ROOT.iterdir() if d.is_dir())
    LOGGER.info(f"Categories ({len(cats)}): {cats}")

    total = 0
    for ci, cat in enumerate(cats, 1):
        test_root = MVTEC_ROOT / cat / "test"
        if not test_root.is_dir():
            LOGGER.warning(f"[{ci}/{len(cats)}] {cat}: no test/ dir, skipping")
            continue

        # ---- Per-category memory bank: restore from cache, or build once + save ----
        mb = model.memory_bank
        bank_path = banks_dir / f"{cat}.pt"
        if bank_path.exists():
            restore_bank(mb, bank_path)
            LOGGER.info(f"[{ci}/{len(cats)}] {cat}: loaded cached bank ({mb.memory_bank.shape[0]} vecs)")
        else:
            train_dir = MVTEC_ROOT / cat / "train" / "good"
            if not train_dir.is_dir():
                train_dir = MVTEC_ROOT / cat / "train"
            LOGGER.info(f"[{ci}/{len(cats)}] {cat}: building bank from {train_dir} ...")
            mb.reset_memory_bank()
            model.load_support_set(str(train_dir), imgsz=args.imgsz, device=device,
                                   batch=args.batch, max_bank_size=args.max_bank,
                                   max_images=args.max_images, verbose=False)
            save_bank(mb, bank_path)
            LOGGER.info(f"[{ci}/{len(cats)}] {cat}: bank built+saved ({mb.memory_bank.shape[0]} vecs) -> {bank_path}")

        # ---- Sample test images + save per-image attention grids ----
        samples = collect_test_images(test_root, args.n_per_category)
        cat_out = out_root / cat
        for img_path, label in samples:
            original = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
            orig_p = cv2.resize(original, (args.panel, args.panel))

            res = y.predict(img_path, imgsz=args.imgsz, prior_mode="heatmap",
                            conf=args.conf, iou=args.iou, device=device, verbose=False)
            r = res[0]
            n_heat = r.boxes.shape[0] if r.boxes is not None else 0
            pred = cv2.resize(cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB), (args.panel, args.panel))
            hmap = getattr(model, "_last_heatmap", None)
            hmap_np = hmap.cpu().numpy().squeeze() if hmap is not None else None

            aux = getattr(model, "_qf_aux_buf", None)
            if aux is None:
                LOGGER.warning(f"{cat}/{label}/{Path(img_path).stem}: no query aux captured, skipping")
                continue
            a_khw = aux["A"][0]                       # (K, h, w)
            obj_k = aux["obj_logits"][0].sigmoid()    # (K,)

            owner_p = query_ownership_panel(orig_p, a_khw, obj_k, args.obj_thresh, args.a_thresh)
            top = concat_samh(
                [cg._title(orig_p, "original"),
                 cg._title(CompareGrid.heatmap_panel(orig_p, hmap_np), "mb heatmap"),
                 cg._title(pred, f"heatmap prior ({n_heat} det)"),
                 cg._title(owner_p, "query ownership")],
                gap=12, gap_color=(255, 255, 255), cols=4,
            )
            grid = query_attention_grid(orig_p, a_khw, obj_k, args.panel, cols=4)
            fig = _vstack_pad([top, grid], gap=16)

            out_path = cat_out / f"{label}__{Path(img_path).stem}.jpg"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out_path), cv2.cvtColor(fig, cv2.COLOR_RGB2BGR))
            total += 1
        LOGGER.info(f"[{ci}/{len(cats)}] {cat}: {len(samples)} grids -> {cat_out}")

    LOGGER.info(f"Done. {total} attention grids across {len(cats)} categories -> {out_root}")


if __name__ == "__main__":
    main()
