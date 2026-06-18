"""Minimal demo: compare for v2 anomaly detection mask prompt.

Run from the repo root:

    cd /Users/louis/workspace/ultra_louis_work/ultralytics
    python docs_yoloa_v2/demo_mask_prompt.py

Writes ONE fixed-path PNG (overwritten each run, so VSCode can keep it open):
    [ original | mask_off | seg_pred | seg_bbox | gt overlay | mask_on ]

Columns 3-4 require a SegBranch in the checkpoint (else blank).
Columns 5-6 require a GT annotation mask (else blank) -- so the script also
runs on unlabelled images.

ultralytics quirks:
  * ``end2end`` / ``max_det`` are MODEL-level -- only baked in on the FIRST
    ``predict()`` call (when ``setup_model`` runs). Pass them on the warmup.
  * ``conf`` / ``iou`` are PER-CALL NMS args.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from ultra_ext.yoloa import get_random_sample
from ultra_ext.im import concat_samh


# ============================================================================
# EDIT
# ============================================================================
# EDIT
# ============================================================================
experiment = "26m_yoloav2_v5_binary_cm20_gauss_pd50_v1"
experiment= '26m_yoloav2seg_v5_binary_cm20_rect_pd50_a1_v1'
experiment= '26m_yoloav2_softhint_rect_pd50_v1'
experiment= '26m_yoloav2_softhint_rect_pd50_seg_a1_v1'


MODEL_PATH = f"/Users/louis/workspace/ultra_louis_work/ultra6/runs/yoloa_v2/{experiment}/weights/best.pt"
SAVE_PATH  = "../runs/temp/demo_mask_prompt/compare.png"
CATEGORY   = "grid"  # from MVTec AD (or "all" for random across all categories)

CONF, IOU, END2END, MAX_DET = 0.1, 0.05, False, 9
GOOD=False  # if True, sample from "good" (non-anomalous) images; else from "bad" (anomalous) ones

def overlay_mask(img_bgr, mask_path, color=(0, 0, 255), alpha=0.45):
    """Red-tint a BGR image wherever the binary mask is non-zero."""
    out = img_bgr.copy()
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return out
    m = cv2.resize(m, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    sel = m > 0
    out[sel] = (alpha * np.array(color) + (1 - alpha) * out[sel]).astype(np.uint8)
    return out

from ultralytics.nn.modules.anomaly_v2 import BboxMaskRenderer

_RENDERER = None

def load_mask_as_prior(path, size=80, scale=0.99, sigma_factor=0.25):
    global _RENDERER
    if _RENDERER is None or _RENDERER.mask_size != size:
        _RENDERER = BboxMaskRenderer(mask_size=size, mode="gauss", sigma_factor=sigma_factor)

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    H, W = img.shape
    ys, xs = np.where(img > 0)
    if len(xs) == 0:
        return torch.zeros(1, 1, size, size)

    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    cx = ((x1 + x2) / 2) / W
    cy = ((y1 + y2) / 2) / H
    w = (x2 - x1) / W
    h = (y2 - y1) / H

    bboxes = torch.tensor([[cx, cy, w, h]], dtype=torch.float32)
    batch_idx = torch.zeros(1, dtype=torch.long)
    mask = _RENDERER(bboxes, batch_idx, batch_size=1)  # (1,1,size,size) in [0,1]
    return mask * scale

def heatmap_to_bbox_prior(heatmap, size=80, thresh=0.5, scale=0.8):
    """Bounding rect of a 2D heatmap (after threshold) -> (1,1,size,size) prior tensor."""
    h = cv2.resize(heatmap.astype("float32"), (size, size), interpolation=cv2.INTER_LINEAR)
    ys, xs = np.where(h > thresh)
    rect = np.zeros((size, size), dtype="float32")
    if len(xs):
        rect[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1] = 1.0
    return torch.from_numpy(rect * scale)[None, None]


def heatmap_overlay(img_bgr, heatmap, alpha=0.45):
    """JET-colormap overlay of a [0,1] heatmap onto a BGR image."""
    h = cv2.resize(heatmap.astype("float32"), (img_bgr.shape[1], img_bgr.shape[0]),
                   interpolation=cv2.INTER_LINEAR)
    h = np.clip(h, 0.0, 1.0)
    cmap = cv2.applyColorMap((h * 255).astype("uint8"), cv2.COLORMAP_JET)
    return cv2.addWeighted(cmap, alpha, img_bgr, 1 - alpha, 0)


def title(img, text, h=72, scale=1.6, thickness=4):
    bar = np.full((h, img.shape[1], 3), 30, np.uint8)  # near-black bar
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x = max(8, (bar.shape[1] - tw) // 2)
    y = (h + th) // 2
    cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack([bar, img])


def blank_like(img):
    """Light-gray placeholder matching ``img`` shape."""
    return np.full_like(img, 230)


def main():
    image_path, mask_path = get_random_sample(CATEGORY, good=GOOD)
    has_anno = mask_path is not None and Path(mask_path).exists()

    y = YOLO(MODEL_PATH)
    print(f"Loaded: {type(y.model).__name__}  mask_mode={y.model.mask_renderer.mode}")

    # Warmup -- bakes end2end/max_det into the model on first call.
    y.predict(image_path, save=False, verbose=False,
              end2end=END2END, max_det=MAX_DET, conf=CONF, iou=IOU)

    img = cv2.imread(str(image_path))

    # 1-2: original + mask-off detection. This forward also populates _seg_logits_buf.
    y.predictor.external_mask = None
    y.predictor.bbox_prompt = None
    r_off = y.predict(image_path, save=False, verbose=False, conf=CONF, iou=IOU)

    panels = [
        title(img, "original"),
        title(r_off[0].plot(), f"mask_off  ({len(r_off[0].boxes)} det)"),
    ]

    # 3-4: SegBranch heatmap + heatmap-guided detection (blank if no seg head).
    seg_branch = getattr(y.model, "seg_branch", None)
    if seg_branch is not None:
        logits = y.model._seg_logits_buf
        if isinstance(logits, tuple):
            logits = logits[0]
        seg_heat = logits.detach().float().sigmoid()[0, 0].cpu().numpy()

        seg_prior = heatmap_to_bbox_prior(seg_heat, size=y.model.mask_size)
        y.predictor.external_mask = seg_prior
        r_seg_bbox = y.predict(image_path, save=False, verbose=False, conf=CONF, iou=IOU)
        y.predictor.external_mask = None

        panels.append(title(heatmap_overlay(img, seg_heat), "seg_pred"))
        panels.append(title(r_seg_bbox[0].plot(), f"seg_bbox  ({len(r_seg_bbox[0].boxes)} det)"))
    else:
        panels.append(title(blank_like(img), "seg_pred (n/a)"))
        panels.append(title(blank_like(img), "seg_bbox (n/a)"))

    # 5-6: GT mask overlay + GT-mask-guided detection (blank if no annotation).
    if has_anno:
        y.predictor.external_mask = load_mask_as_prior(mask_path)
        r_on = y.predict(image_path, save=False, verbose=False, conf=CONF, iou=IOU)
        y.predictor.external_mask = None

        panels.append(title(overlay_mask(img, mask_path), "gt mask overlay"))
        panels.append(title(r_on[0].plot(), f"mask_on   ({len(r_on[0].boxes)} det)"))
    else:
        panels.append(title(blank_like(img), "gt mask (n/a)"))
        panels.append(title(blank_like(img), "mask_on (n/a)"))

    out = concat_samh(panels, gap=12, gap_color=(255, 255, 255))

    save_path = Path(SAVE_PATH).resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(save_path), out)

    print(f"image : {Path(image_path).resolve()}")
    print(f"mask  : {Path(mask_path).resolve() if has_anno else None}")
    print(f"saved : {save_path}")


if __name__ == "__main__":
    main()
