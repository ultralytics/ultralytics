"""YOLOA utilities — categories, YAML loading, visualization helpers."""

import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.utils import YAML

# -- Category constants -------------------------------------------------------

MVTEC_OBJECT = [
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]
MVTEC_TEXTURE = ["carpet", "grid", "leather", "tile", "wood"]
MVTEC_RANDOM = ["bottle", "cable", "capsule", "carpet", "grid"]

CAT_GROUPS = {"object": MVTEC_OBJECT, "texture": MVTEC_TEXTURE, "random5": MVTEC_RANDOM}
VAL_METRICS = ("image_auroc", "pixel_auroc", "mAP10", "mAP25", "mAP50", "mAP10_50", "P", "R")

# YAML keys -> FeatureDiscriminatorScorer kwargs
SCORER_YAML_KEYS = {
    "scorer_noise_std": "noise_std",
    "scorer_steps": "steps",
    "scorer_hidden": "hidden",
    "scorer_n_noise": "n_noise",
    "scorer_batch": "batch",
    "scorer_lr": "lr",
    "scorer_noise_mode": "noise_mode",
}


# -- Path / data helpers ------------------------------------------------------


def collect_test_images(test_root: Path, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Return up to ``n`` (path, defect_type) pairs sampled across a category's test subdirs."""
    pairs = [
        (str(p), sub.name) for sub in sorted(test_root.iterdir()) if sub.is_dir() for p in sorted(sub.glob("*.png"))
    ]
    random.Random(seed).shuffle(pairs)
    return pairs[:n] if n and n > 0 else pairs


def good_dir(root: Path, cat: str) -> Path:
    """Resolve a category's normal-images dir for fitting (train/good, else train)."""
    d = root / cat / "train" / "good"
    return d if d.is_dir() else root / cat / "train"


def model_id_from_ckpt(ckpt: str) -> str:
    """Model id from a ckpt path: readable label + short hash of the full resolved path.

    The label is `<run>__best` for `<run>/weights/best.pt`, else the file stem. The 6-char
    path hash makes the id globally unique so different checkpoints that share a run name and
    weight stem (e.g. the same run under ultra6 `runs/` and the pulled mirror) never collide on
    the shared bank cache dir, while the label keeps output dirs browsable.
    """
    p = Path(ckpt).resolve()
    label = f"{p.parents[1].name}__{p.stem}" if p.parent.name == "weights" else p.stem
    return f"{label}__{hashlib.md5(str(p).encode()).hexdigest()[:6]}"


# -- Fit YAML resolution ------------------------------------------------------

def resolve_infer(yaml: dict) -> dict:
    """Extract prior-shaping inference knobs from YAML."""
    infer = {}
    if yaml.get("heat_edge"):
        infer["heat_edge"] = True
        infer["heat_edge_sigma"] = yaml.get("heat_edge_sigma", 1.0)
    if yaml.get("heat_norm"):
        infer["heat_norm"] = yaml["heat_norm"]
    return infer


# -- Heatmap / mask overlays --------------------------------------------------


def save_heatmap(model, img_path: str, out_path: Path) -> None:
    """Save the model's last prior heatmap as a JET overlay on the original image."""
    hm = getattr(model, "_last_heatmap", None)
    if hm is None:
        return
    h = hm.detach().cpu().numpy().squeeze()
    h = (h - h.min()) / (np.ptp(h) + 1e-9)
    orig = cv2.imread(img_path)
    color = cv2.resize(cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET), (orig.shape[1], orig.shape[0]))
    cv2.imwrite(str(out_path), cv2.addWeighted(orig, 0.55, color, 0.45, 0))


def load_mask_tensor(mask, imgsz: int):
    """Load a GT mask as a (1, 1, imgsz, imgsz) float tensor, or None. Accepts path or numpy array."""
    if mask is None:
        return None
    if isinstance(mask, (str, Path)):
        m = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
    else:
        m = mask
    m = cv2.resize(m, (imgsz, imgsz), interpolation=cv2.INTER_NEAREST).astype(np.float32) / 255.0
    return torch.from_numpy(m).unsqueeze(0).unsqueeze(0)


def run_prior_viz(m, img, prior, imgsz, conf, iou, device, external_mask=None, **kw):
    """Predict with one prior; return (pred_bgr, n_det, heatmap_np)."""
    res = m.predict(
        img,
        prior=prior,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        external_mask=external_mask,
        verbose=False,
        **kw,
    )[0]
    n = 0 if res.boxes is None else res.boxes.shape[0]
    hm = getattr(m.model, "_last_heatmap", None)
    hm_np = hm.detach().cpu().numpy().squeeze() if hm is not None else None
    return res.plot(), n, hm_np


# -- Compare-grid (visualize mode) ---------------------------------------------

def txt_to_mask(txt_path: str, h: int, w: int) -> np.ndarray | None:
    """Render YOLO-seg txt polygon labels into a binary mask (h, w) uint8 0/255.

    Returns None if the txt file is missing or contains no valid polygons (e.g. good images).
    """
    try:
        with open(txt_path, "r") as f:
            lines = f.read().strip().splitlines()
    except (FileNotFoundError, OSError):
        return None
    if not lines:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue
        pts = [(int(coords[i] * w), int(coords[i + 1] * h)) for i in range(0, len(coords), 2)]
        pts_array = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts_array], 255)
    return mask


def _add_title(img: np.ndarray, text: str, bar_h: int = 48) -> np.ndarray:
    """Stack a left-aligned title bar above a BGR image."""
    bar = np.full((bar_h, img.shape[1], 3), 30, np.uint8)
    (_, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.putText(bar, text, (8, (bar_h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return np.vstack([bar, img])


def _heatmap_panel(img: np.ndarray, hmap: np.ndarray | None) -> np.ndarray:
    """JET heatmap overlay (BGR in/out). Min/max are rendered in the title bar."""
    if hmap is None:
        return img
    h = cv2.resize(hmap.astype("float32"), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    cmap = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype("uint8"), cv2.COLORMAP_JET)
    return cv2.addWeighted(cmap, 0.45, img, 0.55, 0)


def _mask_panel(img: np.ndarray, gt_mask: np.ndarray | str | None) -> np.ndarray:
    """Red-tint overlay with defect-pixel percentage (BGR in/out). Accepts array or file path."""
    if gt_mask is None:
        return img
    if isinstance(gt_mask, (str, Path)):
        m = cv2.imread(str(gt_mask), cv2.IMREAD_GRAYSCALE)
    else:
        m = gt_mask
    if m is None:
        return img
    m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    panel = img.copy()
    sel = m > 0
    panel[sel] = (0.45 * np.array([0, 0, 255], dtype=np.float32) + 0.55 * panel[sel].astype(np.float32)).astype(
        np.uint8
    )
    pct = (m > 0).mean() * 100
    label = f"defect={pct:.2f}%"
    cv2.putText(panel, label, (8, panel.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def _hstack(images: list[np.ndarray], gap: int = 8) -> np.ndarray:
    """Horizontal concat of same-height BGR images with white gaps."""
    h = max(im.shape[0] for im in images)
    w = sum(im.shape[1] for im in images) + gap * (len(images) - 1)
    out = np.full((h, w, 3), 255, dtype=np.uint8)
    x = 0
    for im in images:
        y = (h - im.shape[0]) // 2
        out[y : y + im.shape[0], x : x + im.shape[1]] = im
        x += im.shape[1] + gap
    return out


def save_compare_grid(
    *,
    original,
    none_pred,
    heat_heat,
    heat_pred,
    mask_img,
    mask_pred,
    out_path,
    n_none=0,
    n_heat=0,
    n_mask=0,
    original_title="original",
):
    """Build and save the 3x2 comparison grid (segment prior removed)."""
    mb_hmap_title = "mb heatmap"
    if heat_heat is not None:
        mb_hmap_title += f"  [max={heat_heat.max():.3f} min={heat_heat.min():.3f}]"
    row1 = _hstack(
        [
            _add_title(original, original_title),
            _add_title(none_pred, f"None Prior ({n_none} det)"),
            _add_title(_heatmap_panel(original, heat_heat), mb_hmap_title),
            _add_title(heat_pred, f"heatmap prior ({n_heat} det)"),
        ]
    )
    row2 = _hstack(
        [
            _add_title(_mask_panel(original, mask_img), "GT mask"),
            _add_title(mask_pred, f"mask prior ({n_mask} det)"),
        ]
    )
    # Pad row2 to match row1 width
    if row2.shape[1] < row1.shape[1]:
        pad = np.full((row2.shape[0], row1.shape[1] - row2.shape[1], 3), 255, dtype=np.uint8)
        row2 = np.hstack([row2, pad])
    grid = np.vstack([row1, np.full((8, row1.shape[1], 3), 255, dtype=np.uint8), row2])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path
