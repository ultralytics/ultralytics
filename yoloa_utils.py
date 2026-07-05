"""YOLOA utilities — categories, paths, and visualization helpers."""

import hashlib
import random
from pathlib import Path

import cv2
import numpy as np

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


# -- Heatmap / mask overlays --------------------------------------------------

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


def _draw_gt_boxes(img: np.ndarray, txt_path: str | None, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """Return a copy of ``img`` with green bounding boxes from a YOLO label file.

    Supports both YOLO-bbox (class x y w h) and YOLO-seg polygon (class x1 y1 ...)
    formats. Missing/empty label files are a no-op.
    """
    if txt_path is None:
        return img
    try:
        with open(txt_path, "r") as f:
            lines = f.read().strip().splitlines()
    except (FileNotFoundError, OSError):
        return img
    if not lines:
        return img

    h, w = img.shape[:2]
    out = img.copy()
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        vals = list(map(float, parts[1:]))
        if len(vals) == 4:
            # YOLO bbox: x, y, bw, bh (normalized)
            x, y, bw, bh = vals
            x1 = int((x - bw / 2) * w)
            y1 = int((y - bh / 2) * h)
            x2 = int((x + bw / 2) * w)
            y2 = int((y + bh / 2) * h)
        elif len(vals) >= 6 and len(vals) % 2 == 0:
            # YOLO-seg polygon
            xs = vals[0::2]
            ys = vals[1::2]
            x1, x2 = int(min(xs) * w), int(max(xs) * w)
            y1, y2 = int(min(ys) * h), int(max(ys) * h)
        else:
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


def save_simple_grid(
    original,
    none_pred,
    heat_pred,
    heat_hmap,
    gt_txt_path,
    out_path,
    n_none=0,
    n_heat=0,
):
    """Build and save a 1x3 comparison grid.

    Layout:
      - none-prior prediction with GT boxes
      - original image with the heatmap overlay
      - heatmap-prior prediction with GT boxes
    """
    hmap_title = "heatmap prior"
    if heat_hmap is not None:
        hmap_title += f"  [max={heat_hmap.max():.3f} min={heat_hmap.min():.3f}]"
    grid = _hstack(
        [
            _add_title(_draw_gt_boxes(none_pred, gt_txt_path), f"none prior + GT ({n_none} det)"),
            _add_title(_heatmap_panel(original, heat_hmap), hmap_title),
            _add_title(_draw_gt_boxes(heat_pred, gt_txt_path), f"heatmap prior + GT ({n_heat} det)"),
        ]
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), grid)
    return out_path
