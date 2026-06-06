"""Pure-visualisation 4×2 comparison grid for YOLOA v2 prior-mode results.

Takes pre-computed images/heatmaps — no model or inference dependency.

Layout (2 rows × 4 cols):
  Row 1: original | None Prior pred | seg heatmap | seg prior pred
  Row 2: mb heatmap | heatmap prior pred | GT mask | mask prior pred

Usage:
    from compare_grid import CompareGrid
    cg = CompareGrid()
    cg.save(original=img_rgb, none_pred=..., seg_heat=..., seg_pred=...,
            heat_heat=..., heat_pred=..., mask_img=..., mask_pred=...,
            out_path="output.jpg")
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ultra_ext.im import concat_samh


class CompareGrid:
    """Arrange 8 panels into a 4×2 comparison grid (pure layout, no model)."""

    def __init__(self, bar_h: int = 64, gap: int = 12,
                 scale: float = 1.4, thickness: int = 3):
        self.bar_h = bar_h
        self.gap = gap
        self.scale = scale
        self.thickness = thickness

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(self, *, original: np.ndarray,
             none_pred: np.ndarray,
             seg_heat: np.ndarray,
             seg_pred: np.ndarray,
             heat_heat: np.ndarray,
             heat_pred: np.ndarray,
             mask_img: np.ndarray,
             mask_pred: np.ndarray,
             out_path: str | Path,
             n_none: int = 0, n_seg: int = 0,
             n_heat: int = 0, n_mask: int = 0) -> Path:
        """Build and write the 4×2 grid.  All images must be RGB uint8."""

        row1 = [
            self._title(original, "original"),
            self._title(none_pred, f"None Prior ({n_none} det)"),
            self._title(seg_heat, "seg heatmap"),
            self._title(seg_pred, f"segment prior ({n_seg} det)"),
        ]
        row2 = [
            self._title(heat_heat, "mb heatmap"),
            self._title(heat_pred, f"heatmap prior ({n_heat} det)"),
            self._title(mask_img, "GT mask"),
            self._title(mask_pred, f"mask prior ({n_mask} det)"),
        ]

        r1 = concat_samh(row1, gap=self.gap, gap_color=(255, 255, 255), cols=4)
        r2 = concat_samh(row2, gap=self.gap, gap_color=(255, 255, 255), cols=4)
        out = np.vstack([r1, np.full((self.gap, r1.shape[1], 3), 255, dtype=np.uint8), r2])

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        return out_path

    # ------------------------------------------------------------------
    # Panel factories (can be called independently)
    # ------------------------------------------------------------------

    @staticmethod
    def heatmap_overlay(img_rgb: np.ndarray, hmap: np.ndarray | None,
                        alpha: float = 0.45) -> np.ndarray:
        """JET-colormap overlay of a [0,1] heatmap onto an RGB image."""
        if hmap is None:
            return img_rgb
        h = cv2.resize(hmap.astype("float32"), (img_rgb.shape[1], img_rgb.shape[0]),
                       interpolation=cv2.INTER_LINEAR)
        h = np.clip(h, 0.0, 1.0)
        cmap = cv2.applyColorMap((h * 255).astype("uint8"), cv2.COLORMAP_JET)
        cmap_rgb = cv2.cvtColor(cmap, cv2.COLOR_BGR2RGB)
        return (alpha * cmap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)

    @staticmethod
    def heatmap_panel(img_rgb: np.ndarray, hmap: np.ndarray | None,
                      alpha: float = 0.45) -> np.ndarray:
        """Heatmap overlay with max/min annotation burned into the image."""
        panel = CompareGrid.heatmap_overlay(img_rgb, hmap, alpha)
        if hmap is not None:
            label = f"max={hmap.max():.3f}  min={hmap.min():.3f}"
            cv2.putText(panel, label, (8, panel.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        return panel

    @staticmethod
    def mask_overlay(img_rgb: np.ndarray, mask_path: str) -> np.ndarray:
        """Red-tint RGB image where mask is non-zero."""
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return img_rgb
        m = cv2.resize(m, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
        out = img_rgb.copy()
        sel = m > 0
        out[sel] = (0.45 * np.array([255, 0, 0], dtype=np.float32)
                    + 0.55 * out[sel].astype(np.float32)).astype(np.uint8)
        return out

    @staticmethod
    def mask_panel(img_rgb: np.ndarray, mask_path: str | None) -> np.ndarray:
        """Red-tint overlay with defect-pixel percentage annotation."""
        if mask_path is None:
            return img_rgb
        panel = CompareGrid.mask_overlay(img_rgb, mask_path)
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is not None:
            pct = (m > 0).mean() * 100
            label = f"defect={pct:.2f}%"
            cv2.putText(panel, label, (8, panel.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv2.LINE_AA)
        return panel

    @staticmethod
    def find_mask(img_path: str) -> str | None:
        """Find GT annotation mask for a MVTec test image.

        Searches both MVTec-YOLO (flat) and original mvtec_anomaly_detection
        (ground_truth/{defect}/) layouts.
        """
        ip = Path(img_path)
        defect_type = ip.parent.name        # e.g. "poke"
        cat_root = ip.parent.parent.parent  # test/{defect} -> test -> {category}

        # 1) Inside the YOLO-format category root
        for mask_dir in ["ground_truth", "mask"]:
            for suffix in ["_mask.png", ".png"]:
                mp = cat_root / mask_dir / defect_type / f"{ip.stem}{suffix}"
                if mp.is_file():
                    return str(mp)

        # 2) Original MVTec layout: {root}/mvtec_anomaly_detection/{cat}/ground_truth/{defect}/
        # YOLO filenames are "{idx}_{orig_stem}.png"; masks are "{orig_stem}_mask.png".
        category = cat_root.name
        parts = ip.stem.split("_", 1)
        orig_stem = parts[1] if len(parts) > 1 else ip.stem
        mvtec_root = cat_root.parent.parent  # MVTec-YOLO -> MVTEC -> MVTEC root
        for orig_dir in ["mvtec_anomaly_detection", "MVTec-FS"]:
            orig_masks = mvtec_root / orig_dir / category / "ground_truth" / defect_type
            for stem in (orig_stem, ip.stem):  # try both original and YOLO stem
                mp = orig_masks / f"{stem}_mask.png"
                if mp.is_file():
                    return str(mp)
        return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _title(self, img_rgb: np.ndarray, text: str) -> np.ndarray:
        bar = np.full((self.bar_h, img_rgb.shape[1], 3), 30, np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                        self.scale, self.thickness)
        x = max(8, (bar.shape[1] - tw) // 2)
        y = (self.bar_h + th) // 2
        cv2.putText(bar, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.scale, (255, 255, 255), self.thickness, cv2.LINE_AA)
        return np.vstack([bar, img_rgb])
