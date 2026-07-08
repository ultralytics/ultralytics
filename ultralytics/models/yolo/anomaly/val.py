# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly validator — single-pass with optional heatmap prior."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.plotting import Annotator, colors


class YOLOAnomalyValidator(DetectionValidator):
    """Anomaly validator.

    Runs a single validation pass. If an AnomalyMemoryBank is configured and can be
    built from the dataset's train (normal) split, the pass uses the heatmap prior
    (extended IoU grid for coarse localization). Otherwise it falls back to a regular
    detection validation with a warning.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "detect"
        self.args.rect = False

    def init_metrics(self, model) -> None:
        """Set up heatmap flag and try to build the memory bank for heatmap mode."""
        super().init_metrics(model)
        self.model = model  # keep the inference model (AutoBackend in standalone val)
        self.v2_model = model.model if hasattr(model, "backend") else model  # unwrap AutoBackend if present
        # Coarse IoU grid .10:.50 (step .05) — anomaly localization cares about coarse overlap,
        # not tight boxes. Same grid with or without the prior, so both are directly comparable.
        # Columns: .10=0, .25=3, .50=8. mAP10-50 is the mean over the whole grid.
        self.iouv = torch.linspace(0.1, 0.5, 9)
        self.niou = self.iouv.numel()

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Post-process YOLO predictions and return output detections with proto.

        Args:
            preds (list[torch.Tensor]): Raw predictions from the model.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS.
        """
        preds = super().postprocess(preds[0])
        return preds

    def _anomaly_head(self):
        """Return the ``AnomalyDetect`` head of the raw YOLOA model."""
        try:
            return self.v2_model.model[-1]
        except Exception:
            return None

    def _render_gt_mask(self, img_path: str, ori_shape: tuple[int, int], mask_size: int) -> np.ndarray:
        """Render the ground-truth polygon mask and resize it to ``mask_size``."""
        txt_path = Path(img_path).with_suffix(".txt")
        h, w = ori_shape
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        if txt_path.exists():
            for line in Path(txt_path).read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) < 7:  # need at least class + 3 polygon points
                    continue
                vals = list(map(float, parts[1:]))
                xs = vals[0::2]
                ys = vals[1::2]
                pts = np.array([[int(x * w), int(y * h)] for x, y in zip(xs, ys)], dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

        imgsz = self.args.imgsz
        if isinstance(imgsz, (list, tuple)):
            imgsz = imgsz[0]
        lb = LetterBox(
            new_shape=(imgsz, imgsz),
            auto=False,
            scaleup=False,
            stride=int(self.stride),
            padding_value=0,
        )
        mask_lb = lb(image=mask)
        mask_rs = cv2.resize(mask_lb, (mask_size, mask_size), interpolation=cv2.INTER_NEAREST)
        return (mask_rs > 0).astype(np.float32)

    @staticmethod
    def _add_title(img: np.ndarray, text: str, bar_h: int = 32) -> np.ndarray:
        """Stack a left-aligned title bar above a BGR image."""
        bar = np.full((bar_h, img.shape[1], 3), 30, np.uint8)
        (_, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(
            bar, text, (8, (bar_h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA
        )
        return np.vstack([bar, img])

    @staticmethod
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

    def _draw_panel(
        self,
        img: np.ndarray,
        pred: dict[str, torch.Tensor],
        gt_boxes: torch.Tensor,
        title: str,
    ) -> np.ndarray:
        """Draw predictions with class/conf labels and GT boxes on a preprocessed image."""
        annotator = Annotator(img.copy(), line_width=1, font_size=0.35)
        # Predictions
        if len(pred.get("bboxes", [])):
            bboxes = pred["bboxes"]
            confs = pred.get("conf", None)
            clss = pred.get("cls", None)
            for j in range(len(bboxes)):
                box = bboxes[j].tolist()
                c = int(clss[j]) if clss is not None else 0
                label = f"{self.names.get(c, '')} {confs[j]:.2f}" if confs is not None else ""
                annotator.box_label(box, label, color=colors(c, True))
        # GT boxes in green
        if len(gt_boxes):
            for box in gt_boxes:
                annotator.box_label(box.tolist(), "GT", color=(0, 255, 0))
        return self._add_title(annotator.result(), f"{title} ({len(pred.get('bboxes', []))} det)")

    def _draw_heatmap_panel(self, img: np.ndarray, heatmap: torch.Tensor | None, title: str) -> np.ndarray:
        """Overlay the anomaly heatmap on the preprocessed image."""
        if heatmap is None:
            return self._add_title(img, title)
        hmap = heatmap.squeeze().cpu().numpy()
        h = cv2.resize(hmap.astype("float32"), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        cmap = cv2.applyColorMap((np.clip(h, 0, 1) * 255).astype("uint8"), cv2.COLORMAP_JET)
        panel = cv2.addWeighted(cmap, 0.45, img, 0.55, 0)
        return self._add_title(panel, title)

    def plot_predictions(
        self, batch: dict[str, any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """Plot a 1x4 anomaly comparison grid for the first few validation batches.

        The grid contains:
          - none-prior predictions + GT boxes
          - original image with the heatmap overlay
          - heatmap-prior predictions + GT boxes (reuses the provided ``preds``)
          - real-mask-as-prior predictions + GT boxes
        """
        if not preds:
            return

        head = self._anomaly_head()
        if head is None:
            # Fall back to the standard detection plot if the head is not a YOLOA head.
            return super().plot_predictions(batch, preds, ni, max_det)

        mask_size = head.heatmap_processor.mask_size
        imgs = batch["img"]
        b = imgs.shape[0]
        device = imgs.device

        # -- run the variants ---------------------------------------------------
        # Reuse the provided preds as the heatmap-prior boxes; run one forward pass
        # to grab the matching heatmap for the overlay panel.
        raw_heatmap = self.model(imgs, augment=False)
        # AnomalyDetect returns [(detections, heatmap), raw_preds]
        heatmaps = raw_heatmap[0][1] if isinstance(raw_heatmap, (list, tuple)) and len(raw_heatmap) else None

        # 1) none-prior: disable the memory bank temporarily
        mb = getattr(self.v2_model, "memory_bank", None)
        saved_building = getattr(mb, "building", None)
        if mb is not None:
            mb.building = True
        try:
            raw_none = self.model(imgs, augment=False)
            preds_none = self.postprocess(raw_none)
        finally:
            if mb is not None and saved_building is not None:
                mb.building = saved_building

        # 2) real-mask-as-prior
        prior_masks = []
        for i, path in enumerate(batch["im_file"]):
            m = self._render_gt_mask(path, tuple(batch["ori_shape"][i]), mask_size)
            prior_masks.append(torch.from_numpy(m))
        prior_mask = torch.stack(prior_masks).unsqueeze(1).to(device=device, dtype=torch.float32)
        raw_mask = self.model(imgs, augment=False, prior_mask=prior_mask)
        preds_mask = self.postprocess(raw_mask)

        # -- build per-image grids ----------------------------------------------
        h, w = imgs.shape[2], imgs.shape[3]
        for i in range(b):
            # Preprocessed image as BGR
            img = (imgs[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            # GT boxes in input coordinates (xyxy)
            idx = batch["batch_idx"] == i
            gt_norm = batch["bboxes"][idx]
            if len(gt_norm):
                gt_norm_t = torch.from_numpy(gt_norm) if isinstance(gt_norm, np.ndarray) else gt_norm
                gt_xyxy = ops.xywh2xyxy(gt_norm_t) * torch.tensor([w, h, w, h], device=gt_norm_t.device)
            else:
                gt_xyxy = torch.zeros((0, 4), device=imgs.device)

            hmap = heatmaps[i] if heatmaps is not None else None
            panels = [
                self._draw_panel(img, preds_none[i], gt_xyxy, "none prior + GT"),
                self._draw_heatmap_panel(img, hmap, "heatmap"),
                self._draw_panel(img, preds[i], gt_xyxy, "heatmap prior + GT"),
                self._draw_panel(img, preds_mask[i], gt_xyxy, "mask prior + GT"),
            ]
            grid = self._hstack(panels)
            fname = self.save_dir / f"val_batch{ni}_pred_{Path(batch['im_file'][i]).stem}.jpg"
            cv2.imwrite(str(fname), grid)

    def _ood_map_metrics(self) -> dict[str, float]:
        """mAP at IoU {0.10, 0.25, 0.50} and mAP10-50 (mean over the whole .10:.50 grid)."""
        box = self.metrics.box
        all_ap = getattr(box, "all_ap", [])
        out = {
            "mAP10": 0.0,
            "mAP25": 0.0,
            "mAP50": 0.0,
            "mAP10_50": 0.0,
            "P": float(box.mp),
            "R": float(box.mr),
        }
        if not len(all_ap):
            return out
        iouv = self.iouv.cpu().numpy()
        idx = {thr: int(np.where(np.isclose(iouv, thr))[0][0]) for thr in (0.10, 0.25, 0.50)}
        out["mAP10"] = float(all_ap[:, idx[0.10]].mean())
        out["mAP25"] = float(all_ap[:, idx[0.25]].mean())
        out["mAP50"] = float(all_ap[:, idx[0.50]].mean())
        out["mAP10_50"] = float(all_ap.mean())  # mean over the full .10:.50 grid
        return out

    def get_desc(self) -> str:
        """Return the column header with mAP10/mAP25 columns (always, for alignment)."""
        return ("%22s" + "%11s" * 8) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP10",
            "mAP25",
            "mAP50",
            "mAP10-50)",
        )

    def print_results(self) -> None:
        """Print the 'all' row with mAP10/mAP25/mAP50 and the mAP10:50 aggregate."""
        mm = self._ood_map_metrics()
        nt = int(self.metrics.nt_per_class.sum()) if len(self.metrics.nt_per_class) else 0
        pf = "%22s" + "%11i" * 2 + "%11.3g" * 6
        LOGGER.info(
            pf % ("all", self.seen, nt, mm["P"], mm["R"], mm["mAP10"], mm["mAP25"], mm["mAP50"], mm["mAP10_50"])
        )

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        if not self.training or not dist.is_initialized():
            return  # no DDP collectives when running standalone (e.g. OOD eval)
        super().gather_stats()
