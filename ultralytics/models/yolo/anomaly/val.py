# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""YOLO Anomaly validator — single-pass with optional heatmap prior."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
import yaml

from ultralytics.data.augment import LetterBox
from ultralytics.models.yolo.anomaly.predict import AnomalyPredictorHM
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.nn.modules import AnomalyDetect
from ultralytics.utils import LOGGER, ops, TryExcept
from ultralytics.utils.plotting import Annotator, colors


# def heatmap_to_boxes(
#     heatmap: "torch.Tensor",
#     thresh: float = 0.5,
#     max_det: int = 9,
#     min_area: int = 64,
# ) -> "torch.Tensor":
#     """Threshold a spatial heatmap and fit bounding boxes via connected components.
#
#     Args:
#         heatmap (Tensor): (H, W) float tensor, values in [0, 1].
#         thresh (float): Score threshold for foreground pixels.
#         max_det (int): Maximum number of boxes returned.
#         min_area (int): Minimum connected-component area (pixels) to keep.
#
#     Returns:
#         Tensor: Shape (N, 6) — ``[x1, y1, x2, y2, score, class_id=0]``,
#             sorted by score descending.  Returns empty (0, 6) when no
#             component passes the threshold or min_area filter.
#     """
#     import cv2
#     import numpy as np
#     import torch
#
#     h_np = heatmap.detach().cpu().float().numpy()
#     mask = (h_np >= thresh).astype(np.uint8)
#     if mask.sum() == 0:
#         return torch.zeros((0, 6), dtype=torch.float32)
#     num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
#     H, W = h_np.shape
#     boxes = []
#     for lbl in range(1, num):  # skip background label 0
#         x, y, w, h, area = stats[lbl]
#         if area < min_area:
#             continue
#         # Skip components that touch the image border (resize / padding artifacts).
#         if x == 0 or y == 0 or (x + w) >= W or (y + h) >= H:
#             continue
#         score = float(h_np[labels == lbl].mean())
#         boxes.append([float(x), float(y), float(x + w), float(y + h), score, 0.0])
#     if not boxes:
#         return torch.zeros((0, 6), dtype=torch.float32)
#     t = torch.tensor(boxes, dtype=torch.float32)
#     order = t[:, 4].argsort(descending=True)[:max_det]
#     return t[order]


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
        """Return the ``AnomalyDetect`` head of the raw YOLOA model, if present."""
        try:
            head = self.v2_model.model[-1]
            return head if isinstance(head, AnomalyDetect) else None
        except Exception:
            return None

    def _render_gt_mask(self, img_path: str, ori_shape: tuple[int, int], mask_size: int) -> np.ndarray:
        """Render the ground-truth polygon mask and resize it to ``mask_size``."""
        txt_path = Path(img_path).with_suffix(".txt")
        h, w = ori_shape
        mask = np.zeros((h, w), dtype=np.uint8)
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
        cv2.putText(bar, text, (8, (bar_h + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
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

    @TryExcept()
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

        # head = self._anomaly_head()
        # if head is None:
        #     # Fall back to the standard detection plot if the head is not a YOLOA head.
        #     return super().plot_predictions(batch, preds, ni, max_det)

        # mask_size = head.heatmap_processor.mask_size
        mask_size = 80  # hard-code to 80 now
        imgs = batch["img"]
        b = imgs.shape[0]
        device = imgs.device

        # -- run the variants ---------------------------------------------------
        # Reuse the provided preds as the heatmap-prior boxes; run one forward pass
        # to grab the matching heatmap for the overlay panel.
        raw_heatmap = self.model(imgs, augment=False)
        # AnomalyDetect returns [(detections, heatmap), raw_preds]
        heatmaps = raw_heatmap[0][1] if isinstance(raw_heatmap, (list, tuple)) and len(raw_heatmap) else None
        heatmaps_fusion = raw_heatmap[0][2] if isinstance(raw_heatmap, (list, tuple)) and len(raw_heatmap) else None
        if heatmaps_fusion.max() > 1.0:
            heatmaps_fusion = (heatmaps_fusion - heatmaps_fusion.min()) / heatmaps_fusion.max()  # normalize to [0, 1]

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
            hmap_fusion = heatmaps_fusion[i] if heatmaps_fusion is not None else None
            panels = [
                self._draw_panel(img, preds_none[i], gt_xyxy, "none prior + GT"),
                self._draw_heatmap_panel(img, hmap, "heatmap"),
                self._draw_heatmap_panel(img, hmap_fusion, "hmap_fusion"),
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


class YOLOAnomalyValidatorHM(YOLOAnomalyValidator):
    """YOLOA validator that scores heatmap-derived boxes instead of the detection head's NMS output.

    Validation counterpart of ``AnomalyPredictorHM``: ``postprocess`` thresholds the
    ``AnomalyDetect`` heatmap and fits connected-component boxes via the shared
    ``AnomalyPredictorHM._heatmap_to_boxes`` helper, so predict and val agree box-for-box.
    Falls back to the detection head when the model emits no heatmap.
    """

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Threshold the AnomalyDetect heatmap and fit connected-component boxes."""
        if not (isinstance(preds, (list, tuple)) and isinstance(preds[0], (list, tuple))):
            return super().postprocess(preds)  # no heatmap tuple — head boxes

        heatmap = preds[0][1]  # (B, 1, mH, mW)
        mh, mw = heatmap.shape[2], heatmap.shape[3]
        sx, sy = self.args.imgsz / mw, self.args.imgsz / mh  # heatmap px -> imgsz space
        outputs = []
        for i in range(heatmap.shape[0]):
            b = AnomalyPredictorHM._heatmap_to_boxes(heatmap[i, 0]).to(heatmap.device)  # (N, 6)
            if b.shape[0]:
                b[:, [0, 2]] *= sx  # x1, x2
                b[:, [1, 3]] *= sy  # y1, y2
            outputs.append({"bboxes": b[:, :4], "conf": b[:, 4], "cls": b[:, 5], "extra": b[:, 6:]})
        return outputs


class YOLOAnomalyCocoValidator(YOLOAnomalyValidator):
    """Anomaly validator that owns the full offline COCO-eval pipeline for MVTec.

    Two responsibilities:

    1. **Dump** — during ``model.val(..., save_json=True)`` it serializes predictions to
       ``save_dir/predictions.json`` with an integer ``image_id`` = the 0-based index into
       ``val.txt`` order. Native ``DetectionValidator.pred_to_json`` keys on the file-name
       stem, which for MVTec is a non-numeric string (``0_000``) that faster-coco-eval
       rejects and that collides across categories in pooled eval; the integer scheme fixes
       both and matches :meth:`build_coco_gt`.
    2. **Evaluate** — :meth:`evaluate_run` runs multi-confidence COCO eval offline over the
       dumped predictions (per-category + pooled, loose ``.10:.50`` and standard ``.50:.95``
       IoU grids). Because the dump keeps every box (``conf=DUMP_CONF``), each conf floor is
       obtained by ``loadRes(preds, min_score=conf)`` — a single inference pass covers the
       whole sweep, no re-inference and no temp files.

    Driver usage::

        for cat in cats:
            model.set_memory(...)
            model.val(validator=YOLOAnomalyCocoValidator, conf=YOLOAnomalyCocoValidator.DUMP_CONF,
                      iou=YOLOAnomalyCocoValidator.DUMP_IOU, save_json=True,
                      project=str(out_root), name=cat, exist_ok=True, ...)
        YOLOAnomalyCocoValidator.evaluate_run(out_root, cat_to_yaml, conf_sweep, cat_arg)
    """

    # Dump-time thresholds: keep nearly every detection so any conf floor can be applied offline.
    DUMP_CONF = 0.001
    DUMP_IOU = 0.2
    # COCO IoU grids. Loose linspace(.10,.50,9): .10=idx0 .25=idx3 .50=idx8. Standard = canonical COCO.
    IOU_LOOSE = np.linspace(0.1, 0.5, 9)
    IOU_STANDARD = np.linspace(0.5, 0.95, 10)

    # -- dump-pass lifecycle ---------------------------------------------------
    def init_metrics(self, model) -> None:
        """Build the val.txt-ordered {abs_path: int id} map used for stable image ids."""
        super().init_metrics(model)
        val_file = Path(self.data["path"]) / self.data["val"]
        paths = [line.strip() for line in val_file.read_text().splitlines() if line.strip()]
        self._imgid_map = {Path(p).as_posix(): i for i, p in enumerate(paths)}

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, any]) -> None:
        """Serialize one image's predictions with an integer, val.txt-ordered image_id."""
        key = Path(pbatch["im_file"]).as_posix()
        image_id = self._imgid_map.get(key)
        if image_id is None:  # val loader order diverged from val.txt — id scheme would break
            raise KeyError(f"{key} not in val.txt id map (order mismatch between loader and val.txt)")
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh (center)
        box[:, :2] -= box[:, 2:] / 2  # center -> top-left corner
        for b, s in zip(box.tolist(), predn["conf"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": 0,  # single-class — matches build_coco_gt's cls_id=0
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def eval_json(self, stats: dict[str, any]) -> dict[str, any]:
        """Skip the native is_coco/is_lvis path; predictions.json is dumped by the base validator."""
        return stats

    # -- offline COCO eval (moved from run_yoloa; multi-conf, in-memory) --------
    @staticmethod
    def build_coco_gt(yaml_path, rebuild: bool = False) -> dict:
        """Convert YOLO-seg val labels to a COCO ground-truth dict (bbox from polygon).

        ``image_id`` = 0-based index into ``val.txt`` order (integer), matching
        :meth:`pred_to_json`. GT is static (labels only), so the result is cached to
        ``coco_gt.json`` next to the val list and reused unless the cache is missing/stale
        (older than ``val.txt``) or ``rebuild`` is set.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        root = Path(data["path"])
        val_file = root / data["val"]

        cache_path = val_file.with_name("coco_gt.json")
        if not rebuild and cache_path.is_file() and cache_path.stat().st_mtime >= val_file.stat().st_mtime:
            with open(cache_path) as f:
                LOGGER.info(f"  loaded cached GT -> {cache_path}")
                return json.load(f)

        with open(val_file) as f:
            img_paths = [line.strip() for line in f if line.strip()]

        images, annotations = [], []
        ann_id = 0
        for i, img_path_str in enumerate(img_paths):
            image_id = i  # integer — matches pred_to_json list-index
            img = cv2.imread(img_path_str)
            if img is None:
                continue
            h, w = img.shape[:2]
            images.append({"id": image_id, "file_name": img_path_str, "width": w, "height": h})

            label_path = Path(img_path_str).with_suffix(".txt")
            if not label_path.is_file():
                continue  # negative image — no GT annotation needed
            with open(label_path) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) < 5:  # cls_id + ≥2 polygon points (4 coords)
                        continue
                    cls_id = 0  # force single-class: MVTec binary labels still carry original ids
                    points = list(map(float, parts[1:]))
                    xs = [points[j] * w for j in range(0, len(points), 2)]
                    ys = [points[j] * h for j in range(1, len(points), 2)]
                    x_min, y_min = min(xs), min(ys)
                    x_max, y_max = max(xs), max(ys)
                    bw, bh = x_max - x_min, y_max - y_min
                    annotations.append(
                        {
                            "id": ann_id,
                            "image_id": image_id,
                            "category_id": cls_id,
                            "bbox": [round(x_min, 1), round(y_min, 1), round(bw, 1), round(bh, 1)],
                            "area": round(bw * bh, 1),
                            "iscrowd": 0,
                        }
                    )
                    ann_id += 1

        names = data.get("names", ["anomaly"])
        if isinstance(names, list):
            categories = [{"id": i, "name": n} for i, n in enumerate(names)]
        else:
            categories = [{"id": i, "name": n} for i, n in names.items()]

        gt = {"info": {}, "licenses": [], "images": images, "annotations": annotations, "categories": categories}
        with open(cache_path, "w") as f:
            json.dump(gt, f)
        LOGGER.info(f"  cached GT ({len(images)} imgs, {len(annotations)} anns) -> {cache_path}")
        return gt

    @staticmethod
    def _ap_at_iou(precision, iou_thrs, target: float, tol: float = 1e-3) -> float:
        """Mean AP at a single IoU threshold from the COCO precision tensor.

        ``precision`` has shape [T, R, K, A, M]; slice IoU index ``t`` at area=all (0) and
        maxDet=100 (-1) and average the valid (>-1) recall points. Returns NaN if ``target``
        is absent from ``iou_thrs`` (e.g. .10/.25 in the standard .50:.95 grid).
        """
        matches = np.where(np.abs(iou_thrs - target) < tol)[0]
        if precision is None or matches.size == 0:
            return float("nan")
        p = precision[matches[0], :, :, 0, -1]
        p = p[p > -1]
        return round(float(p.mean()), 4) if p.size else 0.0

    @classmethod
    def _coco_ap(cls, anno, preds: list[dict], iou_thrs, min_score: float = 0.0) -> dict:
        """One COCOeval over a prebuilt GT ``anno`` and score-filtered ``preds`` (in-memory)."""
        from faster_coco_eval import COCOeval_faster

        stat_keys = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100"]
        zero = {k: 0.0 for k in stat_keys}
        zero.update({"mAP10": float("nan"), "mAP25": float("nan"), "mAP50": float("nan")})
        if not preds or not any(p["score"] >= min_score for p in preds):
            return zero

        pred = anno.loadRes(preds, min_score=min_score)  # faster-coco-eval filters by score
        val = COCOeval_faster(anno, pred, iouType="bbox", print_function=LOGGER.info)
        val.params.iouThrs = iou_thrs
        val.evaluate()
        val.accumulate()
        val.summarize()

        precision = val.eval["precision"] if getattr(val, "eval", None) else None
        d = val.stats_as_dict  # faster-coco-eval: AR_all=1, AR_second=10, AR_third=100
        return {
            "AP": round(float(d.get("AP_all", 0)), 4),
            "AP50": round(float(d.get("AP_50", 0)), 4),
            "AP75": round(float(d.get("AP_75", 0)), 4),
            "AP_small": round(float(d.get("AP_small", 0)), 4),
            "AP_medium": round(float(d.get("AP_medium", 0)), 4),
            "AP_large": round(float(d.get("AP_large", 0)), 4),
            "AR1": round(float(d.get("AR_all", 0)), 4),
            "AR10": round(float(d.get("AR_second", 0)), 4),
            "AR100": round(float(d.get("AR_third", 0)), 4),
            "mAP10": cls._ap_at_iou(precision, iou_thrs, 0.10),
            "mAP25": cls._ap_at_iou(precision, iou_thrs, 0.25),
            "mAP50": cls._ap_at_iou(precision, iou_thrs, 0.50),
        }

    @classmethod
    def evaluate_run(cls, out_root, cat_to_yaml: dict, cat_to_json: dict, conf_sweep, cat_arg: str, rebuild_gt: bool = False) -> None:
        """Offline multi-conf COCO eval over dumped predictions.json — per-cat + POOLED, one CSV.

        Args:
            out_root: run output root; the CSV is written to ``out_root/coco_<cat_arg>.csv``.
            cat_to_yaml: ordered {category: data-yaml Path} produced by the driver.
            cat_to_json: {category: predictions.json Path} — the exact file each dump pass wrote.
            conf_sweep: conf floors to evaluate (each applied via ``loadRes(min_score=conf)``).
            cat_arg: original ``--cat`` value; names the CSV ``coco_<cat_arg>.csv``.
            rebuild_gt: force rebuild of the cached ``coco_gt.json``.
        """
        from faster_coco_eval import COCO

        out_root = Path(out_root)
        regimes = [("loose(.10:.50)", cls.IOU_LOOSE), ("standard(.50:.95)", cls.IOU_STANDARD)]
        rows = []
        all_gt = {"info": {}, "licenses": [], "images": [], "annotations": [], "categories": [{"id": 0, "name": "anomaly"}]}
        all_preds = []
        ann_offset = 0  # keep annotation ids unique across cats in pooled GT

        for cat, yaml_path in cat_to_yaml.items():
            pred_json = cat_to_json.get(cat)
            if pred_json is None or not Path(pred_json).is_file():
                LOGGER.warning(f"[{cat}] predictions.json missing — skip COCO eval")
                continue
            with open(pred_json) as f:
                preds = json.load(f)

            gt = cls.build_coco_gt(yaml_path, rebuild=rebuild_gt)
            anno = COCO(gt)  # in-memory GT, built once per cat and reused across confs/regimes
            for conf in conf_sweep:
                for regime, iou_thrs in regimes:
                    ap = cls._coco_ap(anno, preds, iou_thrs, min_score=conf)
                    rows.append({"category": cat, "conf": conf, "regime": regime, **ap})
                    print(
                        f"  [{cat}] conf={conf:.3f} {regime}: AP={ap['AP']:.4f} AP50={ap['AP50']:.4f} "
                        f"mAP10={ap['mAP10']:.4f} mAP25={ap['mAP25']:.4f} mAP50={ap['mAP50']:.4f}",
                        flush=True,
                    )

            # accumulate for pooled (offset image_ids to keep them unique across cats)
            img_offset = len(all_gt["images"])
            for img in gt["images"]:
                ic = dict(img)
                ic["id"] = img["id"] + img_offset
                all_gt["images"].append(ic)
            for ann in gt["annotations"]:
                ac = dict(ann)
                ac["id"] += ann_offset
                ac["image_id"] = ann["image_id"] + img_offset
                all_gt["annotations"].append(ac)
            ann_offset = len(all_gt["annotations"])
            for p in preds:
                pc = dict(p)
                pc["image_id"] = p["image_id"] + img_offset
                all_preds.append(pc)

        # pooled across all categories
        if all_gt["images"]:
            print(
                f"\n  POOLED: {len(all_gt['images'])} images, {len(all_gt['annotations'])} GT, {len(all_preds)} preds",
                flush=True,
            )
            anno = COCO(all_gt)
            for conf in conf_sweep:
                for regime, iou_thrs in regimes:
                    ap = cls._coco_ap(anno, all_preds, iou_thrs, min_score=conf)
                    rows.append({"category": "POOLED", "conf": conf, "regime": regime, **ap})
                    print(
                        f"  [POOLED] conf={conf:.3f} {regime}: AP={ap['AP']:.4f} AP50={ap['AP50']:.4f} "
                        f"mAP10={ap['mAP10']:.4f} mAP25={ap['mAP25']:.4f} mAP50={ap['mAP50']:.4f}",
                        flush=True,
                    )

        if rows:
            out_csv = out_root / f"coco_{cat_arg}.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "category", "conf", "regime", "AP", "AP50", "AP75",
                "mAP10", "mAP25", "mAP50",
                "AP_small", "AP_medium", "AP_large",
                "AR1", "AR10", "AR100",
            ]
            with open(out_csv, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames)
                w.writeheader()
                w.writerows(rows)
            print(f"\nCOCO CSV -> {out_csv}", flush=True)
