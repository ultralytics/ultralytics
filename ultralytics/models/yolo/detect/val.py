# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.data.utils import get_split_fraction
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, nms, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (list[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        jdict (list[dict[str, Any]]): List for storing JSON detection results.
        stats (dict[str, list[torch.Tensor]]): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks: dict | None = None) -> None:
        """Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (dict[str, Any], optional): Arguments for the validator.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        conf = args.get("conf") if isinstance(args, dict) else getattr(args, "conf", None)
        self.confusion_matrix_conf = 0.25 if conf is None else conf
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

    @staticmethod
    def _check_max_det(args, datasets: dict[str, torch.utils.data.Dataset]) -> None:
        """Warn when dataset object counts exceed max_det and raise the default limit to the observed maximum."""
        maxima = {
            split: max(
                (
                    len(label["cls"])
                    for subset in getattr(dataset, "datasets", [dataset])
                    if hasattr(subset, "labels")
                    for label in subset.labels
                    if isinstance(label, dict) and getattr(label.get("cls"), "ndim", 0) > 0
                ),
                default=0,
            )
            for split, dataset in datasets.items()
        }
        observed = max(maxima.values())
        if observed <= args.max_det:
            return

        split_counts = ", ".join(f"{split}={count}" for split, count in maxima.items())
        message = (
            f"Dataset images contain up to {observed} objects ({split_counts}), but max_det={args.max_det}. "
            "This mismatch can cap recall and produce invalid validation metrics."
            " Raising it may increase validation cost but cannot increase model or export capacity, which may cap recall."
        )
        if args.max_det == DEFAULT_CFG.max_det:
            args.max_det = observed
            message += f" Setting max_det={observed} to match the observed maximum."
        else:
            message += f" Keeping the user-specified max_det={args.max_det}."
        if RANK in {-1, 0}:
            LOGGER.warning(message)

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Preprocess batch of images for YOLO validation.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.

        Returns:
            (dict[str, Any]): Preprocessed batch.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type not in {"cpu", "mps"})
        batch["img"] = (batch["img"].half() if self.args.quantize == 16 else batch["img"].float()) / 255
        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        if not self.training:
            self._check_max_det(self.args, {self.args.split or "val": self.dataloader.dataset})
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith((f"{os.sep}val2017.txt", f"{os.sep}test-dev2017.txt")))
        )
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        native_model = model.model if getattr(model, "format", None) == "pt" else model
        if self.end2end and hasattr(native_model, "set_head_attr"):
            native_model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
        self.seen = 0
        self.jdict = []
        self.is_custom_json = self.args.save_json and self.args.task == "detect" and not (self.is_coco or self.is_lvis)
        self.gdict = getattr(self, "gdict", None) if self.is_custom_json else None
        self.build_gdict = self.is_custom_json and self.gdict is None
        self.eval_ids = list(self.dataloader.sampler) if self.is_custom_json else None
        self.pred_counts = []
        if self.build_gdict:
            self.gdict = {"images": [], "annotations": [], "categories": [{"id": x} for x in self.class_map]}
        self.metrics.names = model.names
        self.metrics.clear_stats()
        self.metrics.clear_image_metrics()
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor | list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor | list[torch.Tensor]): Raw predictions from the model, or (inference, loss) outputs.

        Returns:
            (list[dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains 'bboxes', 'conf',
                'cls', and 'extra' tensors.
        """
        if self.device.type == "mps":  # MPS: variable-shape NMS ops pollute the graph cache, run on CPU instead
            preds = (preds[0] if isinstance(preds, (list, tuple)) else preds).float().cpu()
        outputs = nms.non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            nc=0 if self.args.task == "detect" else self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )
        return [{"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]} for x in outputs]

    def _prepare_batch(self, si: int, batch: dict[str, Any]) -> dict[str, Any]:
        """Prepare a batch of images and annotations for validation.

        Args:
            si (int): Sample index within the batch.
            batch (dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if cls.shape[0]:
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=bbox.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepare predictions for evaluation against ground truth.

        Args:
            pred (dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Update metrics with new predictions and ground truth.

        Args:
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            batch (dict[str, Any]): Batch data containing ground truth.
        """
        if self.device.type == "mps":  # postprocess runs NMS on CPU for MPS, move the batch there to match
            batch.update({k: v.cpu() for k, v in batch.items() if torch.is_tensor(v)})
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            cls = pbatch["cls"].cpu().numpy()
            im_idx = self.eval_ids[self.seen - 1] if self.is_custom_json else None
            if self.build_gdict:
                boxes = ops.xyxy2ltwh(
                    ops.scale_boxes(pbatch["imgsz"], pbatch["bboxes"].clone(), pbatch["ori_shape"], pbatch["ratio_pad"])
                ).tolist()
                self.gdict["images"].append({"id": im_idx})
                self.gdict["annotations"].extend(
                    {
                        "id": (im_idx << 32 | i) + 1,
                        "image_id": im_idx,
                        "category_id": self.class_map[int(c)],
                        "bbox": b,
                        "area": b[2] * b[3],
                        "iscrowd": 0,
                    }
                    for i, (b, c) in enumerate(zip(boxes, cls))
                )
            predn = self._prepare_pred(pred)
            if self.is_custom_json:
                self.pred_counts.append(len(predn["cls"]))

            no_pred = predn["cls"].shape[0] == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                    "im_name": Path(pbatch["im_file"]).name,
                }
            )
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.confusion_matrix_conf)
                if self.args.visualize:
                    self.confusion_matrix.plot_matches(
                        batch["img"][si],
                        pbatch["im_file"],
                        self.save_dir,
                        self.args.show_labels,
                        self.args.show_conf,
                    )

            if no_pred:
                continue

            if self.args.save_json or self.args.save_txt:
                predn_scaled = self.scale_preds(predn, pbatch)
            if self.args.save_json:
                self.pred_to_json(predn_scaled, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn_scaled,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f"{Path(pbatch['im_file']).stem}.txt",
                )

    def finalize_metrics(self) -> None:
        """Set final values for metrics speed and confusion matrix."""
        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(save_dir=self.save_dir, normalize=normalize, on_plot=self.on_plot)
        self.metrics.speed = self.speed
        self.metrics.confusion_matrix = self.confusion_matrix
        self.metrics.save_dir = self.save_dir

    def _gather_image_metrics(self, metric) -> None:
        """Gather per-image metrics from all GPUs for a single metric object."""
        if RANK == 0:
            gathered_image_metrics = [None] * dist.get_world_size()
            dist.gather_object(metric.image_metrics, gathered_image_metrics, dst=0)
            metric.clear_image_metrics()
            for image_metrics in gathered_image_metrics:
                if image_metrics:
                    metric.image_metrics.update(image_metrics)
        elif RANK > 0:
            dist.gather_object(metric.image_metrics, None, dst=0)
            metric.clear_image_metrics()

    def gather_stats(self) -> None:
        """Gather stats from all GPUs."""
        if RANK == 0:
            gathered_stats = [None] * dist.get_world_size()
            dist.gather_object(self.metrics.stats, gathered_stats, dst=0)
            merged_stats = {key: [] for key in self.metrics.stats}
            for stats_dict in gathered_stats:
                for key, value in stats_dict.items():
                    merged_stats[key].extend(value)
            gathered_json = [None] * dist.get_world_size()
            dist.gather_object(
                (self.jdict, self.gdict if self.build_gdict else None, self.pred_counts), gathered_json, dst=0
            )
            self.jdict = [x for jdict, _, _ in gathered_json for x in jdict]
            self.pred_counts = [x for _, _, counts in gathered_json for x in counts]
            if self.build_gdict:
                for key in "images", "annotations":
                    self.gdict[key] = [x for _, gdict, _ in gathered_json for x in gdict[key]]
            self.metrics.stats = merged_stats
            self._gather_image_metrics(self.metrics.box)
            self.seen = len(self.dataloader.dataset)  # total image count from dataset
        elif RANK > 0:
            dist.gather_object(self.metrics.stats, None, dst=0)
            dist.gather_object((self.jdict, self.gdict if self.build_gdict else None, self.pred_counts), None, dst=0)
            self._gather_image_metrics(self.metrics.box)
            self.jdict = []
            self.metrics.clear_stats()
        if self.args.plots and RANK > -1:
            matrix = torch.as_tensor(self.confusion_matrix.matrix, device=self.device)
            dist.reduce(matrix, dst=0, op=dist.ReduceOp.SUM)
            if RANK == 0:
                self.confusion_matrix.matrix = matrix.cpu().numpy()

    def get_stats(self) -> dict[str, Any]:
        """Calculate and return metrics statistics.

        Returns:
            (dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        stats = self.metrics.results_dict
        if self.args.save_json and self.args.task == "detect":
            stats.update({f"metrics/mAP_{x}(B)": 0.0 for x in ("small", "medium", "large")})
            if self.training:
                stats = self.eval_json(stats)
        self.metrics.clear_stats()
        return stats

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1:
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf
                    % (
                        self.names[c],
                        self.metrics.nt_per_image[c],
                        self.metrics.nt_per_class[c],
                        *self.metrics.class_result(i),
                    )
                )

    def _process_batch(self, preds: dict[str, torch.Tensor], batch: dict[str, Any]) -> dict[str, np.ndarray]:
        """Return correct prediction matrix.

        Args:
            preds (dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for
                10 IoU levels.
        """
        if batch["cls"].shape[0] == 0 or preds["cls"].shape[0] == 0:
            return {"tp": np.zeros((preds["cls"].shape[0], self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: int | None = None) -> torch.utils.data.Dataset:
        """Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        fraction = get_split_fraction(self.args.fraction, self.args.split or "val")
        return build_yolo_dataset(
            self.args, img_path, batch, self.data, mode=mode, stride=self.stride, fraction=fraction
        )

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): DataLoader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(
            dataset,
            batch_size,
            self.args.workers,
            shuffle=False,
            rank=-1,
            drop_last=self.args.compile,
            pin_memory=self.training,
            device=self.device,
        )

    def plot_val_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot validation image samples.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            ni (int): Batch index.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_labels.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )

    def plot_predictions(
        self, batch: dict[str, Any], preds: list[dict[str, torch.Tensor]], ni: int, max_det: int | None = None
    ) -> None:
        """Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (dict[str, Any]): Batch containing images and annotations.
            preds (list[dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (int | None): Maximum number of detections to plot.
        """
        if not preds:
            return
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        batched_preds["bboxes"] = ops.xyxy2xywh(batched_preds["bboxes"])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: dict[str, torch.Tensor], save_conf: bool, shape: tuple[int, int], file: Path) -> None:
        """Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> None:
        """Serialize YOLO predictions to COCO json format.

        Args:
            predn (dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys with
                bounding box coordinates, confidence scores, and class predictions.
            pbatch (dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.

        Examples:
             >>> result = {
             ...     "image_id": 42,
             ...     "file_name": "42.jpg",
             ...     "category_id": 18,
             ...     "bbox": [258.15, 41.29, 348.26, 243.78],
             ...     "score": 0.236,
             ... }
        """
        path = Path(pbatch["im_file"])
        stem = path.stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "file_name": path.name,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def scale_preds(self, predn: dict[str, torch.Tensor], pbatch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Scales predictions to the original image size."""
        return {
            **predn,
            "bboxes": ops.scale_boxes(
                pbatch["imgsz"],
                predn["bboxes"].clone(),
                pbatch["ori_shape"],
                ratio_pad=pbatch["ratio_pad"],
            ),
        }

    def eval_json(self, stats: dict[str, Any]) -> dict[str, Any]:
        """Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (dict[str, Any]): Current statistics dictionary.

        Returns:
            (dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        if self.gdict:
            predictions = iter(self.jdict)
            pred_json = [
                {**next(predictions), "image_id": image["id"]}
                for image, count in zip(self.gdict["images"], self.pred_counts)
                for _ in range(count)
            ]
        else:
            pred_json = self.jdict if self.training else self.save_dir / "predictions.json"
        anno_json = self.gdict or (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: dict[str, Any],
        pred_json: str | Path | list,
        anno_json: str | Path | dict,
        iou_types: str | list[str] = "bbox",
        suffix: str | list[str] = "Box",
    ) -> dict[str, Any]:
        """Evaluate COCO/LVIS or custom COCO-format detection metrics using faster-coco-eval.

        Args:
            stats (dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path | list): Path or in-memory predictions in COCO format.
            anno_json (str | Path | dict): Path or in-memory ground truth in COCO format.
            iou_types (str | list[str]): IoU types to evaluate, such as "bbox", "segm", or "keypoints".
            suffix (str | list[str]): Metric suffixes corresponding to the IoU types.

        Returns:
            (dict[str, Any]): Updated stats dictionary containing the computed COCO-format evaluation metrics.
        """
        if self.args.save_json and len(self.jdict) and (self.is_coco or self.is_lvis or self.gdict):
            LOGGER.info("\nEvaluating faster-coco-eval mAP...")
            try:
                for x in pred_json, anno_json:
                    if isinstance(x, (str, Path)):
                        assert Path(x).is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = getattr(self, "_coco_api", None) or COCO(anno_json)
                self._coco_api = anno
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = (
                        anno.getImgIds()
                        if self.gdict
                        else [int(Path(x).stem) for x in self.dataloader.dataset.im_files]
                    )
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    if not self.training and (self.is_coco or self.is_lvis):
                        stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                        stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]
                        stats["fitness"] = 0.9 * val.stats_as_dict["AP_all"] + 0.1 * val.stats_as_dict["AP_50"]
                    stats["metrics/mAP_small(B)"] = val.stats_as_dict["AP_small"]
                    stats["metrics/mAP_medium(B)"] = val.stats_as_dict["AP_medium"]
                    stats["metrics/mAP_large(B)"] = val.stats_as_dict["AP_large"]
                    if not self.training and self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
