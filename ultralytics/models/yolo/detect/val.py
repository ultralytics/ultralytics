# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import shutil

import cv2
import numpy as np
import torch

from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.engine.results import Results
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import ConfusionMatrix, DetMetrics, box_iou
from ultralytics.utils.plotting import Colors

colors = Colors()
from ultralytics.utils.plotting import plot_images


class DetectionValidator(BaseValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    This class implements validation functionality specific to object detection tasks, including metrics calculation,
    prediction processing, and visualization of results.

    Attributes:
        is_coco (bool): Whether the dataset is COCO.
        is_lvis (bool): Whether the dataset is LVIS.
        class_map (List[int]): Mapping from model class indices to dataset class indices.
        metrics (DetMetrics): Object detection metrics calculator.
        iouv (torch.Tensor): IoU thresholds for mAP calculation.
        niou (int): Number of IoU thresholds.
        lb (List[Any]): List for storing ground truth labels for hybrid saving.
        jdict (List[Dict[str, Any]]): List for storing JSON detection results.
        stats (Dict[str, List[torch.Tensor]]): Dictionary for storing statistics during validation.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionValidator
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml")
        >>> validator = DetectionValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize detection validator with necessary variables and settings.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to use for validation.
            save_dir (Path, optional): Directory to save results.
            args (Dict[str, Any], optional): Arguments for the validator.
            _callbacks (List[Any], optional): List of callback functions.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.is_coco = False
        self.is_lvis = False
        self.class_map = None
        self.args.task = "detect"
        self.iouv = torch.linspace(0.5, 0.95, 10)  # IoU vector for mAP@0.5:0.95
        self.niou = self.iouv.numel()
        self.metrics = DetMetrics()

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch of images for YOLO validation.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.

        Returns:
            (Dict[str, Any]): Preprocessed batch.
        """
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = (batch["img"].half() if self.args.half else batch["img"].float()) / 255
        for k in {"batch_idx", "cls", "bboxes"}:
            batch[k] = batch[k].to(self.device)

        return batch

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO detection validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        val = self.data.get(self.args.split, "")  # validation path
        self.is_coco = (
            isinstance(val, str)
            and "coco" in val
            and (val.endswith(f"{os.sep}val2017.txt") or val.endswith(f"{os.sep}test-dev2017.txt"))
        )  # is COCO
        self.is_lvis = isinstance(val, str) and "lvis" in val and not self.is_coco  # is LVIS
        self.class_map = converter.coco80_to_coco91_class() if self.is_coco else list(range(1, len(model.names) + 1))
        self.args.save_json |= self.args.val and (self.is_coco or self.is_lvis) and not self.training  # run final val
        self.names = model.names
        self.nc = len(model.names)
        self.end2end = getattr(model, "end2end", False)
        self.seen = 0
        self.jdict = []
        self.metrics.names = model.names
        self.confusion_matrix = ConfusionMatrix(names=model.names, save_matches=self.args.plots and self.args.visualize)

    def get_desc(self) -> str:
        """Return a formatted string summarizing class metrics of YOLO model."""
        return ("%22s" + "%11s" * 6) % ("Class", "Images", "Instances", "Box(P", "R", "mAP50", "mAP50-95)")

    def postprocess(self, preds: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[Dict[str, torch.Tensor]]): Processed predictions after NMS, where each dict contains
                'bboxes', 'conf', 'cls', and 'extra' tensors.
        """
        outputs = ops.non_max_suppression(
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

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch of images and annotations for validation.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Batch data containing images and annotations.

        Returns:
            (Dict[str, Any]): Prepared batch with processed annotations.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
        return {
            "cls": cls,
            "bboxes": bbox,
            "ori_shape": ori_shape,
            "imgsz": imgsz,
            "ratio_pad": ratio_pad,
            "im_file": batch["im_file"][si],
        }

    def _prepare_pred(self, pred: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Prepare predictions for evaluation against ground truth.

        Args:
            pred (Dict[str, torch.Tensor]): Post-processed predictions from the model.

        Returns:
            (Dict[str, torch.Tensor]): Prepared predictions in native space.
        """
        if self.args.single_cls:
            pred["cls"] *= 0
        return pred

    def update_metrics(self, preds: List[Dict[str, torch.Tensor]], batch: Dict[str, Any]) -> None:
        """
        Update metrics with new predictions and ground truth.

        Args:
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            batch (Dict[str, Any]): Batch data containing ground truth.
        """
        for si, pred in enumerate(preds):
            self.seen += 1
            pbatch = self._prepare_batch(si, batch)
            predn = self._prepare_pred(pred)
            idx = batch["batch_idx"] == si
            bbox = pbatch["bboxes"]
            labelsn = torch.cat((batch["cls"][idx], bbox), 1)  # native-space labels
            cls = pbatch["cls"].cpu().numpy()
            no_pred = len(predn["cls"]) == 0
            self.metrics.update_stats(
                {
                    **self._process_batch(predn, pbatch),
                    "target_cls": cls,
                    "target_img": np.unique(cls),
                    "conf": np.zeros(0) if no_pred else predn["conf"].cpu().numpy(),
                    "pred_cls": np.zeros(0) if no_pred else predn["cls"].cpu().numpy(),
                }
            )
            # Evaluate
            if self.args.plots:
                self.confusion_matrix.process_batch(predn, pbatch, conf=self.args.conf)
                if self.args.visualize:
                    # self.confusion_matrix.plot_matches(batch["img"][si], pbatch["im_file"], self.save_dir)
                    self.output_bad_cases(predn, labelsn, batch, si, conf=self.args.conf)

            if no_pred:
                continue

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, pbatch)
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
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

    def get_stats(self) -> Dict[str, Any]:
        """
        Calculate and return metrics statistics.

        Returns:
            (Dict[str, Any]): Dictionary containing metrics results.
        """
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        return self.metrics.results_dict

    def print_results(self) -> None:
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
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

    def output_bad_cases(self, detections, labels, batch, si, conf: float = 0.25, iou_thres: float = 0.45):
        """Out the images with overkill and underkill result
        Args:
            # detections (Array[N, 6]): Detected bounding boxes and their associated information.
            #                           Each row should contain (x1, y1, x2, y2, conf, class).
            # labels (Array[M, 5]): Ground truth bounding boxes and their associated class labels.
            #                       Each row should contain (class, x1, y1, x2, y2).
            detections (Dict[str, torch.Tensor]): Dictionary containing detected bounding boxes and their associated information.
                                       Should contain 'cls', 'conf', and 'bboxes' keys, where 'bboxes' can be
                                       Array[N, 4] for regular boxes or Array[N, 5] for OBB with angle.
            batch (Dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' (Array[M, 4]| Array[M, 5]) and
                'cls' (Array[M]) keys, where M is the number of ground truth objects.
            conf (float, optional): Confidence threshold for detections.
            iou_thres (float, optional): IoU threshold for matching detections to ground truth.
        """
        (self.save_dir / "visualizations"/ "false_negative").mkdir(parents=True, exist_ok=True)
        (self.save_dir / "visualizations" / "false_positive").mkdir(parents=True, exist_ok=True)

        conf = 0.25 if conf in {None, 0.001} else conf  # apply 0.25 if default val conf is passed
        gt_cls, gt_bboxes = batch["cls"], batch["bboxes"]
        no_pred = len(detections["cls"]) == 0
        if no_pred:
            detections = torch.empty((0, 6), device=self.device)  # Output all labels
        else:
            detections = torch.cat([detections['bboxes'], detections['conf'].reshape(-1, 1), detections['cls'].reshape(-1, 1)], 1)

        detections = detections[detections[:, 4] > conf]
        # gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        boxes = torch.cat(
            [detections[:, :4], detections[:, 4].reshape(-1, 1), detection_classes.reshape(-1, 1)], dim=1
        ).cpu()

        x = torch.where(iou > iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        # matches is the index of box that meet the iou threshold. like
        # [[          0           0     0.81668], ground truth box index, pred box index, iou
        #  [          1           1     0.88082],
        #  [          2           2      0.9422]]
        # We need to find the pairs not in matches here. That is iou < iou_thres
        # Find the ground truth box without prediction box, and prediction box without ground truth
        x = torch.where(iou > iou_thres)
        torch.where(iou <= iou_thres)  # y[0] is label, y[1] is predict
        labels_matches = matches[:, 0]
        pred_matches = matches[:, 1]

        false_negative = np.setdiff1d(list(range(labels.shape[0])), labels_matches)
        false_positive = np.setdiff1d(list(range(detections.shape[0])), pred_matches)

        if false_negative.shape[0] > 0:
            # plot false negative images
            # In false negative part, show all correct detections, then append the false negative box from label

            fn_labels = labels[false_negative].cpu()
            correct_labels = labels[[i for i in range(labels.shape[0]) if i not in false_negative]].cpu()

            label_boxes = torch.cat(
                [
                    torch.cat(
                        [correct_labels, torch.zeros(correct_labels.shape[0], 1)],  # Correct label boxes
                        dim=1,
                    )[:, torch.tensor([1, 2, 3, 4, 5, 0])],
                    torch.cat(
                        [fn_labels, torch.zeros(fn_labels.shape[0], 1)],  # Under kill label boxes
                        dim=1,
                    )[:, torch.tensor([1, 2, 3, 4, 5, 0])],
                ]
            )  # Rearrange the element position of fn_labels
            detection_boxes = boxes

            file_name = batch["im_file"][si]

            label_color_list = [colors.GREEN_COLOR] * correct_labels.shape[0] + [colors.RED_COLOR] * fn_labels.shape[0]
            detection_color_list = [colors.BLUE_COLOR] * detections.shape[0]

            combined_img = self._generate_combined_img(
                detection_boxes, detection_color_list, label_boxes, label_color_list, file_name
            )

            cv2.imwrite(str(self.save_dir / "visualizations" / "false_negative" / os.path.split(file_name)[1]),
                        combined_img)

        if false_positive.shape[0] > 0:
            # plot false positive images
            # In false positive mode, will show all detection result on images,
            # then mark the false positive part with red

            detection_boxes = boxes
            label_boxes = torch.cat(
                [labels.cpu(), torch.zeros(labels.shape[0], 1)],  # Correct label boxes
                dim=1,
            )[:, torch.tensor([1, 2, 3, 4, 5, 0])]

            label_color_list = [colors.GREEN_COLOR] * labels.shape[0]
            detection_color_list = [colors.BLUE_COLOR] * detections.shape[0]
            [colors.GREEN_COLOR] * detections.shape[0] + [colors.BLUE_COLOR] * labels.shape[0]
            for i in false_positive:
                detection_color_list[i] = colors.RED_COLOR  # Replace the false positive part with red color
            file_name = batch["im_file"][si]

            combined_img = self._generate_combined_img(
                detection_boxes, detection_color_list, label_boxes, label_color_list, file_name
            )

            cv2.imwrite(str(self.save_dir / "visualizations" / "false_positive" / os.path.split(file_name)[1]),
                        combined_img)

    def _generate_combined_img(self, detection_boxes, detection_color_list, label_boxes, label_color_list, file_name):
        label_plot_args = dict(line_width=1, boxes=True, color_list=label_color_list)
        detection_plot_args = dict(line_width=1, boxes=True, color_list=detection_color_list)
        # Prepare label image
        label_result = Results(orig_img=cv2.imread(file_name), path=file_name, names=self.names, boxes=label_boxes)
        label_plotted_img = label_result.plot(**label_plot_args)
        # Prepare detection image
        detection_result = Results(
            orig_img=cv2.imread(file_name), path=file_name, names=self.names, boxes=detection_boxes
        )
        detection_plotted_img = detection_result.plot(**detection_plot_args)

        label_plotted_img = cv2.copyMakeBorder(
            label_plotted_img, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=[127, 127, 127]
        )
        detection_plotted_img = cv2.copyMakeBorder(
            detection_plotted_img, 25, 0, 0, 0, cv2.BORDER_CONSTANT, value=[127, 127, 127]
        )

        label_plotted_img = cv2.putText(
            label_plotted_img, "Ground Truth", (20, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )
        detection_plotted_img = cv2.putText(
            detection_plotted_img, "Predicted", (20, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA
        )

        combined_img = np.concatenate([label_plotted_img, detection_plotted_img], axis=1)
        return combined_img

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with 'bboxes' and 'cls' keys.
            batch (Dict[str, Any]): Batch dictionary containing ground truth data with 'bboxes' and 'cls' keys.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing 'tp' key with correct prediction matrix of shape (N, 10) for 10 IoU levels.
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = box_iou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def build_dataset(self, img_path: str, mode: str = "val", batch: Optional[int] = None) -> torch.utils.data.Dataset:
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)

    def get_dataloader(self, dataset_path: str, batch_size: int) -> torch.utils.data.DataLoader:
        """
        Construct and return dataloader.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Size of each batch.

        Returns:
            (torch.utils.data.DataLoader): Dataloader for validation.
        """
        dataset = self.build_dataset(dataset_path, batch=batch_size, mode="val")
        return build_dataloader(dataset, batch_size, self.args.workers, shuffle=False, rank=-1)  # return dataloader

    def plot_val_samples(self, batch: Dict[str, Any], ni: int) -> None:
        """
        Plot validation image samples.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.
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
        self, batch: Dict[str, Any], preds: List[Dict[str, torch.Tensor]], ni: int, max_det: Optional[int] = None
    ) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (Dict[str, Any]): Batch containing images and annotations.
            preds (List[Dict[str, torch.Tensor]]): List of predictions from the model.
            ni (int): Batch index.
            max_det (Optional[int]): Maximum number of detections to plot.
        """
        # TODO: optimize this
        for i, pred in enumerate(preds):
            pred["batch_idx"] = torch.ones_like(pred["conf"]) * i  # add batch index to predictions
        keys = preds[0].keys()
        max_det = max_det or self.args.max_det
        batched_preds = {k: torch.cat([x[k][:max_det] for x in preds], dim=0) for k in keys}
        # TODO: fix this
        batched_preds["bboxes"][:, :4] = ops.xyxy2xywh(batched_preds["bboxes"][:, :4])  # convert to xywh format
        plot_images(
            images=batch["img"],
            labels=batched_preds,
            paths=batch["im_file"],
            fname=self.save_dir / f"val_batch{ni}_pred.jpg",
            names=self.names,
            on_plot=self.on_plot,
        )  # pred

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO detections to a txt file in normalized coordinates in a specific format.

        Args:
            predn (Dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', and 'cls'.
            save_conf (bool): Whether to save confidence scores.
            shape (Tuple[int, int]): Shape of the original image (height, width).
            file (Path): File path to save the detections.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> None:
        """
        Serialize YOLO predictions to COCO json format.

        Args:
            predn (Dict[str, torch.Tensor]): Predictions dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            pbatch (Dict[str, Any]): Batch dictionary containing 'imgsz', 'ori_shape', 'ratio_pad', and 'im_file'.
        """
        stem = Path(pbatch["im_file"]).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.scale_boxes(
            pbatch["imgsz"],
            predn["bboxes"].clone(),
            pbatch["ori_shape"],
            ratio_pad=pbatch["ratio_pad"],
        )
        box = ops.xyxy2xywh(box)  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c in zip(box.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "score": round(s, 5),
                }
            )

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and return performance statistics.

        Args:
            stats (Dict[str, Any]): Current statistics dictionary.

        Returns:
            (Dict[str, Any]): Updated statistics dictionary with COCO/LVIS evaluation results.
        """
        pred_json = self.save_dir / "predictions.json"  # predictions
        anno_json = (
            self.data["path"]
            / "annotations"
            / ("instances_val2017.json" if self.is_coco else f"lvis_v1_{self.args.split}.json")
        )  # annotations
        return self.coco_evaluate(stats, pred_json, anno_json)

    def coco_evaluate(
        self,
        stats: Dict[str, Any],
        pred_json: str,
        anno_json: str,
        iou_types: Union[str, List[str]] = "bbox",
        suffix: Union[str, List[str]] = "Box",
    ) -> Dict[str, Any]:
        """
        Evaluate COCO/LVIS metrics using faster-coco-eval library.

        Performs evaluation using the faster-coco-eval library to compute mAP metrics
        for object detection. Updates the provided stats dictionary with computed metrics
        including mAP50, mAP50-95, and LVIS-specific metrics if applicable.

        Args:
            stats (Dict[str, Any]): Dictionary to store computed metrics and statistics.
            pred_json (str | Path]): Path to JSON file containing predictions in COCO format.
            anno_json (str | Path]): Path to JSON file containing ground truth annotations in COCO format.
            iou_types (str | List[str]]): IoU type(s) for evaluation. Can be single string or list of strings.
                Common values include "bbox", "segm", "keypoints". Defaults to "bbox".
            suffix (str | List[str]]): Suffix to append to metric names in stats dictionary. Should correspond
                to iou_types if multiple types provided. Defaults to "Box".

        Returns:
            (Dict[str, Any]): Updated stats dictionary containing the computed COCO/LVIS evaluation metrics.
        """
        if self.args.save_json and (self.is_coco or self.is_lvis) and len(self.jdict):
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                for x in pred_json, anno_json:
                    assert x.is_file(), f"{x} file not found"
                iou_types = [iou_types] if isinstance(iou_types, str) else iou_types
                suffix = [suffix] if isinstance(suffix, str) else suffix
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                anno = COCO(anno_json)
                pred = anno.loadRes(pred_json)
                for i, iou_type in enumerate(iou_types):
                    val = COCOeval_faster(
                        anno, pred, iouType=iou_type, lvis_style=self.is_lvis, print_function=LOGGER.info
                    )
                    val.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # images to eval
                    val.evaluate()
                    val.accumulate()
                    val.summarize()

                    # update mAP50-95 and mAP50
                    stats[f"metrics/mAP50({suffix[i][0]})"] = val.stats_as_dict["AP_50"]
                    stats[f"metrics/mAP50-95({suffix[i][0]})"] = val.stats_as_dict["AP_all"]

                    if self.is_lvis:
                        stats[f"metrics/APr({suffix[i][0]})"] = val.stats_as_dict["APr"]
                        stats[f"metrics/APc({suffix[i][0]})"] = val.stats_as_dict["APc"]
                        stats[f"metrics/APf({suffix[i][0]})"] = val.stats_as_dict["APf"]

                if self.is_lvis:
                    stats["fitness"] = stats["metrics/mAP50-95(B)"]  # always use box mAP50-95 for fitness
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
