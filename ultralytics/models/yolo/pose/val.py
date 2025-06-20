# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.checks import check_requirements
from ultralytics.utils.metrics import OKS_SIGMA, PoseMetrics, kpt_iou


class PoseValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on a pose model.

    This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
    specialized metrics for pose evaluation.

    Attributes:
        sigma (np.ndarray): Sigma values for OKS calculation, either OKS_SIGMA or ones divided by number of keypoints.
        kpt_shape (List[int]): Shape of the keypoints, typically [17, 3] for COCO format.
        args (dict): Arguments for the validator including task set to "pose".
        metrics (PoseMetrics): Metrics object for pose evaluation.

    Methods:
        preprocess: Preprocess batch by converting keypoints data to float and moving it to the device.
        get_desc: Return description of evaluation metrics in string format.
        init_metrics: Initialize pose estimation metrics for YOLO model.
        _prepare_batch: Prepare a batch for processing by converting keypoints to float and scaling to original
            dimensions.
        _prepare_pred: Prepare and scale keypoints in predictions for pose processing.
        _process_batch: Return correct prediction matrix by computing Intersection over Union (IoU) between
            detections and ground truth.
        plot_val_samples: Plot and save validation set samples with ground truth bounding boxes and keypoints.
        plot_predictions: Plot and save model predictions with bounding boxes and keypoints.
        save_one_txt: Save YOLO pose detections to a text file in normalized coordinates.
        pred_to_json: Convert YOLO predictions to COCO JSON format.
        eval_json: Evaluate object detection model using COCO JSON format.

    Examples:
        >>> from ultralytics.models.yolo.pose import PoseValidator
        >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
        >>> validator = PoseValidator(args=args)
        >>> validator()
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize a PoseValidator object for pose estimation validation.

        This validator is specifically designed for pose estimation tasks, handling keypoints and implementing
        specialized metrics for pose evaluation.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (Path | str, optional): Directory to save results.
            args (dict, optional): Arguments for the validator including task set to "pose".
            _callbacks (list, optional): List of callback functions to be executed during validation.

        Examples:
            >>> from ultralytics.models.yolo.pose import PoseValidator
            >>> args = dict(model="yolo11n-pose.pt", data="coco8-pose.yaml")
            >>> validator = PoseValidator(args=args)
            >>> validator()

        Notes:
            This class extends DetectionValidator with pose-specific functionality. It initializes with sigma values
            for OKS calculation and sets up PoseMetrics for evaluation. A warning is displayed when using Apple MPS
            due to a known bug with pose models.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.sigma = None
        self.kpt_shape = None
        self.args.task = "pose"
        self.metrics = PoseMetrics()
        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess batch by converting keypoints data to float and moving it to the device."""
        batch = super().preprocess(batch)
        batch["keypoints"] = batch["keypoints"].to(self.device).float()
        return batch

    def get_desc(self) -> str:
        """Return description of evaluation metrics in string format."""
        return ("%22s" + "%11s" * 10) % (
            "Class",
            "Images",
            "Instances",
            "Box(P",
            "R",
            "mAP50",
            "mAP50-95)",
            "Pose(P",
            "R",
            "mAP50",
            "mAP50-95)",
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO pose validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        self.kpt_shape = self.data["kpt_shape"]
        is_pose = self.kpt_shape == [17, 3]
        nkpt = self.kpt_shape[0]
        self.sigma = OKS_SIGMA if is_pose else np.ones(nkpt) / nkpt

    def postprocess(self, preds: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Postprocess YOLO predictions to extract and reshape keypoints for pose estimation.

        This method extends the parent class postprocessing by extracting keypoints from the 'extra'
        field of predictions and reshaping them according to the keypoint shape configuration.
        The keypoints are reshaped from a flattened format to the proper dimensional structure
        (typically [N, 17, 3] for COCO pose format).

        Args:
            preds (torch.Tensor): Raw prediction tensor from the YOLO pose model containing
                bounding boxes, confidence scores, class predictions, and keypoint data.

        Returns:
            (Dict[torch.Tensor]): Dict of processed prediction dictionaries, each containing:
                - 'bboxes': Bounding box coordinates
                - 'conf': Confidence scores
                - 'cls': Class predictions
                - 'keypoints': Reshaped keypoint coordinates with shape (-1, *self.kpt_shape)

        Note:
            If no keypoints are present in a prediction (empty keypoints), that prediction
            is skipped and continues to the next one. The keypoints are extracted from the
            'extra' field which contains additional task-specific data beyond basic detection.
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["keypoints"] = pred.pop("extra").view(-1, *self.kpt_shape)  # remove extra if exists
        return preds

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a batch for processing by converting keypoints to float and scaling to original dimensions.

        Args:
            si (int): Batch index.
            batch (Dict[str, Any]): Dictionary containing batch data with keys like 'keypoints', 'batch_idx', etc.

        Returns:
            (Dict[str, Any]): Prepared batch with keypoints scaled to original image dimensions.

        Notes:
            This method extends the parent class's _prepare_batch method by adding keypoint processing.
            Keypoints are scaled from normalized coordinates to original image dimensions.
        """
        pbatch = super()._prepare_batch(si, batch)
        kpts = batch["keypoints"][batch["batch_idx"] == si]
        h, w = pbatch["imgsz"]
        kpts = kpts.clone()
        kpts[..., 0] *= w
        kpts[..., 1] *= h
        kpts = ops.scale_coords(pbatch["imgsz"], kpts, pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"])
        pbatch["keypoints"] = kpts
        return pbatch

    def _prepare_pred(self, pred: Dict[str, Any], pbatch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare and scale keypoints in predictions for pose processing.

        This method extends the parent class's _prepare_pred method to handle keypoint scaling. It first calls
        the parent method to get the basic prediction boxes, then extracts and scales the keypoint coordinates
        to match the original image dimensions.

        Args:
            pred (Dict[str, torch.Tensor]): Post-processed predictions from the model.
            pbatch (Dict[str, Any]): Processed batch dictionary containing image information including:
                - imgsz: Image size used for inference
                - ori_shape: Original image shape
                - ratio_pad: Ratio and padding information for coordinate scaling

        Returns:
            (Dict[str, Any]): Processed prediction dictionary with keypoints scaled to original image dimensions.
        """
        predn = super()._prepare_pred(pred, pbatch)
        predn["keypoints"] = ops.scale_coords(
            pbatch["imgsz"], pred.get("keypoints").clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"]
        )
        return predn

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Return correct prediction matrix by computing Intersection over Union (IoU) between detections and ground truth.

        Args:
            preds (Dict[str, torch.Tensor]): Dictionary containing prediction data with keys 'cls' for class predictions
                and 'keypoints' for keypoint predictions.
            batch (Dict[str, Any]): Dictionary containing ground truth data with keys 'cls' for class labels,
                'bboxes' for bounding boxes, and 'keypoints' for keypoint annotations.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing the correct prediction matrix including 'tp_p' for pose
                true positives across 10 IoU levels.

        Notes:
            `0.53` scale factor used in area computation is referenced from
            https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384.
        """
        tp = super()._process_batch(preds, batch)
        gt_cls = batch["cls"]
        if len(gt_cls) == 0 or len(preds["cls"]) == 0:
            tp_p = np.zeros((len(preds["cls"]), self.niou), dtype=bool)
        else:
            # `0.53` is from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = ops.xyxy2xywh(batch["bboxes"])[:, 2:].prod(1) * 0.53
            iou = kpt_iou(batch["keypoints"], preds["keypoints"], sigma=self.sigma, area=area)
            tp_p = self.match_predictions(preds["cls"], gt_cls, iou).cpu().numpy()
        tp.update({"tp_p": tp_p})  # update tp with kpts IoU
        return tp

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO pose detections to a text file in normalized coordinates.

        Args:
            predn (Dict[str, torch.Tensor]): Dictionary containing predictions with keys 'bboxes', 'conf', 'cls' and 'keypoints.
            save_conf (bool): Whether to save confidence scores.
            shape (Tuple[int, int]): Shape of the original image (height, width).
            file (Path): Output file path to save detections.

        Notes:
            The output format is: class_id x_center y_center width height confidence keypoints where keypoints are
            normalized (x, y, visibility) values for each point.
        """
        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            boxes=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
            keypoints=predn["keypoints"],
        ).save_txt(file, save_conf=save_conf)

    def pred_to_json(self, predn: Dict[str, torch.Tensor], filename: str) -> None:
        """
        Convert YOLO predictions to COCO JSON format.

        This method takes prediction tensors and a filename, converts the bounding boxes from YOLO format
        to COCO format, and appends the results to the internal JSON dictionary (self.jdict).

        Args:
            predn (Dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', 'cls',
                and 'keypoints' tensors.
            filename (str): Path to the image file for which predictions are being processed.

        Notes:
            The method extracts the image ID from the filename stem (either as an integer if numeric, or as a string),
            converts bounding boxes from xyxy to xywh format, and adjusts coordinates from center to top-left corner
            before saving to the JSON dictionary.
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        box = ops.xyxy2xywh(predn["bboxes"])  # xywh
        box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
        for b, s, c, k in zip(
            box.tolist(),
            predn["conf"].tolist(),
            predn["cls"].tolist(),
            predn["keypoints"].flatten(1, 2).tolist(),
        ):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "bbox": [round(x, 3) for x in b],
                    "keypoints": k,
                    "score": round(s, 5),
                }
            )

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate object detection model using COCO JSON format."""
        if self.args.save_json and self.is_coco and len(self.jdict):
            anno_json = self.data["path"] / "annotations/person_keypoints_val2017.json"  # annotations
            pred_json = self.save_dir / "predictions.json"  # predictions
            LOGGER.info(f"\nEvaluating faster-coco-eval mAP using {pred_json} and {anno_json}...")
            try:
                check_requirements("faster-coco-eval>=1.6.7")
                from faster_coco_eval import COCO, COCOeval_faster

                for x in anno_json, pred_json:
                    assert x.is_file(), f"{x} file not found"
                anno = COCO(anno_json)  # init annotations api
                pred = anno.loadRes(pred_json)  # init predictions api (must pass string, not Path)
                kwargs = dict(cocoGt=anno, cocoDt=pred, print_function=LOGGER.info)
                for i, eval in enumerate(
                    [COCOeval_faster(iouType="bbox", **kwargs), COCOeval_faster(iouType="keypoints", **kwargs)]
                ):
                    eval.params.imgIds = [int(Path(x).stem) for x in self.dataloader.dataset.im_files]  # im to eval
                    eval.evaluate()
                    eval.accumulate()
                    eval.summarize()
                    idx = i * 4 + 2
                    # update mAP50-95 and mAP50
                    stats[self.metrics.keys[idx + 1]], stats[self.metrics.keys[idx]] = eval.stats[:2]
            except Exception as e:
                LOGGER.warning(f"faster-coco-eval unable to run: {e}")
        return stats
