# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch

from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import LOGGER, ops
from ultralytics.utils.metrics import OBBMetrics, batch_probiou


class OBBValidator(DetectionValidator):
    """
    A class extending the DetectionValidator class for validation based on an Oriented Bounding Box (OBB) model.

    This validator specializes in evaluating models that predict rotated bounding boxes, commonly used for aerial and
    satellite imagery where objects can appear at various orientations.

    Attributes:
        args (dict): Configuration arguments for the validator.
        metrics (OBBMetrics): Metrics object for evaluating OBB model performance.
        is_dota (bool): Flag indicating whether the validation dataset is in DOTA format.

    Methods:
        init_metrics: Initialize evaluation metrics for YOLO.
        _process_batch: Process batch of detections and ground truth boxes to compute IoU matrix.
        _prepare_batch: Prepare batch data for OBB validation.
        _prepare_pred: Prepare predictions with scaled and padded bounding boxes.
        plot_predictions: Plot predicted bounding boxes on input images.
        pred_to_json: Serialize YOLO predictions to COCO json format.
        save_one_txt: Save YOLO detections to a txt file in normalized coordinates.
        eval_json: Evaluate YOLO output in JSON format and return performance statistics.

    Examples:
        >>> from ultralytics.models.yolo.obb import OBBValidator
        >>> args = dict(model="yolo11n-obb.pt", data="dota8.yaml")
        >>> validator = OBBValidator(args=args)
        >>> validator(model=args["model"])
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        """
        Initialize OBBValidator and set task to 'obb', metrics to OBBMetrics.

        This constructor initializes an OBBValidator instance for validating Oriented Bounding Box (OBB) models.
        It extends the DetectionValidator class and configures it specifically for the OBB task.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): Dataloader to be used for validation.
            save_dir (str | Path, optional): Directory to save results.
            args (dict | SimpleNamespace, optional): Arguments containing validation parameters.
            _callbacks (list, optional): List of callback functions to be called during validation.
        """
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "obb"
        self.metrics = OBBMetrics()

    def init_metrics(self, model: torch.nn.Module) -> None:
        """
        Initialize evaluation metrics for YOLO obb validation.

        Args:
            model (torch.nn.Module): Model to validate.
        """
        super().init_metrics(model)
        val = self.data.get(self.args.split, "")  # validation path
        self.is_dota = isinstance(val, str) and "DOTA" in val  # check if dataset is DOTA format

    def _process_batch(self, preds: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Compute the correct prediction matrix for a batch of detections and ground truth bounding boxes.

        Args:
            preds (Dict[str, torch.Tensor]): Prediction dictionary containing 'cls' and 'bboxes' keys with detected
                class labels and bounding boxes.
            batch (Dict[str, torch.Tensor]): Batch dictionary containing 'cls' and 'bboxes' keys with ground truth
                class labels and bounding boxes.

        Returns:
            (Dict[str, np.ndarray]): Dictionary containing 'tp' key with the correct prediction matrix as a numpy
                array with shape (N, 10), which includes 10 IoU levels for each detection, indicating the accuracy
                of predictions compared to the ground truth.

        Examples:
            >>> detections = torch.rand(100, 7)  # 100 sample detections
            >>> gt_bboxes = torch.rand(50, 5)  # 50 sample ground truth boxes
            >>> gt_cls = torch.randint(0, 5, (50,))  # 50 ground truth class labels
            >>> correct_matrix = validator._process_batch(detections, gt_bboxes, gt_cls)
        """
        if len(batch["cls"]) == 0 or len(preds["cls"]) == 0:
            return {"tp": np.zeros((len(preds["cls"]), self.niou), dtype=bool)}
        iou = batch_probiou(batch["bboxes"], preds["bboxes"])
        return {"tp": self.match_predictions(preds["cls"], batch["cls"], iou).cpu().numpy()}

    def postprocess(self, preds: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
        """
        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[Dict[str, torch.Tensor]]): Processed predictions with angle information concatenated to bboxes.
        """
        preds = super().postprocess(preds)
        for pred in preds:
            pred["bboxes"] = torch.cat([pred["bboxes"], pred.pop("extra")], dim=-1)  # concatenate angle
        return preds

    def _prepare_batch(self, si: int, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare batch data for OBB validation with proper scaling and formatting.

        Args:
            si (int): Batch index to process.
            batch (Dict[str, Any]): Dictionary containing batch data with keys:
                - batch_idx: Tensor of batch indices
                - cls: Tensor of class labels
                - bboxes: Tensor of bounding boxes
                - ori_shape: Original image shapes
                - img: Batch of images
                - ratio_pad: Ratio and padding information

        Returns:
            (Dict[str, Any]): Prepared batch data with scaled bounding boxes and metadata.
        """
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        if len(cls):
            bbox[..., :4].mul_(torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]])  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad, xywh=True)  # native-space labels
        return {"cls": cls, "bboxes": bbox, "ori_shape": ori_shape, "imgsz": imgsz, "ratio_pad": ratio_pad}

    def _prepare_pred(self, pred: Dict[str, torch.Tensor], pbatch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """
        Prepare predictions by scaling bounding boxes to original image dimensions.

        This method takes prediction tensors containing bounding box coordinates and scales them from the model's
        input dimensions to the original image dimensions using the provided batch information.

        Args:
            pred (Dict[str, torch.Tensor]): Prediction dictionary containing bounding box coordinates and other information.
            pbatch (Dict[str, Any]): Dictionary containing batch information with keys:
                - imgsz (tuple): Model input image size.
                - ori_shape (tuple): Original image shape.
                - ratio_pad (tuple): Ratio and padding information for scaling.

        Returns:
            (Dict[str, torch.Tensor]): Scaled prediction dictionary with bounding boxes in original image dimensions.
        """
        cls = pred["cls"]
        if self.args.single_cls:
            cls *= 0
        bboxes = ops.scale_boxes(
            pbatch["imgsz"], pred["bboxes"].clone(), pbatch["ori_shape"], ratio_pad=pbatch["ratio_pad"], xywh=True
        )  # native-space pred
        return {"bboxes": bboxes, "conf": pred["conf"], "cls": cls}

    def plot_predictions(self, batch: Dict[str, Any], preds: List[torch.Tensor], ni: int) -> None:
        """
        Plot predicted bounding boxes on input images and save the result.

        Args:
            batch (Dict[str, Any]): Batch data containing images, file paths, and other metadata.
            preds (List[torch.Tensor]): List of prediction tensors for each image in the batch.
            ni (int): Batch index used for naming the output file.

        Examples:
            >>> validator = OBBValidator()
            >>> batch = {"img": images, "im_file": paths}
            >>> preds = [torch.rand(10, 7)]  # Example predictions for one image
            >>> validator.plot_predictions(batch, preds, 0)
        """
        for p in preds:
            # TODO: fix this duplicated `xywh2xyxy`
            p["bboxes"][:, :4] = ops.xywh2xyxy(p["bboxes"][:, :4])  # convert to xyxy format for plotting
        super().plot_predictions(batch, preds, ni)  # plot bboxes

    def pred_to_json(self, predn: Dict[str, torch.Tensor], filename: Union[str, Path]) -> None:
        """
        Convert YOLO predictions to COCO JSON format with rotated bounding box information.

        Args:
            predn (Dict[str, torch.Tensor]): Prediction dictionary containing 'bboxes', 'conf', and 'cls' keys
                with bounding box coordinates, confidence scores, and class predictions.
            filename (str | Path): Path to the image file for which predictions are being processed.

        Notes:
            This method processes rotated bounding box predictions and converts them to both rbox format
            (x, y, w, h, angle) and polygon format (x1, y1, x2, y2, x3, y3, x4, y4) before adding them
            to the JSON dictionary.
        """
        stem = Path(filename).stem
        image_id = int(stem) if stem.isnumeric() else stem
        rbox = predn["bboxes"]
        poly = ops.xywhr2xyxyxyxy(rbox).view(-1, 8)
        for r, b, s, c in zip(rbox.tolist(), poly.tolist(), predn["conf"].tolist(), predn["cls"].tolist()):
            self.jdict.append(
                {
                    "image_id": image_id,
                    "category_id": self.class_map[int(c)],
                    "score": round(s, 5),
                    "rbox": [round(x, 3) for x in r],
                    "poly": [round(x, 3) for x in b],
                }
            )

    def save_one_txt(self, predn: Dict[str, torch.Tensor], save_conf: bool, shape: Tuple[int, int], file: Path) -> None:
        """
        Save YOLO OBB detections to a text file in normalized coordinates.

        Args:
            predn (torch.Tensor): Predicted detections with shape (N, 7) containing bounding boxes, confidence scores,
                class predictions, and angles in format (x, y, w, h, conf, cls, angle).
            save_conf (bool): Whether to save confidence scores in the text file.
            shape (Tuple[int, int]): Original image shape in format (height, width).
            file (Path): Output file path to save detections.

        Examples:
            >>> validator = OBBValidator()
            >>> predn = torch.tensor([[100, 100, 50, 30, 0.9, 0, 45]])  # One detection: x,y,w,h,conf,cls,angle
            >>> validator.save_one_txt(predn, True, (640, 480), "detection.txt")
        """
        import numpy as np

        from ultralytics.engine.results import Results

        Results(
            np.zeros((shape[0], shape[1]), dtype=np.uint8),
            path=None,
            names=self.names,
            obb=torch.cat([predn["bboxes"], predn["conf"].unsqueeze(-1), predn["cls"].unsqueeze(-1)], dim=1),
        ).save_txt(file, save_conf=save_conf)

    def eval_json(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate YOLO output in JSON format and save predictions in DOTA format.

        Args:
            stats (Dict[str, Any]): Performance statistics dictionary.

        Returns:
            (Dict[str, Any]): Updated performance statistics.
        """
        if self.args.save_json and self.is_dota and len(self.jdict):
            import json
            import re
            from collections import defaultdict

            pred_json = self.save_dir / "predictions.json"  # predictions
            pred_txt = self.save_dir / "predictions_txt"  # predictions
            pred_txt.mkdir(parents=True, exist_ok=True)
            data = json.load(open(pred_json))
            # Save split results
            LOGGER.info(f"Saving predictions with DOTA format to {pred_txt}...")
            for d in data:
                image_id = d["image_id"]
                score = d["score"]
                classname = self.names[d["category_id"] - 1].replace(" ", "-")
                p = d["poly"]

                with open(f"{pred_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                    f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")
            # Save merged results, this could result slightly lower map than using official merging script,
            # because of the probiou calculation.
            pred_merged_txt = self.save_dir / "predictions_merged_txt"  # predictions
            pred_merged_txt.mkdir(parents=True, exist_ok=True)
            merged_results = defaultdict(list)
            LOGGER.info(f"Saving merged predictions with DOTA format to {pred_merged_txt}...")
            for d in data:
                image_id = d["image_id"].split("__", 1)[0]
                pattern = re.compile(r"\d+___\d+")
                x, y = (int(c) for c in re.findall(pattern, d["image_id"])[0].split("___"))
                bbox, score, cls = d["rbox"], d["score"], d["category_id"] - 1
                bbox[0] += x
                bbox[1] += y
                bbox.extend([score, cls])
                merged_results[image_id].append(bbox)
            for image_id, bbox in merged_results.items():
                bbox = torch.tensor(bbox)
                max_wh = torch.max(bbox[:, :2]).item() * 2
                c = bbox[:, 6:7] * max_wh  # classes
                scores = bbox[:, 5]  # scores
                b = bbox[:, :5].clone()
                b[:, :2] += c
                # 0.3 could get results close to the ones from official merging script, even slightly better.
                i = ops.nms_rotated(b, scores, 0.3)
                bbox = bbox[i]

                b = ops.xywhr2xyxyxyxy(bbox[:, :5]).view(-1, 8)
                for x in torch.cat([b, bbox[:, 5:7]], dim=-1).tolist():
                    classname = self.names[int(x[-1])].replace(" ", "-")
                    p = [round(i, 3) for i in x[:-2]]  # poly
                    score = round(x[-2], 3)

                    with open(f"{pred_merged_txt / f'Task1_{classname}'}.txt", "a", encoding="utf-8") as f:
                        f.writelines(f"{image_id} {score} {p[0]} {p[1]} {p[2]} {p[3]} {p[4]} {p[5]} {p[6]} {p[7]}\n")

        return stats
