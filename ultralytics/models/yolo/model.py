# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ultralytics.data.build import load_inference_source
from ultralytics.engine.results import Results
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.modules.head import ADMBHead, AnomalyDetection
from ultralytics.nn.tasks import (
    ClassificationModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    WorldModel,
    YOLOAnomalyModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import LOGGER, ROOT, YAML, ops


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO models, automatically switching to specialized model types
    (YOLOWorld or YOLOE) based on the model filename. It supports various computer vision tasks including object
    detection, segmentation, classification, pose estimation, and oriented bounding box detection.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, segment, classify, pose, obb).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize a YOLO model with automatic type detection.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.

    Examples:
        Load a pretrained YOLO26n detection model
        >>> model = YOLO("yolo26n.pt")

        Load a pretrained YOLO26n segmentation model
        >>> model = YOLO("yolo26n-seg.pt")

        Initialize from a YAML configuration
        >>> model = YOLO("yolo26n.yaml")
    """

    def __init__(self, model: str | Path = "yolo26n.pt", task: str | None = None, verbose: bool = False):
        """Initialize a YOLO model.

        This constructor initializes a YOLO model, automatically switching to specialized model types (YOLOWorld or
        YOLOE) based on the model filename.

        Args:
            model (str | Path): Model name or path to model file, i.e. 'yolo26n.pt', 'yolo26n.yaml'.
            task (str, optional): YOLO task specification, i.e. 'detect', 'segment', 'classify', 'pose', 'obb'. Defaults
                to auto-detection based on model.
            verbose (bool): Display model info on load.
        """
        path = Path(model if isinstance(model, (str, Path)) else "")
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        elif "yoloe" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOE PyTorch model
            new_instance = YOLOE(path, task=task, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)
            if hasattr(self.model, "model") and "RTDETR" in self.model.model[-1]._get_name():  # if RTDETR head
                from ultralytics import RTDETR

                new_instance = RTDETR(self)
                self.__class__ = type(new_instance)
                self.__dict__ = new_instance.__dict__

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "classify": {
                "model": ClassificationModel,
                "trainer": yolo.classify.ClassificationTrainer,
                "validator": yolo.classify.ClassificationValidator,
                "predictor": yolo.classify.ClassificationPredictor,
            },
            "detect": {
                "model": DetectionModel,
                "trainer": yolo.detect.DetectionTrainer,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo.segment.SegmentationTrainer,
                "validator": yolo.segment.SegmentationValidator,
                "predictor": yolo.segment.SegmentationPredictor,
            },
            "pose": {
                "model": PoseModel,
                "trainer": yolo.pose.PoseTrainer,
                "validator": yolo.pose.PoseValidator,
                "predictor": yolo.pose.PosePredictor,
            },
            "obb": {
                "model": OBBModel,
                "trainer": yolo.obb.OBBTrainer,
                "validator": yolo.obb.OBBValidator,
                "predictor": yolo.obb.OBBPredictor,
            },
        }


class YOLOWorld(Model):
    """YOLO-World object detection model.

    YOLO-World is an open-vocabulary object detection model that can detect objects based on text descriptions without
    requiring training on specific classes. It extends the YOLO architecture to support real-time open-vocabulary
    detection.

    Attributes:
        model: The loaded YOLO-World model instance.
        task: Always set to 'detect' for object detection.
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOv8-World model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        set_classes: Set the model's class names for detection.

    Examples:
        Load a YOLOv8-World model
        >>> model = YOLOWorld("yolov8s-world.pt")

        Set custom classes for detection
        >>> model.set_classes(["person", "car", "bicycle"])
    """

    def __init__(self, model: str | Path = "yolov8s-world.pt", verbose: bool = False) -> None:
        """Initialize YOLOv8-World model with a pre-trained model file.

        Loads a YOLOv8-World model for object detection. If no custom class names are provided, it assigns default COCO
        class names.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = YAML.load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes: list[str]) -> None:
        """Set the model's class names for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = classes


class YOLOE(Model):
    """YOLOE object detection and segmentation model.

    YOLOE is an enhanced YOLO model that supports both object detection and instance segmentation tasks with improved
    performance and additional features like visual and text positional embeddings.

    Attributes:
        model: The loaded YOLOE model instance.
        task: The task type (detect or segment).
        overrides: Configuration overrides for the model.

    Methods:
        __init__: Initialize YOLOE model with a pre-trained model file.
        task_map: Map tasks to their corresponding model, trainer, validator, and predictor classes.
        get_text_pe: Get text positional embeddings for the given texts.
        get_visual_pe: Get visual positional embeddings for the given image and visual features.
        set_vocab: Set vocabulary and class names for the YOLOE model.
        get_vocab: Get vocabulary for the given class names.
        set_classes: Set the model's class names and embeddings for detection.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE detection model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Set vocabulary and class names
        >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])

        Predict with visual prompts
        >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
        >>> results = model.predict("image.jpg", visual_prompts=prompts)
    """

    def __init__(self, model: str | Path = "yoloe-11s-seg.pt", task: str | None = None, verbose: bool = False) -> None:
        """Initialize YOLOE model with a pre-trained model file.

        Args:
            model (str | Path): Path to the pre-trained model file. Supports *.pt and *.yaml formats.
            task (str, optional): Task type for the model. Auto-detected if None.
            verbose (bool): If True, prints additional information during initialization.
        """
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOEModel,
                "validator": yolo.yoloe.YOLOEDetectValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.yoloe.YOLOETrainer,
            },
            "segment": {
                "model": YOLOESegModel,
                "validator": yolo.yoloe.YOLOESegValidator,
                "predictor": yolo.segment.SegmentationPredictor,
                "trainer": yolo.yoloe.YOLOESegTrainer,
            },
        }

    def get_text_pe(self, texts):
        """Get text positional embeddings for the given texts."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_text_pe(texts)

    def get_visual_pe(self, img, visual):
        """Get visual positional embeddings for the given image and visual features.

        This method extracts positional embeddings from visual features based on the input image. It requires that the
        model is an instance of YOLOEModel.

        Args:
            img (torch.Tensor): Input image tensor.
            visual (torch.Tensor): Visual features extracted from the image.

        Returns:
            (torch.Tensor): Visual positional embeddings.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> img = torch.rand(1, 3, 640, 640)
            >>> visual_features = torch.rand(1, 1, 80, 80)
            >>> pe = model.get_visual_pe(img, visual_features)
        """
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_visual_pe(img, visual)

    def set_vocab(self, vocab: list[str], names: list[str]) -> None:
        """Set vocabulary and class names for the YOLOE model.

        This method configures the vocabulary and class names used by the model for text processing and classification
        tasks. The model must be an instance of YOLOEModel.

        Args:
            vocab (list[str]): Vocabulary list containing tokens or words used by the model for text processing.
            names (list[str]): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> model.set_vocab(["person", "car", "dog"], ["person", "car", "dog"])
        """
        assert isinstance(self.model, YOLOEModel)
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get vocabulary for the given class names."""
        assert isinstance(self.model, YOLOEModel)
        return self.model.get_vocab(names)

    def set_classes(self, classes: list[str], embeddings: torch.Tensor | None = None) -> None:
        """Set the model's class names and embeddings for detection.

        Args:
            classes (list[str]): A list of categories i.e. ["person"].
            embeddings (torch.Tensor, optional): Embeddings corresponding to the classes.
        """
        # Verify no background class is present
        assert " " not in classes
        assert isinstance(self.model, YOLOEModel)
        if sorted(list(self.model.names.values())) != sorted(classes):
            if embeddings is None:
                embeddings = self.get_text_pe(classes)  # generate text embeddings if not provided
            self.model.set_classes(classes, embeddings)

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = self.model.names

    def val(
        self,
        validator=None,
        load_vp: bool = False,
        refer_data: str | None = None,
        **kwargs,
    ):
        """Validate the model using text or visual prompts.

        Args:
            validator (callable, optional): A callable validator function. If None, a default validator is loaded.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.
            refer_data (str, optional): Path to the reference data for visual prompts.
            **kwargs (Any): Additional keyword arguments to override default settings.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        custom = {"rect": not load_vp}  # method defaults
        args = {**self.overrides, **custom, **kwargs, "mode": "val"}  # highest priority args on the right

        validator = (validator or self._smart_load("validator"))(args=args, _callbacks=self.callbacks)
        validator(model=self.model, load_vp=load_vp, refer_data=refer_data)
        self.metrics = validator.metrics
        return validator.metrics

    def predict(
        self,
        source=None,
        stream: bool = False,
        visual_prompts: dict[str, list] = {},
        refer_image=None,
        predictor=yolo.yoloe.YOLOEVPDetectPredictor,
        **kwargs,
    ):
        """Run prediction on images, videos, directories, streams, etc.

        Args:
            source (str | int | PIL.Image | np.ndarray, optional): Source for prediction. Accepts image paths, directory
                paths, URL/YouTube streams, PIL images, numpy arrays, or webcam indices.
            stream (bool): Whether to stream the prediction results. If True, results are yielded as a generator as they
                are computed.
            visual_prompts (dict[str, list]): Dictionary containing visual prompts for the model. Must include 'bboxes'
                and 'cls' keys when non-empty.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable): Custom predictor class for visual prompt predictions. Defaults to
                YOLOEVPDetectPredictor.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (list | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts
            >>> prompts = {"bboxes": [[10, 20, 100, 200]], "cls": ["person"]}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts)
        """
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            assert len(visual_prompts["bboxes"]) == len(visual_prompts["cls"]), (
                f"Expected equal number of bounding boxes and classes, but got {len(visual_prompts['bboxes'])} and "
                f"{len(visual_prompts['cls'])} respectively"
            )
            if type(self.predictor) is not predictor:
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": refer_image is None,
                        "batch": 1,
                        "device": kwargs.get("device", None),
                        "half": kwargs.get("half", False),
                        "imgsz": kwargs.get("imgsz", self.overrides.get("imgsz", 640)),
                    },
                    _callbacks=self.callbacks,
                )

            num_cls = (
                max(len(set(c)) for c in visual_prompts["cls"])
                if isinstance(source, list) and refer_image is None  # means multiple images
                else len(set(visual_prompts["cls"]))
            )
            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model)

            if refer_image is None and source is not None:
                dataset = load_inference_source(source)
                if dataset.mode in {"video", "stream"}:
                    # NOTE: set the first frame as refer image for videos/streams inference
                    refer_image = next(iter(dataset))[1][0]
            if refer_image is not None:
                vpe = self.predictor.get_vpe(refer_image)
                self.model.set_classes(self.model.names, vpe)
                self.task = "segment" if isinstance(self.predictor, yolo.segment.SegmentationPredictor) else "detect"
                self.predictor = None  # reset predictor
        elif isinstance(self.predictor, yolo.yoloe.YOLOEVPDetectPredictor):
            self.predictor = None  # reset predictor if no visual prompts
        self.overrides["agnostic_nms"] = True  # use agnostic nms for YOLOE default

        return super().predict(source, stream, **kwargs)




class AnomalyValidator(yolo.detect.DetectionValidator):
    """Validator for YOLOAnomaly models.

    For end2end models (e.g. yoloe26n), non_max_suppression skips IoU-based NMS
    and only confidence-filters the top-k output, leaving overlapping boxes.
    This subclass applies real torchvision IoU NMS after confidence filtering.

    On top of standard detection mAP, when the model is in ``feature_mode='fused_heatmap'``
    this validator also computes:

      * ``image_auroc`` — image-level AUROC, score=max(heatmap), label=1 if image has any GT box.
      * ``pixel_auroc`` — pixel-level AUROC, heatmap vs binary GT mask, anomaly images only.

    The pixel-level GT mask is rasterized from the YOLO segmentation polygons in the
    image's ``.txt`` label file (alongside the image, ``../labels/<stem>.txt`` if a
    ``labels/`` mirror directory exists). All pixels inside any instance polygon are
    labeled 1 (anomalous), all others 0 (normal). Image is "anomaly" if it has ≥1 GT box.
    AUROC is NaN if the model is not in heatmap mode or no GT polygons are found.
    """

    # Pixel-AUROC eval resolution (heatmap+GT both resized to this square before flattening).
    _pix_eval_size: int = 256

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.iouv = torch.tensor([0.1, 0.25] + self.iouv.tolist())
        self.niou = self.iouv.numel()

    def get_desc(self) -> str:
        return ("%22s" + "%11s" * 8) % (
            "Class", "Images", "Instances", "Box(P", "R", "mAP10", "mAP25", "mAP50", "mAP50-95)"
        )

    def init_metrics(self, model: torch.nn.Module) -> None:
        super().init_metrics(model)
        # Reset per-call AUROC accumulators.
        self._anomaly_mode = self.nc == 1  # True when nc=1 (anomaly mode), False in detect mode
        self._heatmaps_for_batch: torch.Tensor | None = None
        self._image_scores: list[float] = []
        self._image_labels: list[int] = []
        self._pixel_scores: list = []  # list[np.ndarray]
        self._pixel_labels: list = []
        self._n_anom_with_gt: int = 0
        self.image_auroc: float = float("nan")
        self.pixel_auroc: float = float("nan")

    def _prepare_batch(self, si: int, batch) -> dict:
        """Prepare batch; in anomaly mode (nc=1) map all GT class ids to 0.

        MVTec-YOLO GT labels carry per-defect-type class ids (0, 1, 2, …), but the
        anomaly head always outputs class 0.  Without this remap the strict
        class-equality check in match_predictions zeros out IoU for every GT box
        whose class id is > 0, making mAP meaninglessly low.
        """
        pbatch = super()._prepare_batch(si, batch)
        if self._anomaly_mode and pbatch["cls"].shape[0]:
            pbatch["cls"] = torch.zeros_like(pbatch["cls"])
        return pbatch

    def postprocess(self, preds):
        """Apply real IoU NMS for end2end models, then delegate. Also stash heatmap for AUROC."""
        # Capture raw heatmap and feature_mode from (y, preds_dict) tuple before unpacking.
        self._heatmaps_for_batch = None
        feature_mode = "per_level"
        ad_conf = 0.4
        if isinstance(preds, (tuple, list)) and len(preds) > 1 and isinstance(preds[1], dict):
            aux = preds[1]
            hm = aux.get("heatmap")
            feature_mode = aux.get("feature_mode", "per_level")
            ad_conf = aux.get("ad_conf", ad_conf)
            if hm is not None:
                self._heatmaps_for_batch = hm.detach().cpu()

        if isinstance(preds, (tuple, list)):
            preds = preds[0]

        if self.end2end:
            import torchvision

            output = []
            for pred in preds:  # pred: [N, 6] = [x1, y1, x2, y2, score, cls]
                mask = pred[:, 4] > self.args.conf
                pred = pred[mask]
                if len(pred) == 0:
                    output.append({"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5], "extra": pred[:, 6:]})
                    continue
                keep = torchvision.ops.nms(pred[:, :4].float(), pred[:, 4].float(), self.args.iou)
                pred = pred[keep[: self.args.max_det]]
                output.append({"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5], "extra": pred[:, 6:]})
        else:
            output = super().postprocess(preds)

        if feature_mode == "fused_heatmap" and self._heatmaps_for_batch is not None:
            output = self._heatmap_proposals(output, self._heatmaps_for_batch, self.args.imgsz, self.end2end, ad_conf=ad_conf)

        return output

    @staticmethod
    def _heatmap_proposals(output, heatmaps, imgsz, end2end, ad_conf=0.4, min_area=4):
        """Replace NMS predictions with connected-component bbox proposals from the fused heatmap.

        Pixels with heatmap score >= ad_conf form the foreground mask; connected components of
        that mask become bbox proposals.  Images where no pixel crosses ad_conf get no proposals.

        Args:
            output: List of per-image predictions (dicts for end2end, tensors [N,6] otherwise).
            heatmaps: [B, 1, H, W] fused heatmap tensor (CPU, float).
            imgsz: Inference image size (int, assumed square).
            end2end: Whether the model uses end2end (dict) output format.
            ad_conf: Absolute pixel threshold — only pixels >= ad_conf contribute to components.
            min_area: Minimum connected-component area in feature-map pixels to keep.
        """
        import cv2
        import numpy as np

        hm_h, hm_w = heatmaps.shape[-2], heatmaps.shape[-1]
        scale_h = imgsz / hm_h
        scale_w = imgsz / hm_w

        new_output = []
        for si in range(len(output)):
            hm = heatmaps[si]
            if hm.dim() == 3:
                hm = hm.squeeze(0)
            hm_np = hm.float().numpy()

            binary = (hm_np >= ad_conf).astype(np.uint8)
            if binary.sum() == 0:
                if end2end:
                    new_output.append({"bboxes": torch.zeros((0, 4)), "conf": torch.zeros(0),
                                       "cls": torch.zeros(0), "extra": torch.zeros((0, 0))})
                else:
                    new_output.append(torch.zeros((0, 6), dtype=torch.float32))
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            rows = []
            for j in range(1, num_labels):
                area = int(stats[j, cv2.CC_STAT_AREA])
                if area < min_area:
                    continue
                x1 = float(stats[j, cv2.CC_STAT_LEFT]) * scale_w
                y1 = float(stats[j, cv2.CC_STAT_TOP]) * scale_h
                x2 = float(stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH]) * scale_w
                y2 = float(stats[j, cv2.CC_STAT_TOP] + stats[j, cv2.CC_STAT_HEIGHT]) * scale_h
                score = float(hm_np[labels == j].max())
                rows.append([x1, y1, x2, y2, score, 0.0])

            if rows:
                boxes = torch.tensor(rows, dtype=torch.float32)
            else:
                boxes = torch.zeros((0, 6), dtype=torch.float32)

            if end2end:
                new_output.append({
                    "bboxes": boxes[:, :4],
                    "conf": boxes[:, 4],
                    "cls": boxes[:, 5],
                    "extra": boxes[:, 6:] if boxes.shape[1] > 6 else boxes.new_zeros(len(boxes), 0),
                })
            else:
                new_output.append(boxes)

        return new_output

    @staticmethod
    def _label_path(im_file: str) -> Path | None:
        """Resolve YOLO label .txt for an image. Tries ``labels/`` mirror first, then sibling."""
        p = Path(im_file)
        # Mirror layout: <root>/images/.../foo.png → <root>/labels/.../foo.txt
        for parts in [p.parts]:
            if "images" in parts:
                idx = parts.index("images")
                mirrored = Path(*parts[:idx], "labels", *parts[idx + 1:]).with_suffix(".txt")
                if mirrored.exists():
                    return mirrored
        # Sibling layout: foo.png → foo.txt next to it (MVTec-YOLO uses this).
        sibling = p.with_suffix(".txt")
        if sibling.exists():
            return sibling
        return None

    @classmethod
    def _gt_mask_from_polygons(cls, im_file: str, mask_size: int) -> "np.ndarray | None":
        """Rasterize all instance polygons from the YOLO label file into a binary mask.

        Returns ``None`` if no label file or no polygons are found.
        Output shape: (mask_size, mask_size), uint8, values in {0, 1}.
        """
        import cv2
        import numpy as np

        label_path = cls._label_path(im_file)
        if label_path is None:
            return None
        try:
            text = label_path.read_text().strip()
        except OSError:
            return None
        if not text:
            return None

        mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
        any_drawn = False
        for line in text.splitlines():
            tokens = line.strip().split()
            # YOLO seg polygon line: cls x1 y1 x2 y2 ... xn yn  (normalized, pairs of points)
            if len(tokens) < 7 or (len(tokens) - 1) % 2 != 0:
                continue
            try:
                coords = np.array(tokens[1:], dtype=np.float32)
            except ValueError:
                continue
            pts = coords.reshape(-1, 2) * mask_size  # to pixel coords on mask_size x mask_size
            pts_int = pts.astype(np.int32)
            cv2.fillPoly(mask, [pts_int], 1)
            any_drawn = True
        return mask if any_drawn else None

    def update_metrics(self, preds, batch) -> None:
        """Standard detection metrics + AUROC accumulation when heatmap is available."""
        super().update_metrics(preds, batch)
        if self._heatmaps_for_batch is None:
            return
        import cv2

        for si in range(len(preds)):
            hm = self._heatmaps_for_batch[si]
            if hm.dim() == 3:
                hm = hm.squeeze(0)
            hm_np = hm.float().numpy()

            self._image_scores.append(float(hm_np.max()))
            n_gt = int((batch["batch_idx"] == si).sum().item())
            label = 1 if n_gt > 0 else 0
            self._image_labels.append(label)

            if label == 1:
                gt_d = self._gt_mask_from_polygons(batch["im_file"][si], self._pix_eval_size)
                if gt_d is None:
                    continue
                hm_d = cv2.resize(hm_np, (self._pix_eval_size, self._pix_eval_size), interpolation=cv2.INTER_LINEAR)
                self._pixel_scores.append(hm_d.flatten())
                self._pixel_labels.append(gt_d.flatten())
                self._n_anom_with_gt += 1

    def finalize_metrics(self) -> None:
        super().finalize_metrics()
        if not self._image_scores:
            return  # no heatmaps captured — model wasn't in fused_heatmap mode
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            LOGGER.warning("sklearn not available — skipping anomaly AUROC computation")
            return
        import numpy as np

        if len(set(self._image_labels)) > 1:
            self.image_auroc = float(roc_auc_score(self._image_labels, self._image_scores))
        if self._pixel_scores:
            ps = np.concatenate(self._pixel_scores)
            pl = np.concatenate(self._pixel_labels)
            if pl.any() and not pl.all():
                self.pixel_auroc = float(roc_auc_score(pl, ps))

        # Surface AUROC on the metrics object so model.val()'s return carries them
        # (validator.metrics is what model.val() returns).
        self.metrics.image_auroc = self.image_auroc
        self.metrics.pixel_auroc = self.pixel_auroc

        # ── TEMP: optimal-F1 threshold ──────────────────────────────────────
        self.metrics.opt_f1_threshold = self._compute_opt_f1_threshold(
            self._pixel_scores, self._pixel_labels
        )
        # ────────────────────────────────────────────────────────────────────

    # ── TEMP: optimal-F1 threshold ──────────────────────────────────────────
    @staticmethod
    def _compute_opt_f1_threshold(pixel_scores: list, pixel_labels: list) -> float:
        """Return the pixel-level heatmap threshold that maximises F1 over the val set."""
        if not pixel_scores or not pixel_labels:
            return float("nan")
        import numpy as np
        from sklearn.metrics import precision_recall_curve

        scores = np.concatenate(pixel_scores)
        labels = np.concatenate(pixel_labels)
        if not labels.any() or labels.all():
            return float("nan")
        precision, recall, thresholds = precision_recall_curve(labels, scores)
        # precision/recall have length n+1; thresholds has length n
        f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-8)
        return float(thresholds[int(np.argmax(f1))])
    # ────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.clear_stats()
        box = self.metrics.box
        all_ap = box.all_ap
        if len(all_ap):
            vals = [box.mp, box.mr, all_ap[:, 0].mean(), all_ap[:, 1].mean(), all_ap[:, 2].mean(), all_ap[:, 2:].mean()]
        else:
            vals = [0.0] * 6
        stats = {
            "metrics/precision(B)": float(vals[0]),
            "metrics/recall(B)": float(vals[1]),
            "metrics/mAP10(B)": float(vals[2]),
            "metrics/mAP25(B)": float(vals[3]),
            "metrics/mAP50(B)": float(vals[4]),
            "metrics/mAP50-95(B)": float(vals[5]),
            "fitness": float(vals[5]),
            "metrics/image_auroc": self.image_auroc,
            "metrics/pixel_auroc": self.pixel_auroc,
        }
        # attach to metrics object so model.val() return carries them
        self.metrics.map10 = stats["metrics/mAP10(B)"]
        self.metrics.map25 = stats["metrics/mAP25(B)"]
        self.metrics.map50 = stats["metrics/mAP50(B)"]
        return stats

    def print_results(self) -> None:
        box = self.metrics.box
        all_ap = box.all_ap
        has_ap = len(all_ap) > 0
        if has_ap:
            mean_vals = [box.mp, box.mr, all_ap[:, 0].mean(), all_ap[:, 1].mean(), all_ap[:, 2].mean(), all_ap[:, 2:].mean()]
        else:
            mean_vals = [0.0] * 6

        auroc_suffix = ""
        if self._image_scores:
            auroc_suffix = f"   img_auroc: {self.image_auroc:.4f}  pix_auroc: {self.pixel_auroc:.4f}"

        pf = "%22s" + "%11i" * 2 + "%11.3g" * 6
        LOGGER.info(pf % ("all", self.seen, self.metrics.nt_per_class.sum(), *mean_vals) + auroc_suffix)
        if self.metrics.nt_per_class.sum() == 0:
            LOGGER.warning(f"no labels found in {self.args.task} set, cannot compute metrics without labels")

        if self.args.verbose and not self.training and self.nc > 1 and len(self.metrics.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                if has_ap:
                    cls_vals = [box.p[i], box.r[i], all_ap[i, 0], all_ap[i, 1], all_ap[i, 2], all_ap[i, 2:].mean()]
                else:
                    cls_vals = [0.0] * 6
                LOGGER.info(pf % (self.names[c], self.metrics.nt_per_image[c], self.metrics.nt_per_class[c], *cls_vals))


class AnomalyPredictor(yolo.detect.DetectionPredictor):
    """Predictor for YOLOAnomaly models.

    Handles the (y, preds_dict) tuple that AnomalyDetection.forward() returns in
    non-export mode: extracts the tensor `y` before passing it to NMS / postprocess.
    """

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        """Unpack model output tuple, run bbox NMS, and attach heatmap to each Result.

        When ``feature_mode='fused_heatmap'`` is on, the AnomalyDetection forward returns
        ``(y_tensor, preds_dict)`` where ``preds_dict['heatmap']`` is the fused heatmap.
        We always run the standard bbox postprocess (so detection mAP still works) and
        attach the upsampled heatmap to each Result as ``.heatmap`` / ``.heatmap_conf``.
        """
        aux_preds = None
        if isinstance(preds, (tuple, list)):
            aux_preds = preds[1] if len(preds) > 1 and isinstance(preds[1], dict) else None
            preds = preds[0]

        # Standard bbox postprocess path (per-level adhead → boxes).
        if getattr(self.model, "end2end", False):
            import torchvision

            output = []
            for pred in preds:  # pred: [N, 6] = [x1, y1, x2, y2, score, cls]
                mask = pred[:, 4] > self.args.conf
                pred = pred[mask]
                if len(pred) == 0:
                    output.append(pred)
                    continue
                keep = torchvision.ops.nms(pred[:, :4].float(), pred[:, 4].float(), self.args.iou)
                output.append(pred[keep[: self.args.max_det]])
            results = self.construct_results(output, img, orig_imgs, **kwargs)
        else:
            results = super().postprocess(preds, img, orig_imgs, **kwargs)

        # Attach fused heatmap to each Result (if AnomalyDetection produced one).
        heatmap_tensor = aux_preds.get("heatmap") if isinstance(aux_preds, dict) else None
        if heatmap_tensor is not None:
            self._attach_heatmaps(results, heatmap_tensor)

        # In fused_heatmap mode: override boxes with connected-component proposals derived
        # from the full-resolution heatmap (already upsampled in _attach_heatmaps).
        if isinstance(aux_preds, dict) and aux_preds.get("feature_mode") == "fused_heatmap":
            ad_conf = aux_preds.get("ad_conf", 0.4)
            self._inject_component_boxes(results, ad_conf=ad_conf)

        return results

    @staticmethod
    def _inject_component_boxes(results: list, ad_conf: float = 0.4, min_area: int = 100) -> None:
        """Replace empty per-level boxes with connected-component proposals from res.heatmap.

        Operates on the full-resolution heatmap already attached by _attach_heatmaps, so box
        coordinates are in original image pixel space.  Pixels below ``ad_conf`` are treated as
        background; images with no pixels >= ad_conf get no proposals.
        """
        import cv2
        import numpy as np
        from ultralytics.engine.results import Boxes

        for res in results:
            hm = getattr(res, "heatmap", None)
            if hm is None:
                continue
            hm_np = hm.float().cpu().numpy()

            binary = (hm_np >= ad_conf).astype(np.uint8)
            if binary.sum() == 0:
                res.boxes = Boxes(torch.zeros((0, 6), dtype=torch.float32), res.orig_img.shape[:2])
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

            rows = []
            for j in range(1, num_labels):
                if int(stats[j, cv2.CC_STAT_AREA]) < min_area:
                    continue
                x1 = float(stats[j, cv2.CC_STAT_LEFT])
                y1 = float(stats[j, cv2.CC_STAT_TOP])
                x2 = float(stats[j, cv2.CC_STAT_LEFT] + stats[j, cv2.CC_STAT_WIDTH])
                y2 = float(stats[j, cv2.CC_STAT_TOP] + stats[j, cv2.CC_STAT_HEIGHT])
                score = float(hm_np[labels == j].max())
                rows.append([x1, y1, x2, y2, score, 0.0])

            boxes_t = torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 6), dtype=torch.float32)
            res.boxes = Boxes(boxes_t, res.orig_img.shape[:2])

    @staticmethod
    def _attach_heatmaps(results: list, heatmaps: torch.Tensor) -> None:
        """Bilinear-upsample each per-image heatmap to its orig shape and attach to the Result."""
        for i, res in enumerate(results):
            hm = heatmaps[i]
            if hm.dim() == 3:
                hm = hm.squeeze(0)
            full = torch.nn.functional.interpolate(
                hm[None, None].float(),
                size=res.orig_img.shape[:2],
                mode="bilinear",
                align_corners=False,
            )[0, 0]
            if full.min() < 0 or full.max() > 1:
                full = full.sigmoid()
            full = full.clamp_(0, 1)
            res.heatmap = full
            res.heatmap_conf = float(full.max())


class YOLOAnomaly(Model):
    """
    YOLO-based training-free anomaly detection model.

    Loads any YOLOE-compatible pretrained model and converts it into anomaly detection
    mode using a memory bank of normal feature representations. No gradient-based
    training is needed: feed normal images via load_support_set() to populate the bank,
    then call predict().

    Attributes:
        model: The underlying YOLOAnomalyModel instance.

    Methods:
        __init__: Initialize from any YOLOE pretrained model file.
        task_map: Map tasks to model, validator, and predictor classes.
        setup: Configure anomaly detection with class names and threshold.
        load_support_set: Feed normal images to build the memory bank.
        save_mb: Save memory-bank payload to disk.
        load_mb: Load memory-bank payload from disk.
        reset_memory_bank: Clear the memory bank for reuse with a new support set.
        get_memory_bank_stats: Return memory bank statistics per detection head.

    Examples:
        One-shot anomaly detection workflow
        >>> model = YOLOAnomaly("yolo26s.pt")
        >>> model.setup(["defect"], conf=0.1)
        >>> model.load_support_set("datasets/mvtec/leather/train/good/")
        >>> results = model.predict("datasets/mvtec/leather/test/crack/")
    """

    def __init__(self, model: str | Path = "yoloe-11s.pt", verbose: bool = False) -> None:
        """
        Initialize YOLOAnomaly from a pretrained model file.

        Loads the checkpoint and automatically upgrades the underlying YOLOEModel to
        YOLOAnomalyModel to enable memory bank methods. If the loaded checkpoint was
        previously saved with anomaly metadata (via save()), the model is restored
        ready to predict without calling setup() or load_mb().

        Args:
            model (str | Path): Path to pretrained model (*.pt), e.g. 'yolo26s.pt'.
            verbose (bool): Print model info on load.

        Raises:
            AssertionError: If the loaded model is not a YOLOEModel instance.
        """
        super().__init__(model=model, task=None, verbose=verbose)
        if not isinstance(self.model, YOLOAnomalyModel):
            if isinstance(self.model, DetectionModel):
                self.model.__class__ = YOLOAnomalyModel
            else:
                raise AssertionError(
                    f"YOLOAnomaly requires a DetectionModel or YOLOEModel checkpoint, "
                    f"but loaded {type(self.model).__name__}."
                )

        # Restore anomaly metadata from checkpoint if previously saved via save()
        self._restore_anomaly_metadata()

    @property
    def task_map(self) -> dict[str, dict[str, Any]]:
        """Map tasks to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "predictor": AnomalyPredictor,
                "validator": AnomalyValidator,
            },
            # Segmentation checkpoints are supported as backbones; anomaly output is
            # always detection-shaped (boxes only), so we reuse AnomalyPredictor.
            "segment": {
                "model": SegmentationModel,
                "predictor": AnomalyPredictor,
                "validator": AnomalyValidator,
            },
        }

    def setup(self, names: list[str]) -> None:
        """
        Configure anomaly detection with class names.

        Must be called before load_support_set() and predict().

        Pass ``["anomaly"]`` (or any custom single name) to enable memory-bank cosine-similarity
        scoring (nc=1).  Pass ``["detect"]`` to use the model's original classification head
        scores with the original class names — useful for baseline comparison.

        Use set_anomaly_args() to configure the detection threshold (ad_conf) and max detections
        (ad_max_det) independently before running predict().

        Args:
            names (list[str]): Anomaly class names, e.g. ["anomaly"] or ["defect", "scratch"].
                Use ["detect"] as a special sentinel to start in original-classifier mode.
        """
        assert isinstance(self.model, YOLOAnomalyModel), (
            f"Expected YOLOAnomalyModel, got {type(self.model).__name__}. "
            "Ensure you loaded a YOLOE or plain YOLO detection model."
        )
        detect_mode = names == ["detect"]
        if detect_mode:
            # Use the model's current (original) class names so vocab embeddings are correct
            init_names = list(self.model.names.values())
        else:
            init_names = names
        self.model.setup_anomaly_detection(init_names)
        if detect_mode:
            self.model.set_anomaly_mode(False)  # confidence = original head scores
        # names are already set correctly inside setup_anomaly_detection / set_anomaly_mode

    def load_support_set(
        self,
        source,
        conf: float = 1e-6,
        imgsz: int = 640,
        device=None,
        verbose: bool = True,
        batch: int = 1,
        **kwargs,
    ) -> list[dict]:
        """Feed normal (non-anomalous) images to populate the memory bank.

        Memory bank is automatically frozen after this call. Run once before predict().
        Subclasses can override individual pipeline stages without touching this method:

            _prepare_support_session()   – one-time setup (enable YOLO memory updates)
            _iter_support_sources()      – normalise source → iterator of image paths
            _extract_support_features()  – Stage 1: backbone feature extraction per image
            _accumulate_features()       – Stage 2: filter + push into bank
            _finalize_memory_bank()      – Stage 3: compact (coreset/PCA/no-op) + freeze

        Args:
            source: Image source — file path, directory, list of paths, etc.
            conf (float): Very low confidence to capture all candidate regions.
            imgsz (int): Inference image size.
            device: Device to run on (e.g. 'cuda:0', 'cpu').
            verbose (bool): Print memory bank stats after building.
            batch (int): Number of images per backbone forward.  ``batch=1`` is
                the original strictly-serial behaviour.  ``batch>1`` groups
                support images into chunks for a single backbone call, which
                is much faster for AnomalyDINO and for YOLOAnomaly in its
                coreset fast-path (``max_bank_size`` set).  OBMA gating in
                YOLOAnomaly is per-image, so for the OBMA path stick with
                ``batch=1`` to keep its semantics.
            **kwargs: Additional keyword arguments forwarded to predict().

        Returns:
            list[dict]: Memory bank statistics per detection head.

        Examples:
            >>> model.load_support_set("datasets/mvtec/leather/train/good/")
        """
        from ultralytics.utils import LOGGER, TQDM

        if verbose:
            LOGGER.info("YOLOAnomaly: building memory bank from support set...")
        self._prepare_support_session(conf=conf, imgsz=imgsz, device=device,
                                      batch=batch, **kwargs)
        items = list(self._iter_support_sources(source))
        batch = max(1, int(batch))
        chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
        pbar = TQDM(chunks, desc="Building memory bank", total=len(items))
        for chunk in pbar:
            feats = self._extract_support_features(chunk)
            self._accumulate_features(feats)
            pbar.update(len(chunk) - 1)   # TQDM iterates chunks; each chunk = len(chunk) items
            sizes, temps = self._head_stats()
            if sizes:
                pbar.set_postfix(
                    bank="/".join(str(s) for s in sizes),
                    T="/".join(f"{t:.2f}" for t in temps),
                )
        return self._finalize_memory_bank(verbose=verbose)

    def _head_stats(self) -> tuple[list[int], list[float]]:
        """Return ``(bank_sizes, temperatures)`` for each AD head (empty if N/A)."""
        try:
            heads = list(self.model._get_ad_heads())
            sizes = [int(h.memory_bank.shape[0]) for h in heads]
            temps = [float(getattr(h, "temperature", float("nan"))) for h in heads]
            return sizes, temps
        except (AttributeError, TypeError):
            return [], []

    # ── pipeline stage hooks ────────────────────────────────────────────────

    def _prepare_support_session(self, conf: float = 1e-6, imgsz: int = 640,
                                 device=None, batch: int = 1, **kwargs) -> None:
        """Stage 0: one-time setup before iterating support images.

        Stores predict kwargs and enables YOLO memory-bank accumulation.
        Override to skip the YOLOAnomalyModel assertion (e.g. AnomalyDINO).
        """
        assert isinstance(self.model, YOLOAnomalyModel), "Call setup() before load_support_set()."
        self._support_conf   = conf
        self._support_imgsz  = imgsz
        self._support_device = device
        self._support_batch  = batch
        self._support_kw     = kwargs
        self.model.set_memory_update(True)

    @staticmethod
    def _iter_support_sources(source):
        """Yield image paths one-by-one from a directory, list, or single path.

        Args:
            source: Directory path, list/tuple of paths, or a single image path/tensor.

        Yields:
            str | Path: Individual image paths (or the original item if non-path).
        """
        if isinstance(source, (list, tuple, set)):
            yield from source
            return
        path = Path(source) if isinstance(source, (str, Path)) else None
        if path and path.is_dir():
            exts = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp", ".pfm"}
            for item in sorted(path.iterdir()):
                if item.is_file() and item.suffix.lower() in exts:
                    yield str(item)
            return
        yield source

    def _extract_support_features(self, items) -> None:
        """Stage 1: run backbone on a chunk of support images, accumulate features.

        Args:
            items: A single image path/tensor, or a list of them.  When a list
                is passed, ``model.predict`` runs them as one batched forward
                so ``ADMBHead.forward`` sees ``B = len(items)``.  The internal
                accumulation logic (OBMA / coreset fast-path) already handles
                ``B > 1``.

        Override to return explicit feature tensors (e.g. AnomalyDINO returns a
        (N, D) patch-token tensor from DINOv2).
        """
        source = items if isinstance(items, (list, tuple)) else [items]
        self.predict(
            source=source,
            conf=self._support_conf,
            imgsz=self._support_imgsz,
            device=self._support_device,
            batch=len(source),
            verbose=False,
            **self._support_kw,
        )
        return None  # YOLO: ADMBHead accumulated internally

    def _accumulate_features(self, feats) -> None:
        """Stage 2: push extracted features into the memory bank.

        Default: no-op — the YOLO path accumulates inside ADMBHead.forward()
        during _extract_support_features().

        Override to maintain an explicit bank tensor (e.g. AnomalyDINO cats
        (N, D) patch-token tensors into self._dino_bank).
        """

    def _compact_memory_bank(self) -> None:
        """Stage 3a: reduce bank size before freezing.

        Default: delegates to YOLOAnomalyModel.freeze_memory_bank() which runs
        k-center coreset compression if max_bank_size is set on the head, then
        disables further accumulation.

        Override for alternative compaction (PCA projection, random subsample, etc.)
        or to compact a custom bank tensor (e.g. AnomalyDINO).
        """
        self.model.freeze_memory_bank()

    def _finalize_memory_bank(self, verbose: bool = True) -> list[dict]:
        """Stage 3b: compact then return per-head stats.

        Args:
            verbose (bool): Log per-head size and feature dimension.

        Returns:
            list[dict]: Each dict has 'size' and 'feature_dim' keys.
        """
        from ultralytics.utils import LOGGER
        self._compact_memory_bank()
        stats = self.model.get_memory_bank_stats()
        if verbose:
            for i, s in enumerate(stats):
                LOGGER.info(f"  Head[{i}]: {s['size']} features, dim={s['feature_dim']}")
        return stats

    # ── shared scoring utilities (usable by subclasses) ────────────────────

    @staticmethod
    def _compute_knn_distances(
        q: "torch.Tensor",
        bank: "torch.Tensor",
        K: "int | list[int]",
    ) -> "torch.Tensor":
        """Compute cosine-distance KNN scores for query features against a memory bank.

        Both *q* and *bank* must be **L2-normalised** before calling so that the
        dot product equals the cosine similarity.

        Args:
            q (Tensor): Query features, shape (N, D).
            bank (Tensor): Normal-feature memory bank, shape (M, D).
            K (int | list[int]): Number of nearest neighbours.  When a list is
                provided, KNN is run independently for each k-value and the
                resulting distances are averaged — useful for ensemble scoring
                without extra bank lookups.

        Returns:
            Tensor: Cosine distances (1 − mean_top_k_cosine_sim), shape (N,).
        """
        import torch
        Ks = [K] if isinstance(K, int) else list(K)
        sims = q @ bank.T  # (N, M)
        dists_list = []
        for k in Ks:
            k = min(k, sims.size(1))
            topk = sims.topk(k, dim=-1).values  # (N, k)
            dists_list.append(1.0 - topk.mean(dim=-1))  # (N,)
        return torch.stack(dists_list).mean(0)  # average over K values

    def _aggregate_scores(
        self,
        patch_scores: "torch.Tensor",
        method: "str | None" = None,
    ) -> "torch.Tensor":
        """Reduce per-patch anomaly scores to a single image-level scalar.

        Supported methods (set via set_anomaly_args(score_aggregation=...)):

        * ``"max"``       – spatial max-pool (default, matches current behaviour).
        * ``"mean"``      – spatial mean; less sensitive to isolated hot pixels.
        * ``"topk_mean"`` – mean of top-10 patches; a soft-max robust to outliers.
        * ``"noise_or"``  – Noisy-OR: ``1 − ∏(1 − pᵢ)`` over all patches;
                            most sensitive to defects spread across multiple patches.

        Args:
            patch_scores (Tensor): Per-patch scores in [0, 1], shape (N,).
            method (str | None): Override the stored score_aggregation setting.

        Returns:
            Tensor: Scalar image-level anomaly score.
        """
        import torch
        method = method or self.anomaly_args.get("score_aggregation", "max")
        if method == "noise_or":
            return 1.0 - (1.0 - patch_scores.clamp(0.0, 1.0)).prod()
        if method == "topk_mean":
            k = min(10, patch_scores.numel())
            return patch_scores.topk(k).values.mean()
        if method == "mean":
            return patch_scores.mean()
        return patch_scores.max()  # "max" — default

    @staticmethod
    def _heatmap_to_boxes(
        heatmap: "torch.Tensor",
        thresh: float = 0.5,
        max_det: int = 9,
        min_area: int = 64,
    ) -> "torch.Tensor":
        """Threshold a spatial heatmap and fit bounding boxes via connected components.

        Args:
            heatmap (Tensor): (H, W) float tensor, values in [0, 1].
            thresh (float): Score threshold for foreground pixels.
            max_det (int): Maximum number of boxes returned.
            min_area (int): Minimum connected-component area (pixels) to keep.

        Returns:
            Tensor: Shape (N, 6) — ``[x1, y1, x2, y2, score, class_id=0]``,
                sorted by score descending.  Returns empty (0, 6) when no
                component passes the threshold or min_area filter.
        """
        import cv2
        import numpy as np
        import torch

        h_np = heatmap.detach().cpu().float().numpy()
        mask = (h_np >= thresh).astype(np.uint8)
        if mask.sum() == 0:
            return torch.zeros((0, 6), dtype=torch.float32)
        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        H, W = h_np.shape
        boxes = []
        for lbl in range(1, num):  # skip background label 0
            x, y, w, h, area = stats[lbl]
            if area < min_area:
                continue
            # Skip components that touch the image border (resize / padding artifacts).
            if x == 0 or y == 0 or (x + w) >= W or (y + h) >= H:
                continue
            score = float(h_np[labels == lbl].mean())
            boxes.append([float(x), float(y), float(x + w), float(y + h), score, 0.0])
        if not boxes:
            return torch.zeros((0, 6), dtype=torch.float32)
        t = torch.tensor(boxes, dtype=torch.float32)
        order = t[:, 4].argsort(descending=True)[:max_det]
        return t[order]

    def reset_memory_bank(self) -> None:
        """
        Clear the memory bank to allow rebuilding with a different support set.

        Does not require reloading the model.
        """
        assert isinstance(self.model, YOLOAnomalyModel)
        self.model.reset_memory_bank()

    def save_mb(self, path: str | Path) -> Path:
        """Save anomaly memory-bank payload for fast restore without rebuilding support set.

        Args:
            path (str | Path): Destination file path.

        Returns:
            Path: Saved file path.
        """
        assert isinstance(self.model, YOLOAnomalyModel), "Call setup() before save_mb()."
        heads_payload = []
        for h in self.model._get_ad_heads():
            heads_payload.append(
                {
                    "memory_bank": h.memory_bank.detach().cpu(),
                    "feature_dim": int(h.feature_dim) if h.feature_dim is not None else None,
                }
            )

        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "admb_cache_v1",
            "heads": heads_payload,
            "num_heads": len(heads_payload),
            "names": dict(self.model.names) if getattr(self.model, "names", None) else None,
        }
        torch.save(payload, save_path)
        return save_path

    def load_mb(self, path: str | Path, freeze: bool = True, verbose: bool = True) -> list[dict]:
        """Load anomaly memory-bank payload previously saved by save_mb().

        Args:
            path (str | Path): Cache file path.
            freeze (bool): Freeze memory-bank updates after loading.
            verbose (bool): Print loaded stats.

        Returns:
            list[dict]: Per-head memory stats after loading.

        Raises:
            FileNotFoundError: If path does not exist.
            ValueError: If cache format is invalid or incompatible.
        """
        from ultralytics.utils import LOGGER

        assert isinstance(self.model, YOLOAnomalyModel), "Call setup() before load_mb()."
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Memory-bank cache not found: {load_path}")

        data = torch.load(load_path, map_location="cpu")
        if not isinstance(data, dict) or data.get("format") != "admb_cache_v1" or "heads" not in data:
            raise ValueError(f"Invalid memory-bank cache format in {load_path}")

        heads = self.model._get_ad_heads()
        if len(data["heads"]) != len(heads):
            raise ValueError(
                f"Memory-bank head count mismatch: cache={len(data['heads'])}, model={len(heads)}"
            )

        for h, hdata in zip(heads, data["heads"]):
            mb = hdata.get("memory_bank", None)
            if not isinstance(mb, torch.Tensor) or mb.dim() != 2:
                raise ValueError("Invalid memory_bank tensor in cache payload.")
            h.memory_bank = mb.to(h.memory_bank.device, dtype=h.memory_bank.dtype)
            fd = hdata.get("feature_dim", None)
            h.feature_dim = int(fd) if fd is not None else int(h.memory_bank.shape[1])

        if freeze:
            self.model.freeze_memory_bank()

        stats = self.model.get_memory_bank_stats()
        if verbose:
            LOGGER.info(f"YOLOAnomaly: loaded memory bank cache from {load_path}")
            for i, s in enumerate(stats):
                LOGGER.info(f"  Head[{i}]: {s['size']} features, dim={s['feature_dim']}")
        return stats

    def get_memory_bank_stats(self) -> list[dict]:
        """
        Return memory bank statistics for all detection heads.

        Returns:
            list[dict]: Per-head stats with keys 'size' and 'feature_dim'.
        """
        assert isinstance(self.model, YOLOAnomalyModel)
        return self.model.get_memory_bank_stats()

    def set_anomaly_args(
        self,
        ad_conf: float | None = None,
        ad_max_det: int | None = None,
        mode: str | None = None,
        accumulate_thresh: float | None = None,
        score_filter_kernel: int | None = None,
        active_layers: list[int] | None = None,
        auto_temperature: bool | None = None,
        calibration_interval: int | None = None,
        em_iters: int | None = None,
        calibration_target_score: float | None = None,
        min_calibration_bank_size: int | None = None,
        temperature: float | None = None,
        K: int | list[int] | None = None,
        feature_mode: str | None = None,
        return_heatmap: bool | None = None,
        heatmap_logits: bool | None = None,
        fused_layers: list[int] | None = None,
        fused_use_pre_clshead: bool | None = None,
        max_bank_size: int | None = None,
        score_aggregation: str | None = None,
        yolo_weight: float | None = None,
    ) -> None:
        """Set anomaly-detection inference parameters and optionally switch mode.

        Args:
            ad_conf (float | None): Confidence threshold for anomaly proposals.
            ad_max_det (int | None): Maximum number of detections per image.
            mode (str | None): If provided, switch to this mode ('anomaly' or 'detect').
            accumulate_thresh (float | None): Score threshold for OBMA memory-bank accumulation.
            score_filter_kernel (int | None): Kernel size for spatial mean filter (1=off, 3/5/7=smoothing).
            active_layers (list[int] | None): Indices of detection layers to enable, e.g. [0, 1] uses only
                                              the first two (P3+P4). None = all layers enabled.
            auto_temperature (bool | None): Enable/disable automatic temperature calibration from data.
            calibration_interval (int | None): Re-calibrate every N support images (0 = calibrate once only).
            em_iters (int | None): EM iteration rounds during bank construction (1=single pass, default).
            calibration_target_score (float | None): Desired anomaly score for typical normal features.
            feature_mode (str | None): Feature scoring path, ``per_level`` or ``fused_heatmap``.
            return_heatmap (bool | None): Attach fused heatmap to the auxiliary prediction dict.
            heatmap_logits (bool | None): Return fused heatmap in logits instead of probabilities.
            max_bank_size (int | None): If set, use coreset fast-path during bank build and compress
                                        bank to this size on freeze. None = normal OBMA (no size cap).
        """
        assert isinstance(self.model, YOLOAnomalyModel), "Call setup() before set_anomaly_args()."
        head = self.model.model[-1]
        assert isinstance(head, AnomalyDetection), "Call setup() before set_anomaly_args()."
        # Build kwargs dict for head.set_anomaly_args (excludes special non-dict params).
        kwargs = {k: v for k, v in {
            "ad_conf": ad_conf,
            "ad_max_det": ad_max_det,
            "accumulate_thresh": accumulate_thresh,
            "score_filter_kernel": score_filter_kernel,
            "auto_temperature": auto_temperature,
            "calibration_interval": calibration_interval,
            "em_iters": em_iters,
            "calibration_target_score": calibration_target_score,
            "min_calibration_bank_size": min_calibration_bank_size,
            "temperature": temperature,
            "K": K,
            "feature_mode": feature_mode,
            "return_heatmap": return_heatmap,
            "heatmap_logits": heatmap_logits,
            "fused_layers": fused_layers,
            "fused_use_pre_clshead": fused_use_pre_clshead,
            "max_bank_size": max_bank_size,
            "score_aggregation": score_aggregation,
            "yolo_weight": yolo_weight,
        }.items() if v is not None}
        head.set_anomaly_args(active_layers=active_layers, mode=mode, **kwargs)

    @property
    def anomaly_args(self) -> dict:
        """Return the current anomaly_args dict from the head for quick inspection."""
        head = self.model.model[-1]
        if isinstance(head, AnomalyDetection):
            return head.anomaly_args
        return {}

    def set_mode(self, mode: str) -> None:
        """
        Switch between anomaly detection and original classification mode.

        In 'anomaly' mode the model outputs a single anomaly score per region based
        on cosine distance to the memory bank (nc=1, ignores original class labels).
        In 'detect' mode the original classification head is restored so the model
        behaves as a standard detector — useful when the loaded weights already
        target specific defect classes.

        Call setup() before set_mode().

        Args:
            mode (str): 'anomaly' for memory-bank scoring, 'detect' for original classes.

        Examples:
            >>> model.set_mode("detect")   # use original defect class outputs
            >>> model.set_mode("anomaly")  # switch back to memory-bank scoring
        """
        assert mode in ("anomaly", "detect"), f"mode must be 'anomaly' or 'detect', got {mode!r}"
        assert isinstance(self.model, YOLOAnomalyModel), (
            "Call setup() before set_mode()."
        )
        self.model.set_anomaly_mode(mode == "anomaly")
        # Propagate updated names to the predictor's AutoBackend if already initialized,
        # so Results objects created on the next predict() use the correct class names.
        if self.predictor is not None and getattr(self.predictor, "model", None) is not None:
            self.predictor.model.names = self.model.names

    def predict(self, source=None, stream: bool = False, **kwargs):
        """
        Run anomaly detection on the given source.

        Detections are regions whose anomaly score exceeds the configured threshold.
        Ensure setup() and load_support_set() have been called beforehand,
        or load a previously saved anomaly model (.pt with embedded memory bank).

        Args:
            source: Image source for inference.
            stream (bool): Yield results as a generator instead of a list.
            **kwargs: Additional keyword arguments passed to the predictor.

        Returns:
            list[Results] | generator: Anomaly detection results.
        """
        return super().predict(source=source, stream=stream, **kwargs)

    @property
    def is_configured(self) -> bool:
        """Check if the model is fully configured for anomaly detection (head + memory bank populated)."""
        head = self.model.model[-1]
        if not isinstance(head, AnomalyDetection) or head.adhead is None:
            return False
        # Check at least one head has real features (beyond the 10-entry padding from _memory_tensor)
        for h in head.iter_ad_heads(include_fused=True):
            if h.memory_bank.numel() > 0 and h.memory_bank.shape[0] > 10:
                return True
        return False

    def get_last_heatmap(self) -> torch.Tensor | None:
        """Return the most recent fused anomaly heatmap produced by the head, if available."""
        head = self.model.model[-1]
        if not isinstance(head, AnomalyDetection):
            return None
        return head.last_heatmap

    def save(self, filename: str | Path = "saved_model.pt") -> None:
        """Save the anomaly model with memory bank and metadata embedded in the checkpoint.

        The saved .pt file can be loaded directly with ``YOLOAnomaly(path)`` and used
        for prediction without calling ``setup()`` or ``load_mb()``.

        Args:
            filename (str | Path): Destination file path.
        """
        from copy import deepcopy
        from datetime import datetime
        from ultralytics import __version__

        self._check_is_pytorch_model()

        # Build anomaly metadata to persist alongside the model
        head = self.model.model[-1]
        anomaly_meta = {}
        if isinstance(head, AnomalyDetection) and head.adhead is not None:
            anomaly_meta = {
                "anomaly_names": getattr(self.model, "_anomaly_names", dict(self.model.names)),
                "original_names": getattr(self.model, "_original_names", {}),
                "original_nc": getattr(self.model, "_original_nc", getattr(head, "original_nc", head.nc)),
                "anomaly_args": dict(head.anomaly_args),
            }

        updates = {
            "model": deepcopy(self.model).half() if isinstance(self.model, torch.nn.Module) else self.model,
            "date": datetime.now().isoformat(),
            "version": __version__,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "anomaly_meta": anomaly_meta,
        }
        torch.save({**self.ckpt, **updates}, filename)

    def _restore_anomaly_metadata(self) -> None:
        """Restore anomaly detection state from checkpoint metadata if available.

        Called automatically by __init__ when loading a previously saved anomaly model.
        If the checkpoint contains anomaly_meta and the model head is already an
        AnomalyDetection with populated adhead, the model is ready to predict immediately.
        """
        from ultralytics.utils import LOGGER

        ckpt = self.ckpt
        if not ckpt or not isinstance(ckpt, dict):
            return

        meta = ckpt.get("anomaly_meta")
        head = self.model.model[-1]
        is_ad_head = isinstance(head, AnomalyDetection) and head.adhead is not None

        if not is_ad_head and meta is None:
            return  # Fresh model, no anomaly state to restore

        if is_ad_head:
            # Model was saved with AnomalyDetection head — restore metadata
            if meta:
                self.model._anomaly_names = meta.get("anomaly_names", {0: "anomaly"})
                self.model._original_names = meta.get("original_names", {})
                self.model._original_nc = meta.get("original_nc", head.nc)
                head.original_nc = self.model._original_nc

                # Restore anomaly_args (new format) or fall back to old per-key format.
                saved_args = meta.get("anomaly_args")
                if saved_args:
                    head.set_anomaly_args(**{k: v for k, v in saved_args.items()
                                            if k not in ("anomaly_mode",)})
                else:
                    # Legacy checkpoint: individual keys stored at the meta top level.
                    legacy = {k: meta[k] for k in head.anomaly_args if k in meta
                              and k != "anomaly_mode"}
                    if legacy:
                        head.set_anomaly_args(**legacy)

                anomaly_mode = (saved_args or meta).get("anomaly_mode", True)
                head.set_anomaly_mode(anomaly_mode)
                if anomaly_mode:
                    self.model.names = {0: list(self.model._anomaly_names.values())[0]}
                else:
                    self.model.names = self.model._original_names

            # Freeze memory bank (loaded model should not accumulate)
            self.model.freeze_memory_bank()

            has_mb = any(
                h.memory_bank.numel() > 0 and h.memory_bank.shape[0] > 10
                for h in head.iter_ad_heads(include_fused=True)
            )
            if has_mb:
                stats = self.model.get_memory_bank_stats()
                LOGGER.info("YOLOAnomaly: restored anomaly model from checkpoint")
                for i, s in enumerate(stats):
                    LOGGER.info(f"  Head[{i}]: {s['size']} features, dim={s['feature_dim']}")


class AnomalyDINO(YOLOAnomaly):
    """Training-free anomaly detector backed by a frozen DINOv2 ViT.

    Reuses ``YOLOAnomaly``'s ``predict()`` and ``val()`` unchanged.  The DINOv2
    backbone is wired in by replacing ``self.model.forward`` with a function
    that returns the same ``(preds, {"heatmap": hm})`` tuple ``AnomalyDetection``
    emits, so ``AnomalyPredictor`` / ``AnomalyValidator`` handle the rest
    (NMS, mAP, image/pixel AUROC).  Only the support-set pipeline stages 1–3
    (feature extraction, accumulation, coreset) are overridden.

    Memory bank: flat (M, D) tensor of L2-normalised DINOv2 patch tokens.
    Heatmap:     (H_in, W_in) cosine-distance map upsampled from the patch grid.
    Boxes:       thresholded heatmap → connected components → xyxy (cls=0).

    Args:
        model: Path to any YOLO/YOLOE checkpoint (used for YOLO predict/val plumbing).
        dino_model: torch.hub DINOv2 variant, e.g. ``"dinov2_vitb14"``.
        device: Device for DINOv2 (None = auto).
        verbose: Verbose logging.

    Examples:
        >>> d = AnomalyDINO("yolo26l.pt", dino_model="dinov2_vitb14")
        >>> d.load_support_set("mvtec/leather/train/good/")
        >>> results = d.predict("mvtec/leather/test/")
        >>> metrics = d.val(data="mvtec/leather.yaml")
    """

    # ImageNet normalisation used by all DINOv2 models.
    _DINO_MEAN = (0.485, 0.456, 0.406)
    _DINO_STD  = (0.229, 0.224, 0.225)
    # 518 = 37 × 14 (patch_size=14) — DINOv2 canonical resolution; overridable per-instance.
    _DINO_SIZE = 518

    def __init__(
        self,
        model: str | Path,
        dino_model: str = "dinov2_vitb14",
        imgsz: int | None = None,
        device: str | None = None,
        verbose: bool = False,
    ) -> None:
        """Initialize AnomalyDINO with a YOLO checkpoint and a DINOv2 backbone.

        Args:
            imgsz: DINOv2 input resolution.  Defaults to 518.  Must be a
                multiple of the patch size (14).  Smaller values (e.g. 448)
                speed up the forward by ~``(imgsz/518)**2``.
        """
        super().__init__(model, verbose=verbose)
        # Per-instance override of the class-level _DINO_SIZE so different
        # AnomalyDINO instances can use different resolutions.
        if imgsz is not None:
            if imgsz % 14 != 0:
                raise ValueError(f"imgsz must be a multiple of 14 (DINOv2 patch size); got {imgsz}")
            self._DINO_SIZE = int(imgsz)
        # Install AnomalyDetection head so predict/val route through
        # AnomalyPredictor/AnomalyValidator.  We deliberately leave end2end=False
        # so post-processing flows through standard non_max_suppression →
        # (Ni, 6) tensors, which DetectionValidator.update_metrics handles
        # correctly (the dict-output end2end branch is incompatible with the
        # default mAP accumulator).
        self.setup(["anomaly"])
        self.model.end2end = False

        # DINOv2 nn.Module stored via __dict__ to bypass nn.Module.__setattr__.
        self.__dict__["_dino"] = None
        self._dino_model_name = dino_model
        self._dino_device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._dino_bank: "torch.Tensor | None" = None
        self._dino_frozen = False
        self._anomaly_args: dict = {
            "K": 5,
            "score_aggregation": "max",
            "max_bank_size": 10000,
            "ad_conf": 0.3,         # raw cosine-distance threshold for boxes
            "ad_max_det": 9,
        }
        # Redirect the YOLO model's forward to our DINOv2-driven implementation.
        # nn.Module.__call__ → self.forward(x); instance attribute shadows the class method.
        self.model.forward = self._dino_forward

    # ── anomaly_args: own dict, not tied to YOLO head ──────────────────────

    @property
    def anomaly_args(self) -> dict:
        """Return AnomalyDINO's own parameter dict."""
        return self._anomaly_args

    def set_anomaly_args(self, **kwargs) -> None:
        """Update inference parameters (K, score_aggregation, max_bank_size, ad_conf, ad_max_det)."""
        for k in ("K", "score_aggregation", "max_bank_size", "ad_conf", "ad_max_det"):
            if k in kwargs and kwargs[k] is not None:
                self._anomaly_args[k] = kwargs[k]

    # ── is_configured ───────────────────────────────────────────────────────

    @property
    def is_configured(self) -> bool:
        """True when the DINOv2 memory bank is built and frozen."""
        return self._dino_bank is not None and self._dino_frozen

    # ── pipeline stage overrides ────────────────────────────────────────────

    def _prepare_support_session(self, **kwargs) -> None:
        """Stage 0: reset DINOv2 bank; skip YOLO memory-update machinery."""
        self._dino_bank   = None
        self._dino_frozen = False

    def _extract_support_features(self, items) -> "torch.Tensor":
        """Stage 1: extract L2-normalised DINOv2 patch tokens, batched.

        Args:
            items: Single image path/tensor or a list of them.  Lists are
                batched into one DINOv2 forward, which dominates wall-time.

        Returns:
            Tensor: (B * N_patches, D) float32 on CPU.
        """
        import torch.nn.functional as F

        sources = items if isinstance(items, (list, tuple)) else [items]
        self._load_dino()
        # Stack per-image preprocessed tensors into one batch.
        imgs = torch.cat([self._preprocess_for_dino(p) for p in sources], dim=0)
        with torch.no_grad():
            out = self._dino.forward_features(imgs)
        patches = out["x_norm_patchtokens"]            # (B, N_patches, D)
        D = patches.shape[-1]
        patches = F.normalize(patches.reshape(-1, D), dim=-1)   # (B*N_patches, D)
        return patches.detach().cpu()

    def _accumulate_features(self, feats: "torch.Tensor") -> None:
        """Stage 2: concatenate patch tokens into the flat bank."""
        self._dino_bank = feats if self._dino_bank is None \
                          else torch.cat([self._dino_bank, feats], dim=0)

    def _compact_memory_bank(self) -> None:
        """Stage 3a: optional k-center coreset on the flat DINOv2 bank."""
        cap = self._anomaly_args.get("max_bank_size")
        if cap and self._dino_bank is not None and self._dino_bank.shape[0] > cap:
            from ultralytics.nn.modules.head import _coreset_subsample
            LOGGER.info(
                "AnomalyDINO: coreset %d → %d", self._dino_bank.shape[0], cap
            )
            self._dino_bank = _coreset_subsample(self._dino_bank, cap)
        self._dino_frozen = True

    def _finalize_memory_bank(self, verbose: bool = True) -> list[dict]:
        """Stage 3b: compact + return stats (one entry; no per-YOLO-head split)."""
        self._compact_memory_bank()
        size = self._dino_bank.shape[0] if self._dino_bank is not None else 0
        dim  = self._dino_bank.shape[1] if self._dino_bank is not None else None
        stats = [{"size": size, "feature_dim": dim}]
        if verbose:
            LOGGER.info(f"  DINOv2 bank: {size} patch tokens, dim={dim}")
        return stats

    # ── DINOv2 backbone helpers ─────────────────────────────────────────────

    def _load_dino(self) -> None:
        """Lazy-load DINOv2 from torch.hub on first call.

        The loaded nn.Module is stored via ``self.__dict__`` directly so that
        ``nn.Module.__setattr__`` does not route it to ``self._modules`` (which
        would make it invisible to normal attribute lookup through
        ``Model.__getattr__``).
        """
        if self.__dict__.get("_dino") is not None:
            return
        LOGGER.info("AnomalyDINO: loading %s from torch.hub…", self._dino_model_name)
        dino = torch.hub.load("facebookresearch/dinov2", self._dino_model_name, verbose=False)
        dino.eval().to(self._dino_device)
        self.__dict__["_dino"] = dino

    def _preprocess_for_dino(self, img_path: "str | Path") -> "torch.Tensor":
        """Load and preprocess one image for DINOv2 input.

        Resizes to _DINO_SIZE × _DINO_SIZE, applies ImageNet normalisation.

        Args:
            img_path: Path to an RGB-compatible image.

        Returns:
            Tensor: (1, 3, H, W) float32 on self._dino_device.
        """
        import torchvision.transforms.functional as TF
        from PIL import Image

        img = Image.open(img_path).convert("RGB")
        img = TF.resize(img, [self._DINO_SIZE, self._DINO_SIZE],
                        interpolation=TF.InterpolationMode.BICUBIC)
        t = TF.to_tensor(img)
        t = TF.normalize(t, mean=list(self._DINO_MEAN), std=list(self._DINO_STD))
        return t.unsqueeze(0).to(self._dino_device)

    def _extract_dino_features(
        self, img_path: "str | Path"
    ) -> "tuple[torch.Tensor, int, int]":
        """Run DINOv2 on one image and return patch tokens + grid shape.

        Args:
            img_path: Path to image.

        Returns:
            tuple:
                - patches (Tensor): (N, D) L2-normalised patch tokens.
                - grid_h (int): Number of patch rows.
                - grid_w (int): Number of patch columns.
        """
        import torch.nn.functional as F

        self._load_dino()
        if isinstance(img_path, torch.Tensor):
            # Already a (1, 3, H, W) ImageNet-normalised tensor from _dino_forward.
            img_t = img_path.to(self._dino_device)
            if img_t.shape[-2:] != (self._DINO_SIZE, self._DINO_SIZE):
                # Bicubic to match the support-set preprocessing in _preprocess_for_dino.
                img_t = F.interpolate(img_t, size=(self._DINO_SIZE, self._DINO_SIZE),
                                      mode="bicubic", align_corners=False)
        else:
            img_t = self._preprocess_for_dino(img_path)   # (1, 3, H, W)
        with torch.no_grad():
            out = self._dino.forward_features(img_t)
        patches = out["x_norm_patchtokens"]            # (1, N, D)
        patches = F.normalize(patches.squeeze(0), dim=-1)  # (N, D)
        patch_size = getattr(self._dino, "patch_size", 14)
        grid_h = img_t.shape[2] // patch_size
        grid_w = img_t.shape[3] // patch_size
        return patches, grid_h, grid_w

    # ── Forward: bridges DINOv2 into the YOLO predict/val pipeline ──────────

    def _dino_forward(self, x, *args, **kwargs):
        """Drop-in replacement for ``YOLOAnomalyModel.forward``.

        Receives a YOLO-preprocessed batch ``(B, 3, H, W)`` of [0, 1] floats and
        emits the same tuple shape ``AnomalyDetection.forward`` produces:
        ``(preds_tensor, {"heatmap": hm})``.  ``preds_tensor`` is the standard
        pre-NMS YOLO format ``(B, 4 + nc, N) = (B, 5, N)`` (xywh + class score),
        so ``AnomalyPredictor`` / ``AnomalyValidator`` flow through their normal
        ``non_max_suppression`` → ``(Ni, 6)`` paths.

        Returns:
            tuple: ``(preds, {"heatmap": hm})``.
        """
        import torch.nn.functional as F

        assert self.is_configured, "Call load_support_set() before predict/val."
        B, _, H_in, W_in = x.shape
        K       = self._anomaly_args.get("K", 5)
        ad_conf = self._anomaly_args.get("ad_conf", 0.3)
        max_det = self._anomaly_args.get("ad_max_det", 9)
        bank    = self._dino_bank.to(self._dino_device)

        # YOLO preprocess emits [0, 1] floats; DINOv2 needs ImageNet normalisation.
        mean = torch.tensor(self._DINO_MEAN, device=x.device).view(1, 3, 1, 1)
        std  = torch.tensor(self._DINO_STD,  device=x.device).view(1, 3, 1, 1)
        x_dino = (x - mean) / std

        boxes_per_img: list[torch.Tensor] = []
        hm_full_list:  list[torch.Tensor] = []
        for i in range(B):
            patches, gh, gw = self._extract_dino_features(x_dino[i:i + 1])
            dists = self._compute_knn_distances(patches, bank, K).clamp(0.0, 2.0)
            hm = dists.reshape(gh, gw)
            hm_full = F.interpolate(
                hm[None, None].float(), size=(H_in, W_in),
                mode="bilinear", align_corners=False,
            )[0, 0]
            # Suppress border artifacts caused by letterbox padding / resizing.
            # DINOv2 patch size=14; on 518 px → 37 patches. Upsampled to ~640–672 px
            # each patch is ~18 px, so 2-patch artifacts need ~36 px margin.
            border = max(24, min(H_in, W_in) // 16)
            hm_full[:border, :] = 0
            hm_full[-border:, :] = 0
            hm_full[:, :border] = 0
            hm_full[:, -border:] = 0
            hm_full_list.append(hm_full.detach().cpu())
            boxes_per_img.append(self._heatmap_to_boxes(hm_full, thresh=ad_conf, max_det=max_det))

        # Pack per-image (M, 6) xyxy boxes into a (B, max_N, 6) end2end-format tensor:
        # rows are [x1, y1, x2, y2, conf, cls].  This is the unambiguous shape that
        # `non_max_suppression` recognises via its `prediction.shape[-1] == 6` shortcut,
        # so it just filters by conf and skips the pre-NMS pipeline (which would otherwise
        # misinterpret the (B, 5, max_N) layout when max_N happens to equal 6).  Padded
        # rows have conf=0 and are dropped by that filter.
        max_n = max((b.shape[0] for b in boxes_per_img), default=0) or 1
        preds = torch.zeros((B, max_n, 6), dtype=torch.float32, device=x.device)
        for i, b in enumerate(boxes_per_img):
            n = b.shape[0]
            if n == 0:
                continue
            preds[i, :n] = b.to(x.device)
        return preds, {"heatmap": torch.stack(hm_full_list, dim=0)}


