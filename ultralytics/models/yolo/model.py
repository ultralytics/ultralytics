# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.cfg import get_cfg
from ultralytics.data.build import load_inference_source
from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.autobackend import check_class_names
from ultralytics.nn.tasks import (
    ClassificationModel,
    DepthModel,
    DetectionModel,
    OBBModel,
    PoseModel,
    SegmentationModel,
    SemanticSegmentationModel,
    WorldModel,
    YOLOEModel,
    YOLOESegModel,
)
from ultralytics.utils import ROOT, YAML


class YOLO(Model):
    """YOLO (You Only Look Once) object detection model.

    This class provides a unified interface for YOLO models, automatically switching to specialized model types
    (YOLOWorld or YOLOE) based on the model filename. It supports various computer vision tasks including object
    detection, instance segmentation, semantic segmentation, classification, pose estimation, and oriented bounding box
    detection.

    Attributes:
        model: The loaded YOLO model instance.
        task: The task type (detect, segment, semantic, classify, pose, obb).
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
            "depth": {
                "model": DepthModel,
                "trainer": yolo.depth.DepthTrainer,
                "validator": yolo.depth.DepthValidator,
                "predictor": yolo.depth.DepthPredictor,
            },
            "semantic": {
                "model": SemanticSegmentationModel,
                "trainer": yolo.semantic.SemanticSegmentationTrainer,
                "validator": yolo.semantic.SemanticSegmentationValidator,
                "predictor": yolo.semantic.SemanticSegmentationPredictor,
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
                "validator": yolo.world.WorldValidator,
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
        get_vocab: Get the vocabulary for the given class names, which become the model's classes as the head is fused.
        set_classes: Set the model's class names and embeddings for detection.
        save_prompt_embeddings: Save the current prompt embeddings and class names to an NPZ file.
        load_prompt_embeddings: Load prompt embeddings and class names from an NPZ file.
        val: Validate the model using text or visual prompts.
        predict: Run prediction on images, videos, directories, streams, etc.

    Examples:
        Load a YOLOE segmentation model
        >>> model = YOLOE("yoloe-11s-seg.pt")

        Predict with visual prompts, whose 'cls' holds one class index per box
        >>> from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
        >>> prompts = {"bboxes": np.array([[10, 20, 100, 200]]), "cls": np.array([0])}
        >>> results = model.predict("image.jpg", visual_prompts=prompts, predictor=YOLOEVPSegPredictor)

        Re-parameterize into a prompt-free model, which no longer accepts prompts
        >>> names = ["person", "car", "dog"]
        >>> model.set_vocab(model.get_vocab(names), names)
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

    def set_vocab(self, vocab: torch.nn.ModuleList, names: list[str]) -> None:
        """Re-parameterize the model into a prompt-free one over the given class names.

        The vocabulary is the fused classification layer `get_vocab` returns for the same names, not the names
        themselves. The model must be an instance of YOLOEModel.

        Args:
            vocab (torch.nn.ModuleList): Fused classification layers returned by `get_vocab` for `names`.
            names (list[str]): List of class names that the model can detect or classify.

        Raises:
            AssertionError: If the model is not an instance of YOLOEModel.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> names = ["person", "car", "dog"]
            >>> model.set_vocab(model.get_vocab(names), names)
        """
        assert isinstance(self.model, YOLOEModel)
        names = check_class_names(names)
        self.predictor = None  # the delegate destructively re-parameterizes the head
        self.model.set_vocab(vocab, names=names)

    def get_vocab(self, names):
        """Get the vocabulary for the given class names, which become the model's classes as the head is fused."""
        assert isinstance(self.model, YOLOEModel)
        self.predictor = None  # the delegate destructively fuses the promptable head
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
        names = self.model.names.values() if isinstance(self.model.names, dict) else self.model.names
        if embeddings is not None or sorted(names) != sorted(classes):
            if embeddings is None:
                embeddings = self.get_text_pe(classes)  # generate text embeddings if not provided
            self.model.set_classes(classes, embeddings)

        # Reset method class names
        if self.predictor:
            self.predictor.model.names = self.model.names

    def _prompt_embedding_model(self) -> str:
        """Return the checkpoint identifier used to bind prompt embeddings to this model."""
        source = self.overrides.get("pretrained") or getattr(self.model, "pt_path", None) or self.ckpt_path
        source = source if isinstance(source, (str, Path)) else self.model.yaml["yaml_file"]
        model = Path(source).stem
        return model[:-4] if model.endswith("-seg") else model

    def save_prompt_embeddings(self, file: str | Path) -> Path:
        """Save the current prompt embeddings and class names to an NPZ file.

        Args:
            file (str | Path): Destination NPZ file path.

        Returns:
            (Path): Path to the saved NPZ file.

        Raises:
            ValueError: If prompt embeddings have not been set or are invalid.
        """
        assert isinstance(self.model, YOLOEModel)
        embeddings = getattr(self.model, "pe", None)
        if not isinstance(embeddings, torch.Tensor) or embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Prompt embeddings must be set before they can be saved.")
        names = list(self.model.names.values()) if isinstance(self.model.names, dict) else list(self.model.names)
        if embeddings.shape[1] != len(names) or not torch.isfinite(embeddings).all():
            raise ValueError("Prompt embeddings must be finite and match the number of class names.")

        file = Path(file)
        if file.suffix.lower() != ".npz":
            raise ValueError(f"Prompt embedding file must have an '.npz' suffix, not '{file.suffix}'.")
        np.savez_compressed(
            file,
            embeddings=embeddings.detach().cpu().float().numpy(),
            names=np.asarray(names, dtype=np.str_),
            model=np.asarray(self._prompt_embedding_model(), dtype=np.str_),
        )
        return file

    def load_prompt_embeddings(self, file: str | Path) -> None:
        """Load prompt embeddings and class names from a model-bound NPZ file.

        Args:
            file (str | Path): Source NPZ file path created by :meth:`save_prompt_embeddings`.

        Raises:
            ValueError: If the file is invalid or belongs to a different YOLOE architecture.
        """
        assert isinstance(self.model, YOLOEModel)
        with np.load(file, allow_pickle=False) as data:
            if set(data.files) != {"embeddings", "names", "model"}:
                raise ValueError("Prompt embedding file must contain 'embeddings', 'names', and 'model'.")
            embeddings, names, model = data["embeddings"], data["names"], data["model"]

        if model.ndim != 0 or model.dtype.kind != "U":
            raise ValueError("Prompt embedding model identifier must be a scalar string.")
        model_name = str(model.item())
        if model_name != self._prompt_embedding_model():
            raise ValueError(
                f"Prompt embeddings for model '{model_name}' cannot be loaded into '{self._prompt_embedding_model()}'."
            )
        if names.ndim != 1 or names.dtype.kind != "U":
            raise ValueError("Prompt embedding class names must be a one-dimensional string array.")
        if embeddings.dtype != np.float32 or embeddings.ndim != 3 or embeddings.shape[0] != 1:
            raise ValueError("Prompt embeddings must be a float32 array with shape (1, classes, dimensions).")
        if embeddings.shape[1] != len(names) or embeddings.shape[2] != self.model.model[-1].embed:
            raise ValueError("Prompt embedding shape does not match the class names or model embedding dimension.")
        if not np.isfinite(embeddings).all():
            raise ValueError("Prompt embeddings must contain only finite values.")
        self.set_classes(names.tolist(), torch.from_numpy(embeddings.copy()).to(next(self.model.parameters()).device))

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
        visual_prompts: dict[str, np.ndarray | list[np.ndarray]] | None = None,
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
            visual_prompts (dict[str, np.ndarray | list[np.ndarray]]): Dictionary containing visual prompts for the
                model. Must include 'bboxes' and 'cls' keys when non-empty, holding either flat arrays or one array per
                image for an explicit list, tuple, or 4-D tensor source with no refer_image.
            refer_image (str | PIL.Image | np.ndarray, optional): Reference image for visual prompts.
            predictor (callable): Custom predictor class for visual prompt predictions. Defaults to
                YOLOEVPDetectPredictor.
            **kwargs (Any): Additional keyword arguments passed to the predictor.

        Returns:
            (list | generator): List of Results objects or generator of Results objects if stream=True.

        Examples:
            >>> model = YOLOE("yoloe-11s-seg.pt")
            >>> results = model.predict("path/to/image.jpg")
            >>> # With visual prompts, whose 'cls' holds one class index per box
            >>> from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
            >>> prompts = {"bboxes": np.array([[10, 20, 100, 200]]), "cls": np.array([0])}
            >>> results = model.predict("path/to/image.jpg", visual_prompts=prompts, predictor=YOLOEVPSegPredictor)
        """
        visual_prompts = visual_prompts if visual_prompts is not None else {}
        if len(visual_prompts):
            assert "bboxes" in visual_prompts and "cls" in visual_prompts, (
                f"Expected 'bboxes' and 'cls' in visual prompts, but got {visual_prompts.keys()}"
            )
            bboxes, classes = visual_prompts["bboxes"], visual_prompts["cls"]
            assert all(hasattr(x, "__len__") and getattr(x, "ndim", 1) > 0 for x in (bboxes, classes)), (
                "Expected non-scalar 'bboxes' and 'cls' visual prompts"
            )
            assert len(bboxes) == len(classes) > 0, "Expected an equal, non-zero number of boxes and classes"
            nested = yolo.yoloe.YOLOEVPDetectPredictor.is_per_image(visual_prompts)  # one prompt array per image
            assert not isinstance(source, np.ndarray) or source.ndim != 4, "4-D NumPy sources are not supported"
            per_image_source = isinstance(source, (list, tuple)) or (
                isinstance(source, torch.Tensor) and source.ndim == 4
            )
            assert not nested or (refer_image is None and per_image_source), (
                "Expected flat 'bboxes' and 'cls' arrays for a non-sequence source or when refer_image is set"
            )
            multi = nested
            pairs = list(zip(bboxes, classes)) if multi else [(bboxes, classes)]
            assert not multi or len(pairs) == len(source), (
                f"Expected one prompt per source image, but got {len(pairs)} prompts for {len(source)} images"
            )
            assert all(
                getattr(b, "ndim", 2) == 2
                and (not multi or b.shape[1:] == (4,))
                and getattr(c, "ndim", 1) == 1
                and len(b) == len(c)
                and all(np.isscalar(x) and not isinstance(x, (str, bytes)) for x in c)
                for b, c in pairs
            ), "Expected non-string scalar class indices for each bounding box"
            per_image = [len(set(c.tolist() if isinstance(c, np.ndarray) else c)) for _, c in pairs]
            assert all(per_image), "Expected at least one class per image"
            num_cls = max(per_image)
            if type(self.predictor) is not predictor:
                args = get_cfg(overrides={**self.overrides, **kwargs})
                self.predictor = predictor(
                    overrides={
                        "task": self.model.task,
                        "mode": "predict",
                        "save": False,
                        "verbose": kwargs.get("verbose", self.overrides.get("verbose", refer_image is None)),
                        "batch": 1,
                        "device": args.device,
                        "quantize": args.quantize,
                        "imgsz": args.imgsz,
                    },
                    _callbacks=self.callbacks,
                )

            self.model.model[-1].nc = num_cls
            self.model.names = [f"object{i}" for i in range(num_cls)]
            self.predictor.set_prompts(visual_prompts.copy())
            self.predictor.setup_model(model=self.model, verbose=self.predictor.args.verbose)

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
