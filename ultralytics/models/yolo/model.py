# Ultralytics YOLO 🚀, AGPL-3.0 license

# import inspect
from pathlib import Path
# from typing import Union
# from abc import abstractmethod, ABC

# import numpy as np
# from PIL import Image
# import torch

from ultralytics.engine.model import Model
from ultralytics.models import yolo
from ultralytics.nn.tasks import ClassificationModel, DetectionModel, OBBModel, PoseModel, SegmentationModel, WorldModel
# from ultralytics.engine.results import Results
from ultralytics.utils import ROOT, yaml_load
# from ultralytics.utils.metrics import DetMetrics  #, SegmentMetrics, PoseMetrics, OBBMetrics, ClassifyMetrics

# metrics = {
#     "detect": DetMetrics,
#     "segment": SegmentMetrics,
#     "pose": PoseMetrics,
#     "obb": OBBMetrics,
#     "classify": ClassifyMetrics,
# }

# class ModelMeta(ABC):

#     # @abstractmethod
#     def predict(
#             self,
#             source: Union[str, Path, np.ndarray, Image.Image, torch.Tensor, list[Union[str, Path, np.ndarray, Image.Image, torch.Tensor]]],
#             conf: float = 0.25,
#             iou: float = 0.45,
#             imgsz: Union[int, tuple[int, int]] = 640,
#             half: bool = False,
#             device: str = None,
#             max_det: int = 100,
#             vid_stride: int = 1,
#             stream_buffer: bool = False,
#             visualize: bool = False,
#             augment: bool = False,
#             agnostic_nms: bool = False,
#             classes: list[int] = None,
#             retina_masks: bool = False,
#             embed: list[int] = None,
#         ) -> list[Results]:
#         """
#         Run inference on a variety of sources.

#         Args:
#             source: Input source.
#             conf: Confidence threshold.
#             iou: IoU threshold.
#             imgsz: Inference size (pixels).
#             half: Use half precision.
#             device: Device to use for inference.
#             max_det: Maximum number of detections per image.
#             vid_stride: Frame stride for video inference.
#             stream_buffer: Stream video frames from a buffer.
#             visualize: Visualize the results.
#             augment: Use augmented inference.
#             agnostic_nms: Use agnostic NMS.
#             classes: List of classes to filter the results.
#             retina_masks: Use retina masks.
#             embed: List of indices to embed in the results.

#         Returns:
#             List[Results]: Inference results.
#         """
#         ...
    
#     # @abstractmethod
#     def train(
#             self,
#             model: str = None,
#             data: str = None,
#             epochs: int = 300,
#             time: int = None,
#             patience: int = 100,
#             batch: int = 16,
#             imgsz: int = 640,
#             save: bool = True,
#             save_period: int = -1,
#             cache: Union[str, bool] = False,
#             device: str = None,
#             workers: int = 8,
#             project: str = None,
#             name: str = None,
#             exist_ok: bool = False,
#             pretrained: bool = True,
#             optimizer: str = "auto",
#             verbose: bool = False,
#             seed: int = 0,
#             deterministic: bool = True,
#             single_cls: bool = False,
#             rect: bool = False,
#             cos_lr: bool = False,
#             close_mosaic: int = 10,
#             resume: bool = False,
#             amp: bool = True,
#             fraction: float = 1.0,
#             profile: bool = False,
#             freeze: Union[int, list[int]] = None,
#             lr0: float = 0.01,
#             lrf: float = 0.01,
#             momentum: float = 0.937,
#             weight_decay: float = 0.0005,
#             warmup_epochs: Union[int, float] = 3.0,
#             warmup_bias_lr: float = 0.1,
#             box: float = 7.5,
#             cls: float = 0.5,
#             dfl: float = 1.5,
#             pose: float = 12.0,
#             kobj: float = 2.0,
#             label_smoothing: float = 0.0,
#             nbs: int = 64,
#             overlap_mask: bool = True,
#             mask_ratio: int = 4,
#             dropout: float = 0.0,
#             val: bool = True,
#             plots: bool = True,
#     ) -> DetMetrics:
#         ...
    
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         for method in ['train', 'predict']:
#             if not hasattr(cls, method):
#                 raise NotImplementedError(f"{cls.__name__} must implement the '{method}' method")

#         # Dynamically add the method signature to the subclass
#         cls.predict.__signature__ = inspect.signature(ModelMeta.predict)
#         cls.train.__signature__ = inspect.signature(ModelMeta.train)

class YOLO(Model):
    """YOLO (You Only Look Once) object detection model."""

    def __init__(self, model="yolov8n.pt", task=None, verbose=False):
        """Initialize YOLO model, switching to YOLOWorld if model filename contains '-world'."""
        path = Path(model)
        if "-world" in path.stem and path.suffix in {".pt", ".yaml", ".yml"}:  # if YOLOWorld PyTorch model
            new_instance = YOLOWorld(path, verbose=verbose)
            self.__class__ = type(new_instance)
            self.__dict__ = new_instance.__dict__
        else:
            # Continue with default YOLO initialization
            super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
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
    """YOLO-World object detection model."""

    def __init__(self, model="yolov8s-world.pt", verbose=False) -> None:
        """
        Initializes the YOLOv8-World model with the given pre-trained model file. Supports *.pt and *.yaml formats.

        Args:
            model (str | Path): Path to the pre-trained model. Defaults to 'yolov8s-world.pt'.
        """
        super().__init__(model=model, task="detect", verbose=verbose)

        # Assign default COCO class names when there are no custom names
        if not hasattr(self.model, "names"):
            self.model.names = yaml_load(ROOT / "cfg/datasets/coco8.yaml").get("names")

    @property
    def task_map(self):
        """Map head to model, validator, and predictor classes."""
        return {
            "detect": {
                "model": WorldModel,
                "validator": yolo.detect.DetectionValidator,
                "predictor": yolo.detect.DetectionPredictor,
                "trainer": yolo.world.WorldTrainer,
            }
        }

    def set_classes(self, classes):
        """
        Set classes.

        Args:
            classes (List(str)): A list of categories i.e. ["person"].
        """
        self.model.set_classes(classes)
        # Remove background if it's given
        background = " "
        if background in classes:
            classes.remove(background)
        self.model.names = classes

        # Reset method class names
        # self.predictor = None  # reset predictor otherwise old names remain
        if self.predictor:
            self.predictor.model.names = classes
