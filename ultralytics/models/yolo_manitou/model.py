# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Any, List, Union

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.models import yolo_manitou
from ultralytics.nn.tasks import (
    DetectionModel,
    DetectionModel_MultiView,
    SegmentationModel,
)


class YOLOManitou(Model):
    """YOLO (You Only Look Once) object detection model."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel,
                "trainer": yolo_manitou.detect.ManitouTrainer,
                "validator": yolo_manitou.detect.ManitouValidator,
                "predictor": yolo_manitou.detect.ManitouPredictor,
            },
            "segment": {
                "model": SegmentationModel,
                "trainer": yolo_manitou.segment.ManitouSegmentationTrainer,
                "validator": yolo_manitou.segment.ManitouSegmentationValidator,
                "predictor": yolo_manitou.segment.ManitouSegmentationPredictor,
            },
        }

    def track(
        self,
        source: Union[str, Path, int, list, tuple, np.ndarray, torch.Tensor] = None,
        stream: bool = False,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        """
        Conducts object tracking on the specified input source using the registered trackers.

        This method performs object tracking using the model's predictors and optionally registered trackers. It handles
        various input sources such as file paths or video streams, and supports customization through keyword arguments.
        The method registers trackers if not already present and can persist them between calls.

        Args:
            source (Union[str, Path, int, List, Tuple, np.ndarray, torch.Tensor], optional): Input source for object
                tracking. Can be a file path, URL, or video stream.
            stream (bool): If True, treats the input source as a continuous video stream.
            persist (bool): If True, persists trackers between different calls to this method.
            **kwargs (Any): Additional keyword arguments for configuring the tracking process.

        Returns:
            (List[ultralytics.engine.results.Results]): A list of tracking results, each a Results object.

        Examples:
            >>> model = YOLO("yolo11n.pt")
            >>> results = model.track(source="path/to/video.mp4", show=True)
            >>> for r in results:
            ...     print(r.boxes.id)  # print tracking IDs

        Notes:
            - This method sets a default confidence threshold of 0.1 for ByteTrack-based tracking.
            - The tracking mode is explicitly set in the keyword arguments.
            - Batch size is set to 1 for tracking in videos.
        """
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker_manitou

            register_tracker_manitou(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = kwargs.get("batch") or 1  # batch-size 1 for tracking in videos
        kwargs["mode"] = "track"
        return self.predict(source=source, stream=stream, **kwargs)


class YOLOManitou_MultiCam(Model):
    """YOLO (You Only Look Once) object detection model for multi-camera scenarios."""

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": DetectionModel_MultiView,
                "trainer": yolo_manitou.detect_multiCam.ManitouTrainer_MultiCam,
                "validator": yolo_manitou.detect_multiCam.ManitouValidator_MultiCam,
                "predictor": yolo_manitou.detect_multiCam.ManitouPredictor_MultiCam,
            },
        }

    def predict(
        self,
        data_cfg,
        predictor=None,
        **kwargs: Any,
    ) -> List[Results]:
        if data_cfg is None:
            (f"'data_cfg' is missing. Using 'data_cfg={data_cfg}'.")

        custom = {"conf": 0.25, "batch": 1, "mode": "predict", "rect": True}  # method defaults
        args = {**self.overrides, **custom, **kwargs}  # highest priority args on the right

        if not self.predictor:
            self.predictor = (predictor or self._smart_load("predictor"))(overrides=args, _callbacks=self.callbacks)
            self.predictor.setup_model(model=self.model)
        else:  # only update args if predictor is already setup
            self.predictor.args = get_cfg(self.predictor.args, args)
            if "project" in args or "name" in args:
                self.predictor.save_dir = get_save_dir(self.predictor.args)
        return self.predictor(data_cfg=data_cfg)

    def track(
        self,
        data_cfg,
        persist: bool = False,
        **kwargs: Any,
    ) -> List[Results]:
        if not hasattr(self.predictor, "trackers"):
            from ultralytics.trackers import register_tracker_manitou_multiview

            register_tracker_manitou_multiview(self, persist)
        kwargs["conf"] = kwargs.get("conf") or 0.1  # ByteTrack-based method needs low confidence predictions as input
        kwargs["batch"] = 1  # batch-size 1 for tracking
        kwargs["mode"] = "track"
        return self.predict(data_cfg=data_cfg, **kwargs)

    def __call__(
        self,
        data_cfg,
        stream: bool = False,
        **kwargs: Any,
    ) -> list:
        return self.predict(data_cfg, **kwargs)
