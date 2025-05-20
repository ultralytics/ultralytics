# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path
from typing import Union, List, Any

import torch
import numpy as np

from ultralytics.engine.model import Model
from ultralytics.engine.results import Results
from ultralytics.models import yolo_manitou
from ultralytics.nn.tasks import (
    DetectionModel,
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
            "detect_multiCam": {
                "model": DetectionModel,
                "trainer": yolo_manitou.detect_multiCam.ManitouTrainer_MultiCam,
                "validator": yolo_manitou.detect_multiCam.ManitouValidator_MultiCam,
                "predictor": yolo_manitou.detect_multiCam.ManitouPredictor_MultiCam,
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