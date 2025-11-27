# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionPredictor


class Stereo3DDetPredictor(DetectionPredictor):
    """Stereo 3D Detection predictor.

    Reuses the detection predictor for now. Custom stereo visualization can be layered on top.
    """

    pass
