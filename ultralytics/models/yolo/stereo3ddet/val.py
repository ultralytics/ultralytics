# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionValidator


class Stereo3DDetValidator(DetectionValidator):
    """Stereo 3D Detection validator.

    Currently inherits detection validator for basic plumbing. Stereo metrics can be added later.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None) -> None:
        super().__init__(dataloader, save_dir, args, _callbacks)
        self.args.task = "stereo3ddet"
