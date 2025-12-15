# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.nn.tasks import DetectionModel


class Stereo3DDetModel(DetectionModel):
    """Placeholder Stereo 3D Detection model.

    For now this extends the standard DetectionModel so we can train/eval the
    stereo3ddet task end-to-end while we iterate on a dedicated stereo head/loss.
    """

    def __init__(self, cfg="yolo11-stereo3ddet.yaml", ch=3, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # Mark task for downstream components
        self.task = "stereo3ddet"
