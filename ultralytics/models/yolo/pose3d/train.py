# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

from ultralytics.models import yolo
from ultralytics.nn.tasks import Pose3DModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK


class Pose3DTrainer(yolo.pose.PoseTrainer):
    """Trainer for the 3D pose task.

    Differs from `PoseTrainer` only in the model and validator it builds, plus one warning: the depth targets are
    metric, and any augmentation that changes apparent object scale changes the depth cue the network reads without
    changing the target, so `scale` and `mosaic` are silently teaching the wrong thing. That interaction is the
    subject of hypothesis H2 and is left on by default so the baseline is comparable to `pose`.

    Examples:
        >>> from ultralytics.models.yolo.pose3d import Pose3DTrainer
        >>> trainer = Pose3DTrainer(overrides=dict(model="yolo26n-pose3d.yaml", data="coco8-pose3d.yaml", epochs=3))
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a Pose3DTrainer, forcing the task to 'pose3d'."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "pose3d"
        super().__init__(cfg, overrides, _callbacks)
        if self.args.scale or self.args.mosaic:
            LOGGER.warning(
                "pose3d: scale/mosaic augmentation changes apparent object size without changing the metric depth "
                "target. Root-depth accuracy may suffer; see the pose3d-research branch, research/design/pose3d-task-design.md (H2)."
            )

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> Pose3DModel:
        """Build and return a `Pose3DModel`."""
        model = Pose3DModel(
            cfg,
            nc=self.data["nc"],
            ch=self.data["channels"],
            data_kpt_shape=self.data["kpt_shape"],
            verbose=verbose and RANK == -1,
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self) -> yolo.pose3d.Pose3DValidator:
        """Return a `Pose3DValidator` for this trainer's save directory and args."""
        return yolo.pose3d.Pose3DValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))
