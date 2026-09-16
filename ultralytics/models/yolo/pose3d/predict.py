# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from ultralytics.models.yolo.pose import PosePredictor
from ultralytics.utils import DEFAULT_CFG


class Pose3DPredictor(PosePredictor):
    """Predictor for the 3D pose task.

    Inference is identical to `pose` — the extra depth channel rides inside the keypoint tensor — so the only
    change is the task name. Results carry keypoints of shape (N, nkpt, 4) as (x_px, y_px, visible, z_encoded);
    call `ultralytics.utils.pose3d.decode_z` or `keypoints_to_camera` to get metres.

    Examples:
        >>> from ultralytics.models.yolo.pose3d import Pose3DPredictor
        >>> predictor = Pose3DPredictor(overrides=dict(model="yolo26n-pose3d.pt", source="bus.jpg"))
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the 3D pose predictor, forcing the task to 'pose3d'."""
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "pose3d"
