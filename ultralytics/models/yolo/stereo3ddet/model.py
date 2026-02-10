# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from ultralytics.nn.tasks import DetectionModel


class Stereo3DDetModel(DetectionModel):
    """Stereo 3D Detection model — standard YOLO with 6-channel input."""

    def __init__(self, cfg, ch=6, nc=None, verbose=True):
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.task = "stereo3ddet"

    def init_criterion(self):
        """Initialize the loss criterion."""
        from ultralytics.models.yolo.stereo3ddet.loss_yolo11 import Stereo3DDetLossYOLO11

        aux_w = None
        use_bbox_loss = True
        if hasattr(self, "yaml") and self.yaml is not None:
            training_config = self.yaml.get("training", {})
            if training_config:
                if "loss_weights" in training_config:
                    aux_w = training_config["loss_weights"]
                if "use_bbox_loss" in training_config:
                    use_bbox_loss = bool(training_config["use_bbox_loss"])

        return Stereo3DDetLossYOLO11(self, loss_weights=aux_w, use_bbox_loss=use_bbox_loss)
