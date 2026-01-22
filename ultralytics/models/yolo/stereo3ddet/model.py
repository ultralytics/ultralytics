# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.nn.tasks import DetectionModel

class Stereo3DDetModel(DetectionModel):
    """Placeholder Stereo 3D Detection model.

    For now this extends the standard DetectionModel so we can train/eval the
    stereo3ddet task end-to-end while we iterate on a dedicated stereo head/loss.
    """

    def __init__(self, cfg, ch=6, nc=None, verbose=True):
        # Load config to get input_channels if ch not explicitly provided
        # When called from Model._new(), cfg is already a dict
        if isinstance(cfg, dict):
            cfg_dict = cfg
        else:
            from ultralytics.nn.tasks import yaml_model_load
            cfg_dict = yaml_model_load(cfg)
        
        # Read input_channels from config if available (for stereo: 6 channels)
        input_channels = cfg_dict.get("input_channels")
        if input_channels is not None:
            ch = input_channels
        
        # Pass cfg (dict or string) to parent - DetectionModel handles both
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        # Mark task for downstream components
        self.task = "stereo3ddet"

    def loss(self, batch, preds=None):
        """Compute loss for stereo3ddet model.
        
        Overrides DetectionModel.loss() to handle YOLO11-style dict predictions with "det" + aux branches.
        
        Args:
            batch (dict): Batch to compute loss on.
            preds (dict | torch.Tensor | list[torch.Tensor], optional): Predictions.
            
        Returns:
            Tuple of (total_loss, loss_items) where loss_items is a tensor.
        """
        # Get preds if not provided
        if preds is None:
            preds = self.forward(batch["img"])
        
        # YOLO11 bbox mapping head (P3-only) returns dict with "det" + aux branches.
        if isinstance(preds, dict) and "det" in preds:
            from ultralytics.models.yolo.stereo3ddet.loss_yolo11 import Stereo3DDetLossYOLO11P3

            if not hasattr(self, "_stereo_yolo11_criterion"):
                # Optional aux weights from YAML training section
                aux_w = None
                if hasattr(self, "yaml") and self.yaml is not None:
                    training_config = self.yaml.get("training", {})
                    if training_config and "loss_weights" in training_config:
                        aux_w = training_config["loss_weights"]
                self._stereo_yolo11_criterion = Stereo3DDetLossYOLO11P3(self, loss_weights=aux_w)

            out = self._stereo_yolo11_criterion(preds, batch)
            return out.total, out.loss_items
        
        # Fallback to parent implementation for standard detection models
        return super().loss(batch, preds=preds)
