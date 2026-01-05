# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.nn.tasks import DetectionModel


class Stereo3DDetModel(DetectionModel):
    """Placeholder Stereo 3D Detection model.

    For now this extends the standard DetectionModel so we can train/eval the
    stereo3ddet task end-to-end while we iterate on a dedicated stereo head/loss.
    """

    def __init__(self, cfg="yolo11-stereo3ddet.yaml", ch=6, nc=None, verbose=True):
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
        
        Overrides DetectionModel.loss() to handle models that return dict predictions.
        If preds is a dict (stereo3ddet format), we need to use StereoCenterNetLoss instead of v8DetectionLoss.
        
        Args:
            batch (dict): Batch to compute loss on.
            preds (dict | torch.Tensor | list[torch.Tensor], optional): Predictions.
            
        Returns:
            Tuple of (total_loss, loss_items) where loss_items is a tensor.
        """
        # Get preds if not provided
        if preds is None:
            preds = self.forward(batch["img"])
        
        # New path: YOLO11 bbox mapping head (P3-only) returns dict with "det" + aux branches.
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

        # Legacy path: CenterNet-style stereo3ddet dict prediction - use StereoCenterNetLoss
        stereo_keys = {"heatmap", "offset", "bbox_size", "lr_distance", "right_width",
                       "dimensions", "orientation", "vertices", "vertex_offset", "vertex_dist"}
        if isinstance(preds, dict) and stereo_keys.issubset(set(preds.keys())):
            # This is a stereo3ddet dict prediction - we need to use StereoCenterNetLoss
            # Check if we have access to the criterion through the model structure
            from ultralytics.models.yolo.stereo3ddet.stereo_yolo_v11 import StereoCenterNetLoss
            
            # Initialize criterion if needed
            if not hasattr(self, "_stereo_criterion"):
                # Try to get num_classes from model or batch
                num_classes = getattr(self, "nc", 3)
                if "targets" in batch and isinstance(batch["targets"], dict):
                    # Infer from targets if available
                    if "heatmap" in batch["targets"]:
                        num_classes = batch["targets"]["heatmap"].shape[1] if len(batch["targets"]["heatmap"].shape) > 1 else 3
                
                # Read loss weights from config if available
                loss_weights = None
                if hasattr(self, "yaml") and self.yaml is not None:
                    training_config = self.yaml.get("training", {})
                    if training_config and "loss_weights" in training_config:
                        loss_weights = training_config["loss_weights"]
                
                self._stereo_criterion = StereoCenterNetLoss(num_classes=num_classes, loss_weights=loss_weights)
            
            # Compute loss using StereoCenterNetLoss
            targets = batch.get("targets", {})
            if not targets:
                # If targets not in batch, we can't compute loss
                raise ValueError("Stereo3ddet requires 'targets' dict in batch for loss computation")
            
            total_loss, loss_dict = self._stereo_criterion(preds, targets)
            
            # Convert loss_dict to loss_items tensor in fixed order
            loss_items_list = [
                loss_dict.get("heatmap", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("offset", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("bbox_size", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("lr_distance", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("right_width", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("dimensions", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("orientation", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("vertices", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("vertex_offset", torch.tensor(0.0, device=total_loss.device)),
                loss_dict.get("vertex_dist", torch.tensor(0.0, device=total_loss.device)),
            ]
            loss_items = torch.stack(loss_items_list)  # [10]
            
            return total_loss, loss_items
        
        # Fallback to parent implementation for standard detection models
        return super().loss(batch, preds=preds)
