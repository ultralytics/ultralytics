# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv

class Stereo3DDetModel(DetectionModel):
    """Stereo 3D Detection model with weight-shared backbone.

    Implements the paper's weight-sharing backbone approach where left and right
    images are processed separately through the same backbone, preserving epipolar
    geometry for accurate depth estimation.
    """

    def __init__(self, cfg, ch=6, nc=None, verbose=True):
        # Load config to get input_channels if ch not explicitly provided
        # When called from Model._new(), cfg is already a dict
        if isinstance(cfg, dict):
            cfg_dict = cfg.copy()
        else:
            from ultralytics.nn.tasks import yaml_model_load
            cfg_dict = yaml_model_load(cfg)
        
        # Store original input_channels for reference
        original_input_channels = cfg_dict.get("input_channels", 6)
        
        # Override input_channels to 3 for backbone (weight-sharing processes left/right separately)
        cfg_dict["input_channels"] = 3
        
        # Find backbone end (where head starts) BEFORE calling super().__init__()
        # This is needed because parent __init__ calls forward() to determine stride
        num_backbone_layers = len(cfg_dict["backbone"])
        self.backbone_end_idx = num_backbone_layers - 1  # Last backbone layer index
        
        # Build model with ch=3 for weight-shared backbone
        # Temporarily disable verbose to avoid confusing output during construction
        super().__init__(cfg=cfg_dict, ch=3, nc=nc, verbose=False)
        
        # Get backbone output channels from the last backbone layer
        # The last layer's output channels are stored in ch list after parsing
        # We need to find the channel count after backbone processing
        # Since we built with ch=3, we can trace through to get final backbone channels
        # For now, we'll determine it from the model structure
        # The backbone output is at layer self.backbone_end_idx
        # We'll get the actual channel count during forward pass or from model structure
        
        # Create 1×1 conv modules for channel reduction after feature fusion
        # This will reduce 2C channels (concatenated left+right) to C channels
        # We need separate convs for each saved backbone layer that's referenced by head
        # We'll determine channel counts during first forward pass
        self.fusion_convs = nn.ModuleDict()  # ModuleDict for automatic dtype handling
        self.backbone_out_channels = {}  # Dict mapping layer index to channel count
        
        # Mark task for downstream components
        self.task = "stereo3ddet"
        
        # Re-enable verbose output if requested
        if verbose:
            self.info()
            from ultralytics.utils import LOGGER
            LOGGER.info("")

    def _process_backbone(self, x):
        """Process single 3-channel image through backbone layers.
        
        Args:
            x: [B, 3, H, W] tensor
            
        Returns:
            List of all backbone layer outputs (for skip connections).
        """
        y = []
        for i in range(self.backbone_end_idx + 1):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [y[j] if j != -1 else x for j in m.f]
            x = m(x)
            y.append(x)
        return y

    def _fuse_stereo_features(self, y_left, y_right, device):
        """Fuse left/right features with channel reduction.
        
        Args:
            y_left: List of left backbone outputs
            y_right: List of right backbone outputs
            device: Device for fusion convs
            
        Returns:
            List with fused features at saved layers, None elsewhere.
        """
        saved_layers = [i for i in range(self.backbone_end_idx + 1) 
                        if i in self.save or i == self.backbone_end_idx]
        
        # Lazy init fusion convs
        for i in saved_layers:
            key = str(i)
            if key not in self.fusion_convs:
                ch = y_left[i].shape[1]
                self.fusion_convs[key] = Conv(ch * 2, ch, k=1, s=1, p=0, act=True).to(device)
        
        # Fuse and reduce channels
        y_fused = []
        for i in range(self.backbone_end_idx + 1):
            if i in saved_layers:
                fused = torch.cat([y_left[i], y_right[i]], dim=1)
                y_fused.append(self.fusion_convs[str(i)](fused))
            else:
                y_fused.append(None)
        return y_fused

    def _forward_head(self, y_fused, profile=False, visualize=False, embed=None):
        """Process head layers using parent's _predict_once pattern.
        
        Args:
            y_fused: List of fused backbone features (None for non-saved layers)
            profile: Print computation time of each layer if True
            visualize: Save feature maps if True
            embed: List of feature vectors/embeddings to return
            
        Returns:
            Model output (dict, tensor, or tuple depending on head type).
        """
        y = [None] * len(self.model)
        # Copy fused backbone outputs
        for i in range(self.backbone_end_idx + 1):
            if y_fused[i] is not None:
                y[i] = y_fused[i]
        
        # Start from fused backbone output
        x = y_fused[self.backbone_end_idx]
        
        # Reuse parent's _predict_once pattern starting from head
        dt, embeddings = [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        
        for i in range(self.backbone_end_idx + 1, len(self.model)):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y[i] = x if m.i in self.save else None
            if visualize:
                from ultralytics.nn.tasks import feature_visualization
                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Override to handle weight-shared backbone for stereo input.
        
        Args:
            x: Input tensor [B, 6, H, W] (stereo) or [B, 3, H, W] (initialization)
            profile: Print computation time of each layer if True
            visualize: Save feature maps if True
            embed: List of feature vectors/embeddings to return
            
        Returns:
            Model output (dict, tensor, or tuple depending on head type).
        """
        # Handle 3-channel input (initialization)
        if x.shape[1] == 3:
            return super()._predict_once(x, profile, visualize, embed)
        
        # Split 6-channel stereo input
        left, right = x[:, :3], x[:, 3:]
        
        # Process through shared backbone
        y_left = self._process_backbone(left)
        y_right = self._process_backbone(right)
        
        # Fuse features
        y_fused = self._fuse_stereo_features(y_left, y_right, x.device)
        
        # Continue with head using parent's _predict_once pattern
        return self._forward_head(y_fused, profile, visualize, embed)

    def init_criterion(self):
        """Initialize the loss criterion for the Stereo3DDetModel.
        
        Returns:
            Stereo3DDetLossYOLO11P3: Loss criterion for stereo 3D detection.
        
        Note:
            This method is called lazily from loss() when criterion is None.
            At that point, model.args should be set by the trainer.
        """
        from ultralytics.models.yolo.stereo3ddet.loss_yolo11 import Stereo3DDetLossYOLO11P3
        
        # Optional aux weights from YAML training section
        aux_w = None
        if hasattr(self, "yaml") and self.yaml is not None:
            training_config = self.yaml.get("training", {})
            if training_config and "loss_weights" in training_config:
                aux_w = training_config["loss_weights"]
        
        # Note: model.args is expected to be set by trainer before loss() is called
        # If args is not set, the loss class will raise AttributeError
        return Stereo3DDetLossYOLO11P3(self, loss_weights=aux_w)