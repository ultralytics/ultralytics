# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.nn.tasks import DetectionModel
from ultralytics.nn.modules.conv import Conv


class Stereo3DDetModel(DetectionModel):
    """Stereo 3D Detection model with weight-shared backbone.

    Architecture:
        1. Split 6-channel input into left (3ch) and right (3ch)
        2. Process both through shared backbone separately
        3. Concatenate backbone outputs -> stereo_concat (preserved for stereo branches)
        4. Fuse via 1x1 conv -> fused features for neck
        5. Process fused through neck -> P3 features
        6. Head receives: P3 for detection/3D, stereo_concat for stereo branches
    """

    def __init__(self, cfg, ch=6, nc=None, verbose=True):
        if isinstance(cfg, dict):
            cfg_dict = cfg.copy()
        else:
            from ultralytics.nn.tasks import yaml_model_load
            cfg_dict = yaml_model_load(cfg)

        # Override to 3 for weight-shared backbone (processes left/right separately)
        cfg_dict["input_channels"] = 3

        # Backbone end index (needed before super().__init__ which calls forward)
        self.backbone_end_idx = len(cfg_dict["backbone"]) - 1

        # Build model with ch=3
        super().__init__(cfg=cfg_dict, ch=3, nc=nc, verbose=False)

        # Lazy-initialized fusion conv (2C -> C)
        self.fusion_conv = None
        self._backbone_channels = None

        self.task = "stereo3ddet"

        if verbose:
            self.info()

    def _get_or_create_fusion_conv(self, channels: int, device) -> Conv:
        """Lazily create 1x1 conv to fuse concatenated left+right features."""
        if self.fusion_conv is None or self._backbone_channels != channels:
            self._backbone_channels = channels
            self.fusion_conv = Conv(
                c1=channels * 2, c2=channels, k=1, s=1, p=0, act=True
            ).to(device)
        return self.fusion_conv

    def _process_backbone(self, x):
        """Process single 3-channel image through backbone layers."""
        y = []
        for i in range(self.backbone_end_idx + 1):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [y[j] if j != -1 else x for j in m.f]
            x = m(x)
            y.append(x)
        return y

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        """Forward pass for stereo 3D detection."""
        # 3-channel input: initialization pass
        if x.shape[1] == 3:
            return super()._predict_once(x, profile, visualize, embed)

        # Split stereo input
        left, right = x[:, :3], x[:, 3:]

        # Process through shared backbone
        y_left = self._process_backbone(left)
        y_right = self._process_backbone(right)

        # Backbone outputs
        backbone_left = y_left[self.backbone_end_idx]
        backbone_right = y_right[self.backbone_end_idx]

        # Stereo concat (preserved for stereo branches at head)
        stereo_concat = torch.cat([backbone_left, backbone_right], dim=1)

        # Fuse for neck processing
        fusion_conv = self._get_or_create_fusion_conv(backbone_left.shape[1], x.device)
        fused = fusion_conv(stereo_concat)

        # Build y_fused with left features for skip connections, fused at backbone end
        y_fused = [None] * (self.backbone_end_idx + 1)
        for i in range(self.backbone_end_idx + 1):
            if i in self.save:
                y_fused[i] = y_left[i]
        y_fused[self.backbone_end_idx] = fused

        return self._forward_neck_and_head(y_fused, stereo_concat)

    def _forward_neck_and_head(self, y_fused, stereo_concat):
        """Process neck layers with fused features, then head with stereo_concat."""
        y = [None] * len(self.model)
        for i in range(self.backbone_end_idx + 1):
            if y_fused[i] is not None:
                y[i] = y_fused[i]

        x = y_fused[self.backbone_end_idx]

        # Find head index
        head_idx = None
        for i in range(self.backbone_end_idx + 1, len(self.model)):
            if hasattr(self.model[i], "stereo_aux"):
                head_idx = i
                break

        # Process neck layers
        for i in range(self.backbone_end_idx + 1, head_idx if head_idx else len(self.model)):
            m = self.model[i]
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y[i] = x if m.i in self.save else None

        # Head with stereo_concat upsampled to P3 resolution
        if head_idx is not None:
            m = self.model[head_idx]
            # Upsample stereo_concat from backbone resolution (P5) to P3 resolution
            p3_h, p3_w = x.shape[2], x.shape[3]
            stereo_upsampled = torch.nn.functional.interpolate(
                stereo_concat, size=(p3_h, p3_w), mode="bilinear", align_corners=False
            )
            x = m([x] if not isinstance(x, list) else x, x_stereo=stereo_upsampled)

        return x

    def init_criterion(self):
        """Initialize the loss criterion."""
        from ultralytics.models.yolo.stereo3ddet.loss_yolo11 import Stereo3DDetLossYOLO11P3

        aux_w = None
        if hasattr(self, "yaml") and self.yaml is not None:
            training_config = self.yaml.get("training", {})
            if training_config and "loss_weights" in training_config:
                aux_w = training_config["loss_weights"]

        return Stereo3DDetLossYOLO11P3(self, loss_weights=aux_w)
