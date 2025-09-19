# Ultralytics YOLO 🚀, AGPL-3.0 license

"""
MDE (Monocular Depth Estimation) head for YOLO models.

This module contains the Detect_MDE class that extends YOLO detection with depth estimation capabilities.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import DFL
from ultralytics.nn.modules.conv import Conv, DWConv


class Detect_MDE(nn.Module):
    """
    YOLO Detect head with Monocular Depth Estimation (MDE).
    
    This class extends the standard YOLO detection head to include depth estimation
    for each detected object. It predicts bounding boxes, class probabilities, and
    depth values simultaneously.
    
    Attributes:
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor (includes depth channel).
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        cv_depth (nn.ModuleList): Convolution layers for depth estimation.
        dfl (nn.Module): Distribution Focal Loss layer.
        beta (float): Depth activation parameter.
        
    Methods:
        forward: Perform forward pass and return predictions with depth.
        depth_activation: Apply log-sigmoid activation for depth prediction.
        
    Examples:
        Create an MDE detection head for 5 classes (KITTI dataset)
        >>> mde_head = Detect_MDE(nc=5, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = mde_head(x)
    """
    
    # Class attributes for compatibility with YOLO framework
    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    end2end = False  # end2end
    max_det = 300  # max_det
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output
    
    def __init__(self, nc: int = 80, ch: tuple = (), reg_max: int = 16, beta: float = -14.4):
        """
        Initialize the MDE detection head.
        
        Args:
            nc (int): Number of classes.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
            reg_max (int): DFL channels for box regression.
            beta (float): Depth activation parameter (default: -14.4 from paper).
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels
        self.no = nc + self.reg_max * 4 + 1  # number of outputs per anchor (+1 for depth)
        self.stride = torch.zeros(self.nl)  # strides computed during build
        self.beta = beta  # depth activation parameter
        
        # Channel dimensions
        c2 = max((16, ch[0] // 4, self.reg_max * 4))  # box regression channels
        c3 = max(ch[0], min(self.nc, 100))  # classification channels
        c_depth = max(ch[0], min(self.nc, 100))  # depth estimation channels
        
        # Box regression branch
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c2, 3), 
                Conv(c2, c2, 3), 
                nn.Conv2d(c2, 4 * self.reg_max, 1)
            ) for x in ch
        )
        
        # Classification branch
        self.cv3 = nn.ModuleList(
            nn.Sequential(
                Conv(x, c3, 3), 
                Conv(c3, c3, 3), 
                nn.Conv2d(c3, self.nc, 1)
            ) for x in ch
        )
        
        # Depth estimation branch
        self.cv_depth = nn.ModuleList(
            nn.Sequential(
                Conv(x, c_depth, 3), 
                Conv(c_depth, c_depth, 3), 
                nn.Conv2d(c_depth, 1, 1)  # Single depth value per anchor
            ) for x in ch
        )
        
        # Distribution Focal Loss for box regression
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
    
    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Forward pass through the MDE head.
        
        Args:
            x (list[torch.Tensor]): List of feature maps from different scales.
            
        Returns:
            list[torch.Tensor]: List of predictions with [box, class, depth] concatenated.
        """
        for i in range(self.nl):
            # Box regression prediction
            box = self.cv2[i](x[i])
            
            # Classification prediction
            cls = self.cv3[i](x[i])
            
            # Depth prediction (no activation here - let loss handle it)
            depth = self.cv_depth[i](x[i])
            
            # Concatenate all predictions: [box, class, depth]
            x[i] = torch.cat((box, cls, depth), 1)
        
        return x
    
    def depth_activation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply log-sigmoid activation for depth prediction.
        
        This implements the depth activation function from the paper:
        fd = β * log(sigmoid(Od)) where β = -14.4
        
        Args:
            x (torch.Tensor): Raw depth predictions.
            
        Returns:
            torch.Tensor: Activated depth predictions.
        """
        # Add small epsilon to prevent log(0)
        return self.beta * torch.log(torch.sigmoid(x) + 1e-7)
    
    def bias_init(self):
        """Initialize detection head biases."""
        import math
        # Initialize classification bias to -log((1 - π) / π) where π = 0.01
        for a, b, s in zip(self.cv2, self.cv3, self.stride):
            a[-1].bias.data[:] = 1.0  # box (4*(reg_max+1))
            b[-1].bias.data[:self.nc] = math.log(5 / self.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
        
        # Initialize depth bias to predict reasonable depth values
        for d in self.cv_depth:
            d[-1].bias.data[:] = 0.0  # depth (start with neutral prediction)
    
    def forward_end2end(self, x: list[torch.Tensor]) -> tuple:
        """
        Forward pass for end-to-end detection (if needed).
        
        Args:
            x (list[torch.Tensor]): List of feature maps.
            
        Returns:
            tuple: End-to-end predictions.
        """
        # This would be implemented if end-to-end training is needed
        # For now, return the standard forward pass
        return self.forward(x)
