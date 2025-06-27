import torch
import torch.nn as nn
import torch.nn.functional as F

# version = 5
class RSR(nn.Module):
    def __init__(self, topk_ratio_rgb=0.75, topk_ratio_ir=0.75, pool_scale=16):
        """
        topk_ratio_*: Fraction of the spectral features to retain
        pool_scale: Defines the relative downsampling factor (e.g., 16 -> feature map is 1/16 of input resolution)
        """
        super(RSR, self).__init__()
        self.topk_ratio_rgb = topk_ratio_rgb
        self.topk_ratio_ir = topk_ratio_ir
        self.pool_scale = pool_scale

        # RGB & IR convolutional feature reducers
        self.rgb_conv = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.ir_conv = nn.Conv2d(1, 1, kernel_size=3, padding=1)

    def forward(self, x):
        if isinstance(x, dict):
            x = x["img"]  # Handle YOLO dict-style inputs

        assert x.ndim == 4, f"Expected 4D input (B, C, H, W), got {x.shape}"
        B, C, H, W = x.shape
        rgb, ir = x[:, :3, :, :], x[:, 3:, :, :]  # Split channels

        # 🔹 Fourier Transform
        rgb_freq = torch.fft.fft2(rgb, norm='ortho')
        ir_freq = torch.fft.fft2(ir, norm='ortho')

        # 🔹 Feature Map Pooling Size Based on Resolution
        pool_h, pool_w = max(H // self.pool_scale, 8), max(W // self.pool_scale, 8)

        # 🔹 Reduce & Pool Spatially to Extract Frequency Importance Map
        rgb_feat = F.adaptive_avg_pool2d(F.relu(self.rgb_conv(rgb)), (pool_h, pool_w))  # [B, 3, H//scale, W//scale]
        ir_feat = F.adaptive_avg_pool2d(F.relu(self.ir_conv(ir)), (pool_h, pool_w))     # [B, 1, H//scale, W//scale]

        # 🔹 Flatten and Compute Top-k Indices
        rgb_feat_flat = rgb_feat.view(B, -1)
        ir_feat_flat = ir_feat.view(B, -1)
        D_rgb, D_ir = rgb_feat_flat.shape[1], ir_feat_flat.shape[1]

        k_rgb = int(D_rgb * self.topk_ratio_rgb)
        k_ir = int(D_ir * self.topk_ratio_ir)

        _, rgb_indices = torch.topk(rgb_feat_flat, k_rgb, dim=1)
        _, ir_indices = torch.topk(ir_feat_flat, k_ir, dim=1)

        # 🔹 Build Binary Masks
        def create_mask(indices, D, C, Hf, Wf):
            mask = torch.zeros((indices.size(0), D), device=indices.device)
            mask.scatter_(1, indices, 1.0)
            return mask.view(-1, C, Hf, Wf)

        rgb_mask = create_mask(rgb_indices, D_rgb, rgb_feat.shape[1], pool_h, pool_w)
        ir_mask = create_mask(ir_indices, D_ir, ir_feat.shape[1], pool_h, pool_w)

        # 🔹 Upsample masks to match original image size
        rgb_mask_up = F.interpolate(rgb_mask, size=(H, W), mode='bilinear', align_corners=False)
        ir_mask_up = F.interpolate(ir_mask, size=(H, W), mode='bilinear', align_corners=False)

        # 🔹 Apply frequency-domain masking
        assert rgb_mask_up.shape[1] == rgb_freq.shape[1], f"RGB mismatch: {rgb_mask_up.shape[1]} vs {rgb_freq.shape[1]}"
        assert ir_mask_up.shape[1] == ir_freq.shape[1], f"IR mismatch: {ir_mask_up.shape[1]} vs {ir_freq.shape[1]}"

        rgb_freq_filtered = rgb_freq * rgb_mask_up
        ir_freq_filtered = ir_freq * ir_mask_up

        # 🔹 Inverse FFT to reconstruct filtered spatial images
        rgb_filtered = torch.fft.ifft2(rgb_freq_filtered, norm='ortho').real
        ir_filtered = torch.fft.ifft2(ir_freq_filtered, norm='ortho').real

        # 🔹 Concatenate back into multimodal image
        output = torch.cat([rgb_filtered, ir_filtered], dim=1)

        return output
