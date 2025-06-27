import torch
import torch.nn as nn
import torch.nn.functional as F
# version = v2 and v3 
# V3 is different in adding the learnable weight alpha in the forward function
# Redundant Spectrum Removal (RSR) Module
class RSR(nn.Module):
    def __init__(self, topk_rgb=2688, topk_ir=896): # 75% form rgb and IR channels
        super(RSR, self).__init__()
        # Store them separately
        self.topk_rgb = topk_rgb
        self.topk_ir = topk_ir

        # --- Check these layer definitions ---
        self.rgb_embed = nn.Sequential(
            # This Conv2d dictates the channels for the RGB path mask calculation
            nn.Conv2d(3, 3, kernel_size=3, padding=1),  # Should output 3 channels
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.ir_embed = nn.Sequential(
            # This Conv2d dictates the channels for the IR path mask calculation
            nn.Conv2d(1, 1, kernel_size=3, padding=1),  # Should output 1 channel
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        # --- End Check ---

    def forward(self, x):
        if isinstance(x, dict):
            x = x["img"]

        assert x.ndim == 4, f"Expected 4D input (B, C, H, W), got {x.shape}"
        print(f"🔍 RSR Input Shape: {x.shape}") # Confirms [4, 4, 512, 512]

        rgb, ir = x[:, :3, :, :], x[:, 3:, :, :] # rgb=[B,3,H,W], ir=[B,1,H,W]

        rgb_freq = torch.fft.fft2(rgb, norm='ortho') # [B,3,H,W] complex
        ir_freq = torch.fft.fft2(ir, norm='ortho')   # [B,1,H,W] complex

        rgb_feat = self.rgb_embed(rgb) # Shape depends on rgb_embed Conv2d
        ir_feat = self.ir_embed(ir)   # Shape depends on ir_embed Conv2d

        B, C_rgb_feat, H_feat, W_feat = rgb_feat.shape # Get actual channels from rgb_feat
        _, C_ir_feat, _, _ = ir_feat.shape            # Get actual channels from ir_feat

        D_rgb = C_rgb_feat * H_feat * W_feat
        D_ir = C_ir_feat * H_feat * W_feat

        # Use the specific topk values, ensuring they don't exceed the total dimension
        k_rgb = min(self.topk_rgb, D_rgb)
        k_ir = min(self.topk_ir, D_ir)

        rgb_feat_flat = rgb_feat.view(B, -1)
        ir_feat_flat = ir_feat.view(B, -1)

        _, rgb_indices = torch.topk(rgb_feat_flat, self.topk_rgb, dim=1)
        _, ir_indices = torch.topk(ir_feat_flat, self.topk_ir, dim=1)

        def create_mask(indices, total_size, C_feat, H_feat, W_feat):
            mask = torch.zeros((indices.size(0), total_size), device=indices.device)
            mask.scatter_(1, indices, 1.0)
            return mask.view(B, C_feat, H_feat, W_feat) # Use actual feature channels

        rgb_mask = create_mask(rgb_indices, D_rgb, C_rgb_feat, H_feat, W_feat) # [B, C_rgb_feat, 32, 32]
        ir_mask = create_mask(ir_indices, D_ir, C_ir_feat, H_feat, W_feat)     # [B, C_ir_feat, 32, 32]

        _, _, Hf, Wf = rgb.shape
        # Interpolate masks - crucial step, check channel numbers here
        # align_corners=False is generally recommended for feature maps
        rgb_mask_up = F.interpolate(rgb_mask, size=(Hf, Wf), mode='bilinear', align_corners=False) # [B, C_rgb_feat, Hf, Wf]
        ir_mask_up = F.interpolate(ir_mask, size=(Hf, Wf), mode='bilinear', align_corners=False)   # [B, C_ir_feat, Hf, Wf]

        # Check for broadcast errors if C_xxx_feat doesn't match frequency channels
        # This check assumes C_rgb_feat=3 and C_ir_feat=1 based on corrected embed layers
        assert rgb_mask_up.shape[1] == rgb_freq.shape[1], f"RGB Mask/Freq channel mismatch: {rgb_mask_up.shape[1]} vs {rgb_freq.shape[1]}"
        assert ir_mask_up.shape[1] == ir_freq.shape[1], f"IR Mask/Freq channel mismatch: {ir_mask_up.shape[1]} vs {ir_freq.shape[1]}"

        # Apply masks
        # These multiplications preserve the channels of the *frequency* data
        rgb_freq_filtered = rgb_freq * rgb_mask_up # [B, 3, Hf, Wf] * [B, C_rgb_feat, Hf, Wf] -> Error if C_rgb_feat != 3
        ir_freq_filtered = ir_freq * ir_mask_up   # [B, 1, Hf, Wf] * [B, C_ir_feat, Hf, Wf] -> Error if C_ir_feat != 1

        # Inverse FFT - preserves channels of the input frequency data
        rgb_filtered = torch.fft.ifft2(rgb_freq_filtered, norm='ortho').real # Should be [B, 3, Hf, Wf]
        ir_filtered = torch.fft.ifft2(ir_freq_filtered, norm='ortho').real   # Should be [B, 1, Hf, Wf]

        # --- The Final Concatenation ---
        output = torch.cat([rgb_filtered, ir_filtered], dim=1)
        print(f"🔍 RSR Output Shape: {output.shape}") # You saw [4, 6, 512, 512] here

        return output