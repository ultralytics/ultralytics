import torch
import torch.nn as nn
import torch.nn.functional as F
# version = v4
# Redundant Spectrum Removal (RSR) Module
class RSR(nn.Module):
    def __init__(self): # 75% form rgb and IR channels
        super(RSR, self).__init__()

        # Learnable soft mask generators for RGB and IR
        self.rgb_mask_gen = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(3, 3, kernel_size=1),
            nn.Sigmoid()
        )

        self.ir_mask_gen = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        if isinstance(x, dict):
            x = x["img"]

        assert x.ndim == 4, f"Expected 4D input (B, C, H, W), got {x.shape}"
        print(f"🔍 RSR Input Shape: {x.shape}") # Confirms [4, 4, 512, 512]

        rgb, ir = x[:, :3, :, :], x[:, 3:, :, :] # rgb=[B,3,H,W], ir=[B,1,H,W]

        rgb_freq = torch.fft.fft2(rgb, norm='ortho') # [B,3,H,W] complex
        ir_freq = torch.fft.fft2(ir, norm='ortho')   # [B,1,H,W] complex

        # Generate soft masks in spatial domain
        rgb_mask = self.rgb_mask_gen(rgb)  # [B, 3, H, W], in [0,1]
        ir_mask = self.ir_mask_gen(ir)     # [B, 1, H, W], in [0,1]

        # Optional debug info
        print(f"🧠 Mask mean (RGB): {rgb_mask.mean().item():.3f}, IR: {ir_mask.mean().item():.3f}")

        # Apply soft mask to frequency components (real + imag separately)
        rgb_freq_filtered = rgb_freq.real * rgb_mask + 1j * rgb_freq.imag * rgb_mask
        ir_freq_filtered = ir_freq.real * ir_mask + 1j * ir_freq.imag * ir_mask

        # Inverse FFT to reconstruct filtered image
        rgb_filtered = torch.fft.ifft2(rgb_freq_filtered, norm='ortho').real
        ir_filtered = torch.fft.ifft2(ir_freq_filtered, norm='ortho').real

        # Concatenate and return
        output = torch.cat([rgb_filtered, ir_filtered], dim=1)
        print(f"✅ RSR Output Shape: {output.shape}")

        return output