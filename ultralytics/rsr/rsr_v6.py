import torch
import torch.nn as nn
import torch.nn.functional as F
# Version= v6
#  the most similar to the paper

class RSR(nn.Module):
    def __init__(self, topk_ratio_rgb=0.75, topk_ratio_ir=0.75, use_soft_filter=True):
        super(RSR, self).__init__()
        self.topk_ratio_rgb = topk_ratio_rgb
        self.topk_ratio_ir = topk_ratio_ir
        self.use_soft_filter = use_soft_filter

        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # operates on amplitude
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )
        self.ir_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((32, 32))
        )

    def generate_filter(self, amp, encoder, topk_ratio, use_soft_filter):
        B, _, H, W = amp.shape
        feat = encoder(amp)  # (B, 16, 32, 32)
        B, C, Hf, Wf = feat.shape
        flat_feat = feat.view(B, -1)

        k = int(flat_feat.shape[1] * topk_ratio)
        topk_vals, topk_idx = torch.topk(flat_feat, k=k, dim=1)

        mask = torch.zeros_like(flat_feat)
        if use_soft_filter:
            mask.scatter_(1, topk_idx, topk_vals)
            mask = F.normalize(mask, p=1, dim=1)
        else:
            mask.scatter_(1, topk_idx, 1.0)

        # Reshape to (B, 16, 32, 32)
        mask = mask.view(B, C, Hf, Wf)
        # Upsample to (B, 16, H, W)
        mask_up = F.interpolate(mask, size=(H, W), mode='nearest')
        # Average across channels to get (B, 1, H, W)
        mask_up = mask_up.mean(dim=1, keepdim=True)
        return mask_up

    def forward(self, x):
        if isinstance(x, dict):
            x = x["img"]

        assert x.ndim == 4 and x.shape[1] == 4, f"Expected [B, 4, H, W], got {x.shape}"
        B, _, H, W = x.shape

        rgb, ir = x[:, :3], x[:, 3:]  # [B, 3, H, W], [B, 1, H, W]

        rgb_fft = torch.fft.fft2(rgb, norm='ortho')
        ir_fft = torch.fft.fft2(ir, norm='ortho')

        amp_rgb = torch.abs(rgb_fft).mean(dim=1, keepdim=True)
        amp_ir = torch.abs(ir_fft).mean(dim=1, keepdim=True)
        phase_rgb = torch.angle(rgb_fft)
        phase_ir = torch.angle(ir_fft)

        filter_rgb = self.generate_filter(amp_rgb, self.rgb_encoder, self.topk_ratio_rgb, self.use_soft_filter)
        filter_ir = self.generate_filter(amp_ir, self.ir_encoder, self.topk_ratio_ir, self.use_soft_filter)

        filter_rgb = filter_rgb.expand(-1, 3, -1, -1)
        filter_ir = filter_ir.expand(-1, 1, -1, -1)

        real_rgb = torch.cos(phase_rgb) * torch.abs(rgb_fft)
        imag_rgb = torch.sin(phase_rgb) * torch.abs(rgb_fft)
        complex_rgb = torch.complex(real_rgb, imag_rgb)

        real_ir = torch.cos(phase_ir) * torch.abs(ir_fft)
        imag_ir = torch.sin(phase_ir) * torch.abs(ir_fft)
        complex_ir = torch.complex(real_ir, imag_ir)

        filtered_rgb_fft = complex_rgb * filter_rgb
        filtered_ir_fft = complex_ir * filter_ir

        rgb_filtered = torch.fft.ifft2(filtered_rgb_fft, norm='ortho').real
        ir_filtered = torch.fft.ifft2(filtered_ir_fft, norm='ortho').real

        return torch.cat([rgb_filtered, ir_filtered], dim=1)
