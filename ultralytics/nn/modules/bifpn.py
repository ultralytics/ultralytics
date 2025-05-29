import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):
    """Convolutional layer with optional Normalization and Activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, nn.Sequential(nn.Identity(), nn.ReLU(inplace=True)) if act else nn.Identity()
        )
        self.norm = nn.BatchNorm2d(c2) if norm else nn.Identity()
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BiFPN_Add(nn.Module):
    """
    BiFPN weighted addition module.

    Performs weighted feature fusion.
    """

    def __init__(self, channels):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001  # Small value to avoid division by zero
        self.conv = ConvNormAct(channels, channels, k=3, s=1, p=1)  # Example conv after fusion

    def forward(self, x_top, x_bottom):
        # Resize x_top to match x_bottom's spatial dimensions if needed
        if x_top.shape[-2:] != x_bottom.shape[-2:]:
            x_top = F.interpolate(x_top, size=x_bottom.shape[-2:], mode="bilinear", align_corners=False)

        w = F.relu(self.w)
        w = w / (torch.sum(w) + self.epsilon)  # Normalize weights

        fused_features = w[0] * x_top + w[1] * x_bottom
        return self.conv(fused_features)


class BiFPN_Add3(nn.Module):
    """
    BiFPN weighted addition module for 3 inputs.

    Performs weighted feature fusion for 3 inputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = ConvNormAct(channels, channels, k=3, s=1, p=1)  # Example conv after fusion

    def forward(self, x_top, x_mid, x_bottom):
        # Resize inputs to match x_mid's spatial dimensions (assuming x_mid is target size)
        if x_top.shape[-2:] != x_mid.shape[-2:]:
            x_top = F.interpolate(x_top, size=x_mid.shape[-2:], mode="bilinear", align_corners=False)
        if x_bottom.shape[-2:] != x_mid.shape[-2:]:
            x_bottom = F.interpolate(x_bottom, size=x_mid.shape[-2:], mode="bilinear", align_corners=False)

        w = F.relu(self.w)
        w = w / (torch.sum(w) + self.epsilon)  # Normalize weights

        fused_features = w[0] * x_top + w[1] * x_mid + w[2] * x_bottom
        return self.conv(fused_features)


# --- Anda juga akan membutuhkan kelas seperti BiFPN_Concat untuk jalur tertentu ---
class BiFPN_Concat(nn.Module):
    """BiFPN concatenation module (similar to PAN/FPN but with possible adjustments or additional processing after
    concatenation).
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        # Ini adalah contoh, Anda mungkin perlu menyesuaikan bagaimana BiFPN mengelola concat.
        # Biasanya BiFPN fokus pada weighted addition. Namun, ada jalur di mana concat mungkin terjadi.
        # Untuk kesederhanaan, kita bisa asumsikan ini hanyalah conv biasa setelah FPN/PAN upsample/downsample
        # sebelum operasi BiFPN Add berikutnya.
        total_in_channels = sum(in_channels_list)
        self.conv = ConvNormAct(total_in_channels, out_channels, k=1, s=1)  # Conv 1x1 to reduce channels

    def forward(self, *inputs):
        # inputs should be a tuple of feature maps
        # Ensure all inputs are upsampled/downsampled to a common resolution before concat
        # This part depends heavily on your BiFPN topology

        # Contoh: Jika Anda menggabungkan 2 fitur, dan yang satu perlu di-upsample
        # Asumsikan inputs[0] adalah fitur yang lebih rendah resolusi dan perlu di-upsample
        # if inputs[0].shape[-2:] != inputs[1].shape[-2:]:
        #    inputs[0] = F.interpolate(inputs[0], size=inputs[1].shape[-2:], mode='bilinear', align_corners=False)

        concatenated_features = torch.cat(inputs, 1)  # Concatenate along channel dimension
        return self.conv(concatenated_features)
