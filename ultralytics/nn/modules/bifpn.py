import torch
import torch.nn as nn
import torch.nn.functional as F

# Pastikan Conv diimpor atau didefinisikan di sini
# Jika Conv tidak ada di file bifpn.py, Anda perlu mengimpornya dari ultralytics.nn.modules.conv
# Contoh: from ultralytics.nn.modules.conv import Conv, DWConv
# Atau definisikan ulang Conv dan DWConv jika Anda tidak ingin impor eksternal di dalam module ini.
# Untuk kesederhanaan, saya akan asumsikan Conv dan DWConv bisa diimpor.
from ultralytics.nn.modules.conv import Conv  # <--- PENTING: Tambahkan baris ini


class ConvNormAct(nn.Module):
    """Convolutional layer with optional Normalization and Activation."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, norm=True):
        super().__init__()
        # Gunakan autopad jika tersedia, atau hitung padding manual
        if p is None:
            p = k // 2  # Hitung padding secara otomatis untuk k=3, p=1
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)  # Bias False karena ada BatchNorm
        self.norm = nn.BatchNorm2d(c2) if norm else nn.Identity()
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class BiFPN_Add(nn.Module):
    """
    BiFPN weighted addition module.

    Performs weighted feature fusion for 2 inputs.
    """

    def __init__(self, channels):
        super().__init__()
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001  # Small value to avoid division by zero
        self.conv = ConvNormAct(channels, channels, k=3, s=1, p=1)  # Conv setelah fusi

    def forward(self, x_top, x_bottom):
        # Resize x_top to match x_bottom's spatial dimensions if needed
        # Ini terjadi pada jalur top-down (upsample)
        if x_top.shape[-2:] != x_bottom.shape[-2:]:
            x_top = F.interpolate(x_top, size=x_bottom.shape[-2:], mode="bilinear", align_corners=False)

        w = F.relu(self.w)
        w = w / (torch.sum(w) + self.epsilon)  # Normalisasi bobot

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
        self.conv = ConvNormAct(channels, channels, k=3, s=1, p=1)  # Conv setelah fusi

    def forward(self, x_top, x_mid, x_bottom):
        # Resize inputs to match x_mid's spatial dimensions (assuming x_mid is target size)
        # Ini terjadi pada jalur bottom-up (upsample/downsample)
        if x_top.shape[-2:] != x_mid.shape[-2:]:
            x_top = F.interpolate(x_top, size=x_mid.shape[-2:], mode="bilinear", align_corners=False)
        if x_bottom.shape[-2:] != x_mid.shape[-2:]:
            x_bottom = F.interpolate(x_bottom, size=x_mid.shape[-2:], mode="bilinear", align_corners=False)

        w = F.relu(self.w)
        w = w / (torch.sum(w) + self.epsilon)  # Normalisasi bobot

        fused_features = w[0] * x_top + w[1] * x_mid + w[2] * x_bottom
        return self.conv(fused_features)


class BiFPN_Concat(nn.Module):
    """
    BiFPN concatenation module (bukan operasi fusi berbobot utama BiFPN, lebih ke preprocessing).

    Biasanya BiFPN fokus pada weighted addition.
    """

    def __init__(self, in_channels_list, out_channels):
        super().__init__()
        total_in_channels = sum(in_channels_list)
        self.conv = ConvNormAct(total_in_channels, out_channels, k=1, s=1)  # Conv 1x1 untuk mengurangi channel

    def forward(self, *inputs):
        # inputs seharusnya tuple dari feature maps
        # Pastikan semua input sudah di-upsample/downsample ke resolusi umum sebelum concat
        # Ini tergantung topologi BiFPN spesifik Anda.

        # Contoh: Jika Anda menggabungkan 2 fitur, dan yang satu perlu di-upsample
        # if inputs[0].shape[-2:] != inputs[1].shape[-2:]:
        #    inputs[0] = F.interpolate(inputs[0], size=inputs[1].shape[-2:], mode='bilinear', align_corners=False)

        concatenated_features = torch.cat(inputs, 1)  # Konkat sepanjang dimensi channel
        return self.conv(concatenated_features)


# --- Kelas BiFPNBlock yang diperlukan oleh BiFPN ---
class BiFPNBlock(nn.Module):
    """Single block of Bi-directional Feature Pyramid Network."""

    def __init__(self, c_out, epsilon=0.0001):  # c_out adalah channel output untuk semua level
        super().__init__()
        self.epsilon = epsilon

        # Konvolusi untuk setiap jalur
        # p_td: top-down path convs
        self.p6_td = BiFPN_Add(c_out)
        self.p5_td = BiFPN_Add(c_out)
        self.p4_td = BiFPN_Add(c_out)
        self.p3_td = BiFPN_Add(c_out)  # Konvolusi di P3

        # p_out: bottom-up path convs
        self.p4_out = BiFPN_Add3(c_out)  # Untuk P4, menerima 3 input (P4_backbone, P4_td, P3_out upsampled)
        self.p5_out = BiFPN_Add3(c_out)  # Untuk P5, menerima 3 input
        self.p6_out = BiFPN_Add3(c_out)  # Untuk P6, menerima 3 input
        self.p7_out = BiFPN_Add3(c_out)  # Untuk P7, menerima 3 input

    def forward(self, inputs):
        # inputs: [p3_x, p4_x, p5_x, p6_x, p7_x]
        # p3_x, p4_x, p5_x, p6_x, p7_x adalah fitur yang sudah memiliki channel yang sama (c_out)

        p3_x, p4_x, p5_x, p6_x, p7_x = inputs

        # --- Top-Down Pathway ---
        # P7_td adalah P7_x (fitur paling dalam)
        p7_td = p7_x

        # P6_td: fusi P6_x dan upsample P7_td
        p6_td = self.p6_td(p6_x, F.interpolate(p7_td, scale_factor=2, mode="nearest"))

        # P5_td: fusi P5_x dan upsample P6_td
        p5_td = self.p5_td(p5_x, F.interpolate(p6_td, scale_factor=2, mode="nearest"))

        # P4_td: fusi P4_x dan upsample P5_td
        p4_td = self.p4_td(p4_x, F.interpolate(p5_td, scale_factor=2, mode="nearest"))

        # P3_td: fusi P3_x dan upsample P4_td
        p3_td = self.p3_td(p3_x, F.interpolate(p4_td, scale_factor=2, mode="nearest"))

        # --- Bottom-Up Pathway ---
        # P3_out adalah P3_td
        p3_out = p3_td

        # P4_out: fusi P4_x (original), P4_td (top-down), dan downsample P3_out
        p4_out = self.p4_out(p4_x, p4_td, F.interpolate(p3_out, scale_factor=0.5, mode="nearest"))

        # P5_out: fusi P5_x (original), P5_td (top-down), dan downsample P4_out
        p5_out = self.p5_out(p5_x, p5_td, F.interpolate(p4_out, scale_factor=0.5, mode="nearest"))

        # P6_out: fusi P6_x (original), P6_td (top-down), dan downsample P5_out
        p6_out = self.p6_out(p6_x, p6_td, F.interpolate(p5_out, scale_factor=0.5, mode="nearest"))

        # P7_out: fusi P7_x (original), P7_td (top-down), dan downsample P6_out
        p7_out = self.p7_out(p7_x, p7_td, F.interpolate(p6_out, scale_factor=0.5, mode="nearest"))

        return [p3_out, p4_out, p5_out, p6_out, p7_out]  # Mengembalikan list 5 fitur


# --- Kelas BiFPN (Wrapper utama) ---
class BiFPN(nn.Module):
    """Bi-directional Feature Pyramid Network wrapper."""

    def __init__(self, c1, c2, n=1, epsilon=0.0001):  # c1 bisa int atau list [c3_in, c4_in, c5_in]
        super().__init__()

        if isinstance(c1, (list, tuple)):
            # c1 adalah list dari channel input P3, P4, P5 dari backbone
            c3_in, c4_in, c5_in = c1
        else:
            # Jika c1 hanya satu int, asumsikan semua input channel sama
            c3_in = c4_in = c5_in = c1

        # Konvolusi 1x1 untuk menyesuaikan channel dari input backbone ke channel BiFPN (c2)
        self.p3_conv = Conv(c3_in, c2, 1)
        self.p4_conv = Conv(c4_in, c2, 1)
        self.p5_conv = Conv(c5_in, c2, 1)

        # p6_in didapatkan dari downsampling P5_in
        self.p6_in_conv = Conv(c5_in, c2, 3, 2)

        # p7_in didapatkan dari downsampling P6_in
        self.p7_in_conv = Conv(c2, c2, 3, 2)

        bifpns = []
        for _ in range(n):  # n adalah jumlah pengulangan BiFPNBlock
            bifpns.append(BiFPNBlock(c2, epsilon))
        self.bifpn_blocks = nn.Sequential(*bifpns)  # Ganti nama variable agar tidak ambigu

    def forward(self, x):
        # Input x diharapkan berupa list/tuple dari 3 fitur: [P3_backbone, P4_backbone, P5_backbone]
        if not isinstance(x, (list, tuple)):
            raise ValueError("BiFPN expects a list or tuple of 3 input feature maps (P3, P4, P5).")
        if len(x) != 3:
            raise ValueError(f"BiFPN expects exactly 3 inputs (P3, P4, P5), got {len(x)}")

        c3_feat, c4_feat, c5_feat = x

        # Sesuaikan channel input dari backbone ke channel BiFPN (c2)
        p3_in = self.p3_conv(c3_feat)
        p4_in = self.p4_conv(c4_feat)
        p5_in = self.p5_conv(c5_feat)

        # Hitung P6_in dan P7_in dari P5_in dan P6_in
        p6_in = self.p6_in_conv(c5_feat)  # P6 dari P5 backbone
        p7_in = self.p7_in_conv(p6_in)  # P7 dari P6 yang baru dihitung

        # Kumpulkan semua fitur untuk BiFPNBlock
        features_for_bifpn_block = [p3_in, p4_in, p5_in, p6_in, p7_in]

        # Jalankan BiFPN blocks
        # Output dari self.bifpn_blocks akan berupa list dari 5 tensor: [p3_out, p4_out, p5_out, p6_out, p7_out]
        fused_features = self.bifpn_blocks(features_for_bifpn_block)

        # Mengembalikan hanya P3_out, P4_out, P5_out sebagai tuple untuk Detect head
        return fused_features[0], fused_features[1], fused_features[2]  # P3_out, P4_out, P5_out
