import torch
import torch.nn as nn
import torch.nn.functional as F

# --- CBAM Module (used for local attention replacement) ---
class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        ca = torch.sigmoid(avg_out + max_out)
        x = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * sa
        return x

# --- Cross-Attention Layer ---
class CrossAttention(nn.Module):
    def __init__(self, in_channels, window_size=16, num_heads=4):
        super().__init__()
        self.in_channels = in_channels
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads

        self.q_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, query, key_value):
        B, C, H, W = query.shape
        W_size = self.window_size

        # Pad to be divisible by window size
        pad_h = (W_size - H % W_size) % W_size
        pad_w = (W_size - W % W_size) % W_size
        query = F.pad(query, (0, pad_w, 0, pad_h))
        key_value = F.pad(key_value, (0, pad_w, 0, pad_h))
        H_pad, W_pad = query.shape[-2:]

        def partition(x):
            x = x.unfold(2, W_size, W_size).unfold(3, W_size, W_size)
            return x.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, W_size, W_size)

        def reverse(x, H_out, W_out):
            B_ = x.shape[0] // ((H_pad // W_size) * (W_pad // W_size))
            x = x.reshape(B_, H_pad // W_size, W_pad // W_size, C, W_size, W_size)
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B_, C, H_pad, W_pad)
            return x[:, :, :H_out, :W_out]

        Q = partition(self.q_proj(query))
        K = partition(self.k_proj(key_value))
        V = partition(self.v_proj(key_value))

        B_, C, H_w, W_w = Q.shape
        N = H_w * W_w

        Q = Q.view(B_, self.num_heads, self.head_dim, N).transpose(2, 3)
        K = K.view(B_, self.num_heads, self.head_dim, N).transpose(2, 3)
        V = V.view(B_, self.num_heads, self.head_dim, N).transpose(2, 3)

        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_probs = torch.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, V)
        attn_output = attn_output.transpose(2, 3).reshape(B_, C, H_w, W_w)

        return self.out_proj(reverse(attn_output, H, W))

# --- Global Attention (8x8 partition) ---
class GlobalAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_pooled = F.adaptive_avg_pool2d(x, (8, 8))
        attn = self.conv2(F.relu(self.bn1(self.conv1(x_pooled))))
        attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
        attn = torch.sigmoid(attn)
        return x * attn + x

# --- Enhanced CMAF Module with revised cross-attention and learnable fusion ---
class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embed_dim=96, num_heads=4):
        super().__init__()
        self.embed_dim = embed_dim

        self.embeds = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, embed_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(embed_dim),
                nn.ReLU()
            ) for _ in range(4)
        ])

        self.cbams = nn.ModuleList([CBAM(embed_dim) for _ in range(4)])
        self.cross_attn = CrossAttention(embed_dim, window_size=16, num_heads=4)
        self.global_attn = GlobalAttention(embed_dim)
        self.out_proj = nn.Conv2d(embed_dim, 3, kernel_size=1)
        self.skip_gate = nn.Parameter(torch.tensor(0.5))

        # Learnable fusion weights for 4 local + 4 cross-attended outputs
        self.fusion_weights = nn.Parameter(torch.ones(8))

    def forward(self, x):
        assert x.shape[1] == 4, "Input must have 4 channels (R, G, B, IR)"
        B, _, H, W = x.shape
        inputs = [x[:, i:i+1] for i in range(4)]

        # Embedding + local CBAM attention
        feats = [cbam(embed(inp)) for inp, embed, cbam in zip(inputs, self.embeds, self.cbams)]

        # Fixed cross-attention pairs: R↔G, G↔B, B↔IR, IR↔R
        RG = self.cross_attn(feats[0], feats[1])
        GB = self.cross_attn(feats[1], feats[2])
        BI = self.cross_attn(feats[2], feats[3])
        IRR = self.cross_attn(feats[3], feats[0])

        # Learnable fusion
        all_feats = feats + [RG, GB, BI, IRR]
        weights = torch.softmax(self.fusion_weights, dim=0)
        fused = sum(w * f for w, f in zip(weights, all_feats))

        # Global attention refinement on fused output
        refined = self.global_attn(fused)

        # Final projection and skip connection
        projected = torch.tanh(self.out_proj(refined))
        output = self.skip_gate * projected + (1 - self.skip_gate) * x[:, :3, ::2, ::2]  # match resolution
        output_upsampled = F.interpolate(output, size=(512, 512), mode='bilinear', align_corners=False)
        return torch.clamp(output_upsampled, 0, 1)
