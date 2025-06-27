import torch
import torch.nn as nn
import torch.nn.functional as F
# VERSION = 3
class ConvEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96):
        super().__init__()
        mid_dim = embed_dim // 2

        self.branch1 = nn.Conv2d(in_chans, mid_dim, kernel_size=3, padding=1, stride=2)
        self.branch2 = nn.Conv2d(in_chans, mid_dim, kernel_size=5, padding=2, stride=2)

        self.proj = nn.Conv2d(mid_dim * 2, embed_dim, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        x = torch.cat([b1, b2], dim=1)
        x = self.proj(x)
        return x



class CrossAttention(nn.Module):
    def __init__(self, in_channels, window_size=8, num_heads=4):
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

        assert H % W_size == 0 and W % W_size == 0, "Height and width must be divisible by window_size"

        def partition(x):
            x = x.unfold(2, W_size, W_size).unfold(3, W_size, W_size)  # (B, C, nH, nW, W, W)
            return x.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, W_size, W_size)

        def reverse(x, H, W):
            B_ = x.shape[0] // ((H // W_size) * (W // W_size))
            x = x.reshape(B_, H // W_size, W // W_size, C, W_size, W_size)
            x = x.permute(0, 3, 1, 4, 2, 5).reshape(B_, C, H, W)
            return x

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

        attn_output = reverse(attn_output, H, W)
        return self.out_proj(attn_output)


class LocalAttention(nn.Module):
    def __init__(self, in_channels):
        super(LocalAttention, self).__init__()

        # Apply 3 convolution layers to generate attention map
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        # Compute local attention weights
        attn = self.conv2(F.relu(self.bn2(self.conv1(x))))

        return attn

class GlobalAttention(nn.Module):
    def __init__(self, in_channels, num_partitions=(8, 8)):  # 5 rows, 10 columns
        super(GlobalAttention, self).__init__()
        self.num_partitions = num_partitions  # Defines partitioning of the feature map

        # Global attention feature extractor
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_part, w_part = self.num_partitions

        # 1. Partition-wise Average Pooling (downsampling to partition size)
        x_pooled = F.adaptive_avg_pool2d(x, (h_part, w_part))  # Shape: (B, C, 5, 10)

        # 2. Generate global attention scores
        attn = self.conv2(F.relu(self.bn1(self.conv1(x_pooled))))  # Compute attention scores

        # interpolate back to original size
        attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)  # Shape: (B, C, H, W)
        attn = torch.sigmoid(attn)  # Apply sigmoid to get attention weights

        return attn

class CrossModalAttentionFusion(nn.Module):
    def __init__(self, embed_dim=96):
        super().__init__()

        self.embed_dim = embed_dim

        # Convolutional Embedding
        self.pe_r = ConvEmbed(in_chans=1, embed_dim=embed_dim)
        self.pe_g = ConvEmbed(in_chans=1, embed_dim=embed_dim)
        self.pe_b = ConvEmbed(in_chans=1, embed_dim=embed_dim)
        self.pe_ir = ConvEmbed(in_chans=1, embed_dim=embed_dim)

        # Local Attention
        self.local_attn_r = LocalAttention(embed_dim)
        self.local_attn_g = LocalAttention(embed_dim)
        self.local_attn_b = LocalAttention(embed_dim)
        self.local_attn_ir = LocalAttention(embed_dim)

        # Cross Attention
        self.cross_attn = CrossAttention(embed_dim, window_size=8, num_heads=4)
        # Global Attention
        self.global_attn = GlobalAttention(embed_dim)

        # Final projection to 3-channel image
        self.final_conv = nn.Conv2d(embed_dim, 3, kernel_size=1)

    def forward(self, x):
        """
        Input: x tensor of shape (B, 4, H, W) or dict with 'img'
        Output: tensor of shape (B, 3, H, W)
        """
        if isinstance(x, dict):
            x = x["img"]

        assert x.ndim == 4 and x.shape[1] == 4, f"Expected input of shape (B, 4, H, W), got {x.shape}"

        print(f"🔍 CrossModalAttentionFusion Input Shape: {x.shape}"
              f" (B, C, H, W)")
        R, G, B, IR = x[:, 0:1], x[:, 1:2], x[:, 2:3], x[:, 3:4]

        # Embedding
        R_feat = self.pe_r(R)
        G_feat = self.pe_g(G)
        B_feat = self.pe_b(B)
        IR_feat = self.pe_ir(IR)

        # Local attention
        la_r = self.local_attn_r(R_feat)
        la_g = self.local_attn_g(G_feat)
        la_b = self.local_attn_b(B_feat)
        la_ir = self.local_attn_ir(IR_feat)

        # Softmax across modalities
        attn = torch.softmax(torch.stack([la_r, la_g, la_b, la_ir], dim=1), dim=1)
        la_r, la_g, la_b, la_ir = attn.unbind(dim=1)

        # Apply local attention
        R_weighted = R_feat * la_r
        G_weighted = G_feat * la_g
        B_weighted = B_feat * la_b
        IR_weighted = IR_feat * la_ir

        # Cross-attention pairs
        R_G = self.cross_attn(R_weighted, G_weighted)
        G_R = self.cross_attn(G_weighted, R_weighted)
        B_IR = self.cross_attn(B_weighted, IR_weighted)
        IR_B = self.cross_attn(IR_weighted, B_weighted)

        # OLD Fuse features
        # fused = (R_weighted + G_weighted + B_weighted + IR_weighted) / 4
        # NEW Fuse features
        fused = (R_weighted + G_weighted + B_weighted + IR_weighted + R_G + G_R + B_IR + IR_B) / 8
        
        global_attn_map = self.global_attn(fused)

        global_output = fused * global_attn_map + fused

        # Final projection
        output = self.final_conv(global_output)
        output = F.interpolate(output, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        # Skip connection
        final_output_normalized = torch.sigmoid(output)
        final_output = final_output_normalized + x[:, :3, :, :]
        final_output = torch.clamp(final_output, 0, 1)
        print(f"✅ CrossModalAttentionFusion Output Shape: {final_output.shape}")

        return final_output  # (B, 3, H, W)
