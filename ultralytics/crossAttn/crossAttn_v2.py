import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvEmbed(nn.Module):
    def __init__(self, in_chans=1, embed_dim=96):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 32, kernel_size=3, stride=2, padding=1),  # 1024 → 512
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),         # 512 → 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),  # 256 → 128
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),

            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1),  # 128 → 64
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.encoder(x)  # Output: (B, embed_dim, 64, 64)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        # linear layers for query, key, value
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, query, key_value):
        B, N, C = query.shape

        # Linear projections
        Q = self.q_proj(query).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        K = self.k_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(key_value).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, heads, N, N)
        attn_probs = F.softmax(attn_scores, dim=-1)

        attn_output = torch.matmul(attn_probs, V)  # (B, heads, N, head_dim)

        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).reshape(B, N, C)

        # Final projection
        output = self.out_proj(attn_output)

        return output

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

        # 3. Return the pooled attention map (before softmax)
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
        self.cross_attn = CrossAttention(embed_dim=embed_dim, num_heads=4)

        # Fusion projection
        self.fused_proj = nn.Linear(embed_dim * 4, embed_dim)

        # Global Attention
        self.global_attn = GlobalAttention(embed_dim)

        #  Normalization
        self.norm = nn.LayerNorm(embed_dim)

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

        # Flatten for attention
        def flat(x): return x.flatten(2).transpose(1, 2)

        R_flat, G_flat = flat(R_weighted), flat(G_weighted)
        B_flat, IR_flat = flat(B_weighted), flat(IR_weighted)

        # Cross-attention pairs
        R_G = self.cross_attn(R_flat, G_flat)
        G_R = self.cross_attn(G_flat, R_flat)
        B_IR = self.cross_attn(B_flat, IR_flat)
        IR_B = self.cross_attn(IR_flat, B_flat)

        # Fuse and project
        fused = torch.cat([R_G, G_R, B_IR, IR_B], dim=-1)
        fused = self.fused_proj(fused)  # (B, N, embed_dim)

        fused = self.norm(fused)  # Normalize across the feature dimension to stabilize feature magnitude before global attention

        # Reshape to feature map
        B_, N, C = fused.shape
        H = W = int(N ** 0.5)
        fused_featmap = fused.transpose(1, 2).reshape(B_, C, H, W)

        # Global Attention
        global_attn_map = self.global_attn(fused_featmap)
        global_attn_map = torch.sigmoid(global_attn_map)
        global_attn_map = F.interpolate(global_attn_map, size=(H, W), mode='bilinear', align_corners=False)

        # Apply and residual
        final_featmap = fused_featmap * global_attn_map + fused_featmap

        # Final projection
        final_output = self.final_conv(final_featmap)

        # Upsample to original size
        final_output = F.interpolate(final_output, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        final_output = torch.sigmoid(final_output)

        print("📊 final_output before skip connection min:", final_output.min().item())
        print("📊 final_output before skip connection max:", final_output.max().item())
        print("📊 final_output before skip connection mean:", final_output.mean().item())
        rgb = x[:, :3, :, :]
        final_output = 0.7 * final_output + 0.3 * rgb

        final_output = torch.clamp(final_output, 0, 1)
        print(f"✅ CrossModalAttentionFusion Output Shape: {final_output.shape}")

        return final_output  # (B, 3, H, W)
