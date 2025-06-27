import torch
import torch.nn as nn
import torch.nn.functional as F

# Inception Block A
class InceptionBlockA(nn.Module):
    def __init__(self, in_channels, out_channels=128):
        super(InceptionBlockA, self).__init__()

        # Ensure `out_channels` is divisible across branches
        assert out_channels % 4 == 0, "out_channels should be divisible by 4 for balance"

        branch_out = out_channels // 4  # Each branch contributes 1/4th of `out_channels`

        # 1x1 Convolution Branch
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        # 3x3 Convolution Branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_out, branch_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        # 5x5 Convolution Branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, branch_out // 2, kernel_size=1),  # Reduce channels before 5x5
            nn.BatchNorm2d(branch_out // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_out // 2, branch_out, kernel_size=5, padding=2),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

        # 3x3 MaxPooling Branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_out, kernel_size=1),
            nn.BatchNorm2d(branch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1x1(x)
        b2 = self.branch3x3(x)
        b3 = self.branch5x5(x)
        b4 = self.branch_pool(x)
        
        return torch.cat([b1, b2, b3, b4], dim=1)  # Concatenate along the channel dimension

class LocalAttention(nn.Module):
    def __init__(self, in_channels):
        super(LocalAttention, self).__init__()

        # Apply 3 convolution layers to generate attention map
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # Compute local attention weights
        attn = self.conv3(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))))

        return attn

class GlobalAttention(nn.Module):
    def __init__(self, in_channels, num_partitions=(8, 8)):  
        super(GlobalAttention, self).__init__()
        self.num_partitions = num_partitions  # Defines partitioning of the feature map

        # Global attention feature extractor
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h_part, w_part = self.num_partitions

        # 1. Partition-wise Average Pooling (downsampling to partition size)
        x_pooled = F.adaptive_avg_pool2d(x, (h_part, w_part))  # Shape: (B, C, 5, 10)

        # 2. Generate global attention scores
        attn = self.conv3(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x_pooled))))))  # Compute attention scores

        # 3. Return the pooled attention map (before softmax)
        return attn

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

        # Pad to fit window size
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

# Full MEFA Module
class MEFA(nn.Module):
    def __init__(self):
        super(MEFA, self).__init__()

        # Initial Inception Blocks
        self.inception_rgb = InceptionBlockA(in_channels=3)
        self.inception_ir = InceptionBlockA(in_channels=1)

        # Attention Mechanisms
        self.local_attn = LocalAttention(in_channels=128)
        self.global_attn = GlobalAttention(in_channels=128)

        # Cross-Attention for RGB and IR Features
        self.cross_attn_rgb = CrossAttention(in_channels=128, window_size=16)
        self.cross_attn_ir = CrossAttention(in_channels=128, window_size=16)

        # Second Inception Block (Feature Fusion)
        self.inception_fused = InceptionBlockA(in_channels=256)

        # Final Convolution to Return 3-Channel Output
        self.final_conv = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=1)

    def forward(self, x):
        """
        x: (B, 4, H, W) - Input 4-channel image (RGB + IR)
        Returns: (B, 3, H, W) - Processed 3-channel image
        """
        if isinstance(x, dict):
            x = x["img"]  # Extract image batch from YOLO's dataset dictionary

        # Ensure batch dimension is present
        assert x.ndim == 4, f"Expected 4D input (B, C, H, W), got {x.shape} in MEFA_CA"

        # Debugging: Print input batch shape
        print(f"🔍 MEFA_CA Input Shape: {x.shape}")
        
        # Split RGB and IR channels
        rgb, ir = x[:, :3, :, :], x[:, 3:, :, :]

        # Apply Initial Inception Blocks
        f_rgb = self.inception_rgb(rgb)
        f_ir = self.inception_ir(ir)

        # 🔁 Inject Cross-Attention
        f_rgb = self.cross_attn_rgb(f_rgb, f_ir)  # RGB attends to IR
        f_ir = self.cross_attn_ir(f_ir, f_rgb)    # IR attends to RGB

        # Apply Local Attention
        la_rgb = self.local_attn(f_rgb)
        la_ir = self.local_attn(f_ir)

        # Normalize Local Attention
        attn = torch.softmax(torch.stack([la_rgb, la_ir], dim=1), dim=1)
        la_rgb, la_ir = attn.unbind(dim=1)

        # Multiply by Local Attention Weights
        f_rgb = f_rgb * la_rgb
        f_ir = f_ir * la_ir

        # Concatenate Features
        fused_features = torch.cat([f_rgb, f_ir], dim=1)

        # Apply Global Attention
        ga_rgb = self.global_attn(f_rgb)
        ga_ir = self.global_attn(f_ir)

        # Normalize Global Attention Using Sigmoid
        ga_rgb = torch.sigmoid(ga_rgb)
        ga_ir = torch.sigmoid(ga_ir)

        # Upscale Attention Maps to Match Input Size
        ga_rgb = F.interpolate(ga_rgb, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
        ga_ir = F.interpolate(ga_ir, size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)

        # Pass Through Final Inception Block
        refined_features = self.inception_fused(fused_features)

        # Apply Global Attention
        final_rgb = refined_features * ga_rgb + refined_features
        final_ir = refined_features * ga_ir + refined_features

        # Concatenate Final Features
        fused_output = torch.cat([final_rgb, final_ir], dim=1)

        # Reduce to 3-Channel Output
        final_output = self.final_conv(fused_output)

        if final_output is None:
            raise RuntimeError("❌ MEFA_CA is returning None instead of a tensor!")

        print(f"✅ MEFA_CA Output Shape: {final_output.shape}")  # Debugging

        return final_output  # Shape: (B, 3, H, W)