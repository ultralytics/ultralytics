######################################## IEEE Access terrasegnet by AI Little monster  start ########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
 
     
class SSAM(nn.Module):
    '''Spectral Spatial Attention Module (SSAM)'''
    def __init__(self, in_channels, groups=4, reduction_ratio=2, dropout=0.1):
        super(SSAM, self).__init__()
        assert in_channels % groups == 0, "in_channels harus bisa dibagi habis oleh groups"
        self.groups = groups
        self.group_channels = in_channels // groups
 
        # Spectral Group Attention
        self.spectral_convs = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(self.group_channels, max(4, self.group_channels // reduction_ratio), 1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(max(4, self.group_channels // reduction_ratio), self.group_channels, 1),
                nn.Sigmoid()
            ) for _ in range(groups)
        ])
 
        # Spatial Attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Sigmoid()
        )
 
        # Global Semantic Feedback
        self.semantic_feedback = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid()
        )
 
        # Final Fusion
        self.final_fusion = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Dropout2d(dropout)
        )
 
    def forward(self, x):
        identity = x
 
        # Spectral Group Attention
        chunks = torch.chunk(x, self.groups, dim=1)
        spec_out = []
        for i in range(self.groups):
            attn = self.spectral_convs[i](chunks[i])
            spec_out.append(chunks[i] * attn)
        spectral = torch.cat(spec_out, dim=1)
 
        # Spatial & Semantic Attention
        spatial = self.spatial_attn(x)
        semantic = self.semantic_feedback(x)
 
        # Fusion
        combined = spectral * (spatial * semantic)
        out = self.final_fusion(combined)
 
        return identity + out
 
class CAAM(nn.Module):
    '''Convolutional Axial Attention Module (CAAM)'''
    def __init__(self, dim, squeeze_factor=4, dropout=0.1):
        super(CAAM, self).__init__()
        self.squeeze_factor = squeeze_factor
        squeezed_dim = max(1, dim // squeeze_factor)
        
        # Squeeze projections
        self.squeeze_proj = nn.Sequential(
            nn.Conv2d(dim, squeezed_dim, 1, bias=False),
            nn.BatchNorm2d(squeezed_dim)
        )
        self.unsqueeze_proj = nn.Sequential(
            nn.Conv2d(squeezed_dim, dim, 1, bias=False), # Output dari attn (setelah h_conv/w_conv) adalah squeezed_dim
            nn.BatchNorm2d(dim)
        )
        
        # Axial attention components - Average Pooling
        self.h_avg_pool = nn.AdaptiveAvgPool2d((None, 1))
        self.w_avg_pool = nn.AdaptiveAvgPool2d((1, None))
        
        # Axial attention components - Max Pooling
        self.h_max_pool = nn.AdaptiveMaxPool2d((None, 1))
        self.w_max_pool = nn.AdaptiveMaxPool2d((1, None))
        
        self.h_conv = nn.Conv2d(2 * squeezed_dim, squeezed_dim, (1, 3), padding=(0, 1), padding_mode='replicate', bias=False) # groups default adalah 1
        self.w_conv = nn.Conv2d(2 * squeezed_dim, squeezed_dim, (3, 1), padding=(1, 0), padding_mode='replicate', bias=False) # groups default adalah 1
 
        # Context-aware global descriptor
        self.global_context = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, 1, bias=False),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
 
    def forward(self, x):
        H, W = x.shape[2:]
        identity = x
        
        # Squeeze channel dimension
        x_squeezed = self.squeeze_proj(x) # [B, squeezed_dim, H, W]
        
        # Height-wise attention
        h_avg_features = self.h_avg_pool(x_squeezed) # [B, squeezed_dim, H, 1]
        h_max_features = self.h_max_pool(x_squeezed) # [B, squeezed_dim, H, 1]
        # Concat at channel dim (dim=1)
        h_pooled_features = torch.cat((h_avg_features, h_max_features), dim=1) # [B, 2 * squeezed_dim, H, 1]
        
        h_attn = self.h_conv(h_pooled_features) # Input: 2*squeezed_dim, Output: squeezed_dim. [B, squeezed_dim, H, 1]
        h_attn = h_attn.expand(-1, -1, H, W)    # [B, squeezed_dim, H, W]
        
        # Width-wise attention
        w_avg_features = self.w_avg_pool(x_squeezed) # [B, squeezed_dim, 1, W]
        w_max_features = self.w_max_pool(x_squeezed) # [B, squeezed_dim, 1, W]
        # Concat at channel dim
        w_pooled_features = torch.cat((w_avg_features, w_max_features), dim=1) # [B, 2 * squeezed_dim, 1, W]
 
        w_attn = self.w_conv(w_pooled_features) # Input: 2*squeezed_dim, Output: squeezed_dim. [B, squeezed_dim, 1, W]
        w_attn = w_attn.expand(-1, -1, H, W)    # [B, squeezed_dim, H, W]
        
        # Combine axial attentions
        attn = h_attn + w_attn
        attn = self.dropout(attn)
        
        # Restore channel dimension
        out = self.unsqueeze_proj(attn)
 
        # Context-aware modulation
        context = self.global_context(identity)
        out = out * context
 
        return identity + out
 
 
######################################## IEEE Access terrasegnet by AI Little monster  end ########################################