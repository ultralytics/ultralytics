# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Vision Transformer (ViT-B/16) backbone for ReID, with CLIP weight loading.

Reference: TransReID (He et al. ICCV'21) and CLIP-ReID (Li et al. AAAI'23). v1 wires only
the base ViT-B/16; JPM/SIE will land in v2.

Weight init: the ImageNet/CLIP visual encoder weights from OpenAI CLIP ViT-B/16
(TorchScript .pt) can be loaded via load_clip_weights. We extract only the `visual.*`
keys and remap to our state_dict naming.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class _PatchEmbed(nn.Module):
    """16x16 patch embedding via stride-16 conv (matches CLIP `visual.conv1`)."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                       # B, D, H/p, W/p
        return x.flatten(2).transpose(1, 2)    # B, N, D


class _Attention(nn.Module):
    """Multi-head self-attention using CLIP's combined QKV linear."""

    def __init__(self, dim: int = 768, heads: int = 12):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        # CLIP stores qkv as one big linear (named attn.in_proj_*); keep that layout.
        self.in_proj_weight = nn.Parameter(torch.empty(3 * dim, dim))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * dim))
        self.out_proj = nn.Linear(dim, dim)
        nn.init.xavier_uniform_(self.in_proj_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)   # B, N, 3D
        qkv = qkv.reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                                     # B, h, N, hd
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.out_proj(x)


class _MLP(nn.Module):
    def __init__(self, dim: int = 768, mlp_ratio: float = 4.0):
        super().__init__()
        h = int(dim * mlp_ratio)
        self.c_fc = nn.Linear(dim, h)
        self.c_proj = nn.Linear(h, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class _Block(nn.Module):
    def __init__(self, dim: int = 768, heads: int = 12):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, heads)
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class ViTBackbone(nn.Module):
    """ViT-B/16 image encoder (matches CLIP's visual tower).

    Returns a 4D tensor [B, D, 1, 1] so it slots into Ultralytics' YOLO-style head pipeline
    (ReID head pools to 1x1 anyway). The CLS token carries the full-image feature.
    """

    def __init__(self, img_size: int = 224, patch_size: int = 16, embed_dim: int = 768, depth: int = 12, heads: int = 12, ch: int = 3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.patch_embed = _PatchEmbed(patch_size, ch, embed_dim)
        n_patches = (img_size // patch_size) ** 2
        self.class_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.positional_embedding = nn.Parameter(torch.zeros(n_patches + 1, embed_dim))
        self.ln_pre = nn.LayerNorm(embed_dim)
        self.transformer = nn.ModuleDict({
            "resblocks": nn.ModuleList([_Block(embed_dim, heads) for _ in range(depth)]),
        })
        self.ln_post = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.class_embedding, std=0.02)
        nn.init.trunc_normal_(self.positional_embedding, std=0.02)
        # CLIP visual encoder expects its own normalization stats (not ImageNet).
        # The ReID dataset passes [0,1] tensors with mean=(0,0,0) std=(1,1,1) by default —
        # so we normalize inside the backbone, making this transparent to the data pipeline.
        self.register_buffer("clip_mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("clip_std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Apply CLIP normalization in-band (input is [0,1] from ReidDataset).
        x = (x - self.clip_mean) / self.clip_std
        x = self.patch_embed(x)                                                   # B, N, D
        cls = self.class_embedding.to(x.dtype).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)                                            # B, N+1, D
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        for blk in self.transformer["resblocks"]:
            x = blk(x)
        cls_out = self.ln_post(x[:, 0])                                            # B, D
        return cls_out.unsqueeze(-1).unsqueeze(-1)                                 # B, D, 1, 1


def _resize_pos_embed(pos: torch.Tensor, new_grid: Tuple[int, int]) -> torch.Tensor:
    """Bicubic-resize a CLS-prepended position embedding from (N+1, D) to a new grid."""
    cls, patches = pos[:1], pos[1:]
    n = patches.shape[0]
    old = int(math.sqrt(n))
    assert old * old == n, "expected square old grid"
    D = patches.shape[1]
    p2d = patches.reshape(1, old, old, D).permute(0, 3, 1, 2)        # 1, D, old, old
    p2d = F.interpolate(p2d, size=new_grid, mode="bicubic", align_corners=False)
    patches = p2d.permute(0, 2, 3, 1).reshape(new_grid[0] * new_grid[1], D)
    return torch.cat([cls, patches], dim=0)


def load_clip_visual_into(model: ViTBackbone, clip_path: str, strict: bool = False) -> dict:
    """Convenience wrapper: load CLIP ViT-B/16 from a TorchScript file path."""
    sd = torch.jit.load(clip_path, map_location="cpu").state_dict()
    visual = {k[len("visual."):]: v for k, v in sd.items() if k.startswith("visual.")}
    return load_clip_visual_into_sd(model, visual, strict=strict)


def load_clip_visual_into_sd(model: ViTBackbone, visual: dict, strict: bool = False) -> dict:
    """Load a CLIP visual-tower state_dict (keys WITHOUT the `visual.` prefix) into ViTBackbone.
    Pos embedding is bicubic-resized when the model's patch grid differs from CLIP's 14x14.
    """
    new = {}
    info = {"loaded": 0, "skipped": []}
    new["class_embedding"] = visual["class_embedding"]
    new["positional_embedding"] = visual["positional_embedding"]
    # Resize positional_embedding if grid sizes differ.
    grid_now = model.img_size // model.patch_size
    grid_clip = int(math.sqrt(visual["positional_embedding"].shape[0] - 1))
    if grid_now != grid_clip:
        new["positional_embedding"] = _resize_pos_embed(visual["positional_embedding"], (grid_now, grid_now))
    new["patch_embed.proj.weight"] = visual["conv1.weight"]
    new["ln_pre.weight"] = visual["ln_pre.weight"]
    new["ln_pre.bias"] = visual["ln_pre.bias"]
    new["ln_post.weight"] = visual["ln_post.weight"]
    new["ln_post.bias"] = visual["ln_post.bias"]
    # Transformer blocks: 12 of them
    for i in range(12):
        src = f"transformer.resblocks.{i}"
        dst = f"transformer.resblocks.{i}"
        new[f"{dst}.ln_1.weight"] = visual[f"{src}.ln_1.weight"]
        new[f"{dst}.ln_1.bias"] = visual[f"{src}.ln_1.bias"]
        new[f"{dst}.ln_2.weight"] = visual[f"{src}.ln_2.weight"]
        new[f"{dst}.ln_2.bias"] = visual[f"{src}.ln_2.bias"]
        new[f"{dst}.attn.in_proj_weight"] = visual[f"{src}.attn.in_proj_weight"]
        new[f"{dst}.attn.in_proj_bias"] = visual[f"{src}.attn.in_proj_bias"]
        new[f"{dst}.attn.out_proj.weight"] = visual[f"{src}.attn.out_proj.weight"]
        new[f"{dst}.attn.out_proj.bias"] = visual[f"{src}.attn.out_proj.bias"]
        new[f"{dst}.mlp.c_fc.weight"] = visual[f"{src}.mlp.c_fc.weight"]
        new[f"{dst}.mlp.c_fc.bias"] = visual[f"{src}.mlp.c_fc.bias"]
        new[f"{dst}.mlp.c_proj.weight"] = visual[f"{src}.mlp.c_proj.weight"]
        new[f"{dst}.mlp.c_proj.bias"] = visual[f"{src}.mlp.c_proj.bias"]
    missing, unexpected = model.load_state_dict(new, strict=strict)
    info["loaded"] = len(new)
    info["missing"] = list(missing)
    info["unexpected"] = list(unexpected)
    return info
