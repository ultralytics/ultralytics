# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved
# Modified for EfficientSAM3 and ported into Ultralytics by SimonZeng7108528.

from __future__ import annotations

import torch
import torch.nn as nn

from .mobile_clip import MobileCLIPTextTransformer


class TextStudentEncoder(nn.Module):
    """Lightweight SAM3 text encoder backed by a MobileCLIP text transformer.

    This encoder replaces the heavy CLIP ViT-L text encoder (353 M params) with a compact MobileCLIP transformer
    (42–124 M params depending on the variant).  It tokenises input strings with the standard OpenAI CLIP BPE
    vocabulary and projects the transformer output to the SAM3 model dimension.

    Three backbone variants are available:

    - **S0** (``backbone_type="S0"``): MobileCLIP-T variant (``model_name="mct"``), dim=512, 4 layers, ~42 M params.
    - **S1** (``backbone_type="S1"``): Base transformer, dim=512, 12 layers, ~64 M params.
    - **L** (``backbone_type="L"``): Base transformer, dim=768, 12 layers, ~124 M params.

    Args:
        cfg (dict): MobileCLIP configuration dictionary passed directly to :class:`MobileCLIPTextTransformer`.
            Required keys: ``dim``, ``model_name``, ``n_transformer_layers``, ``n_heads_per_layer``,
            ``ffn_multiplier_per_layer``, ``norm_layer``, ``context_length``, ``vocab_size``,
            ``causal_masking``.
        context_length (int): Operational context length for tokenisation and positional embeddings.
        output_dim (int): Output dimension (SAM3 d_model, typically 256).

    Examples:
        >>> cfg = {
        ...     "dim": 512,
        ...     "model_name": "mct",
        ...     "n_transformer_layers": 4,
        ...     "n_heads_per_layer": 8,
        ...     "ffn_multiplier_per_layer": 4.0,
        ...     "norm_layer": "layer_norm_fp32",
        ...     "context_length": 77,
        ...     "vocab_size": 49408,
        ...     "causal_masking": False,
        ... }
        >>> encoder = TextStudentEncoder(cfg, context_length=16, output_dim=256)
        >>> mask, memory, embeds = encoder(["a cat", "a dog"], device=torch.device("cpu"))
    """

    def __init__(self, cfg: dict, context_length: int, output_dim: int) -> None:
        """Initialize TextStudentEncoder with backbone encoder and linear projection head."""
        super().__init__()
        self.context_length = context_length

        # MobileCLIP text transformer (includes embedding layer and positional embedding).
        # Build at the checkpoint's original context_length (cfg["context_length"], typically 77)
        # so that _load_checkpoint() can restore weights without a size mismatch.
        # set_context_length() is called after loading to truncate to the operational length.
        self.encoder = MobileCLIPTextTransformer(cfg=cfg, projection_dim=cfg["dim"])

        # Linear projection from MobileCLIP dim to SAM3 d_model.
        self.projector = nn.Linear(cfg["dim"], output_dim)

    def set_context_length(self, context_length: int) -> None:
        """Resize positional embeddings after checkpoint loading.

        Checkpoints are typically saved with the default context length of 77 that the model was initially
        pre-trained with. Call this method after :func:`~torch.load` to truncate the positional embeddings to the
        operationally shorter context (e.g. 16 or 32) that the model was fine-tuned with.

        Args:
            context_length (int): New context length (must be ≤ the checkpoint's context length).
        """
        self.context_length = context_length
        if hasattr(self.encoder, "resize_pos_embed"):
            self.encoder.resize_pos_embed(context_length)

    def reparameterize(self) -> None:
        """Fuse all re-parameterisable RepMixer blocks for faster inference.

        Delegates to :meth:`MobileCLIPTextTransformer.reparameterize`.  Safe to call on all variants;
        non-MCT checkpoints (S1, L) have no RepMixer blocks and the call is a no-op.
        Call after weights are loaded and :meth:`set_context_length` has been applied.
        """
        self.encoder.reparameterize()

    def forward(
        self, text: list, input_boxes=None, device: torch.device | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenise text, encode, project, and return SAM3-compatible outputs.

        Args:
            text (list[str]): Input text strings to encode.
            input_boxes: Unused; retained for API compatibility with :class:`VETextEncoder`.
            device (torch.device | None): Target device for the tokenised tensor.

        Returns:
            text_attention_mask (torch.Tensor): Boolean mask of shape ``(B, seq_len)`` where ``True`` marks *padding*
                tokens to be ignored.
            text_memory (torch.Tensor): Encoded token features of shape ``(seq_len, B, output_dim)``.
            input_embeds (torch.Tensor): Raw token + positional embeddings of shape ``(seq_len, B, dim)`` (before the
                transformer).

        Raises:
            ImportError: If the CLIP package is not installed.
        """
        try:
            import clip
        except ImportError:
            from ultralytics.utils.checks import check_requirements

            check_requirements("git+https://github.com/ultralytics/CLIP.git")
            import clip

        # 1. Tokenise using the standard CLIP BPE tokeniser.
        if device is None:
            device = next(self.parameters()).device
        tokenized = clip.tokenize(text, context_length=self.context_length, truncate=True).to(device)

        # 2. Compute token + positional embeddings.
        input_embeds = self.encoder.forward_embedding(tokenized)  # (B, seq_len, dim)

        # 3. Run the MobileCLIP transformer (pass pre-computed embeddings to avoid a second embedding lookup).
        text_memory = self.encoder(input_embeds, return_all_tokens=True, input_is_embeddings=True)  # (B, seq_len, dim)

        # 4. Project to SAM3 d_model.
        text_memory = self.projector(text_memory)  # (B, seq_len, output_dim)

        # 5. Build attention mask: True = padding (token id == 0), False = valid.
        text_attention_mask = tokenized == 0

        # SAM3 VETextEncoder returns (seq_len, B, dim) — match that convention.
        return text_attention_mask, text_memory.transpose(0, 1), input_embeds.transpose(0, 1)
