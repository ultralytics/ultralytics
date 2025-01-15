# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import copy
from typing import Optional

import torch
from torch import Tensor, nn

from .blocks import RoPEAttention


class MemoryAttentionLayer(nn.Module):
    """
    Implements a memory attention layer with self-attention and cross-attention mechanisms for neural networks.

    This class combines self-attention, cross-attention, and feedforward components to process input tensors and
    generate memory-based attention outputs.

    Attributes:
        d_model (int): Dimensionality of the model.
        dim_feedforward (int): Dimensionality of the feedforward network.
        dropout_value (float): Dropout rate for regularization.
        self_attn (RoPEAttention): Self-attention mechanism using RoPE (Rotary Position Embedding).
        cross_attn_image (RoPEAttention): Cross-attention mechanism for image processing.
        linear1 (nn.Linear): First linear layer of the feedforward network.
        linear2 (nn.Linear): Second linear layer of the feedforward network.
        norm1 (nn.LayerNorm): Layer normalization for self-attention output.
        norm2 (nn.LayerNorm): Layer normalization for cross-attention output.
        norm3 (nn.LayerNorm): Layer normalization for feedforward network output.
        dropout1 (nn.Dropout): Dropout layer after self-attention.
        dropout2 (nn.Dropout): Dropout layer after cross-attention.
        dropout3 (nn.Dropout): Dropout layer after feedforward network.
        activation (nn.ReLU): Activation function for the feedforward network.
        pos_enc_at_attn (bool): Flag to add positional encoding at attention.
        pos_enc_at_cross_attn_queries (bool): Flag to add positional encoding to cross-attention queries.
        pos_enc_at_cross_attn_keys (bool): Flag to add positional encoding to cross-attention keys.

    Methods:
        forward: Performs the full memory attention operation on input tensors.
        _forward_sa: Performs self-attention on input tensor.
        _forward_ca: Performs cross-attention between target and memory tensors.

    Examples:
        >>> layer = MemoryAttentionLayer(d_model=256, dim_feedforward=2048, dropout=0.1)
        >>> tgt = torch.randn(1, 100, 256)
        >>> memory = torch.randn(1, 100, 64)
        >>> pos = torch.randn(1, 100, 256)
        >>> query_pos = torch.randn(1, 100, 256)
        >>> output = layer(tgt, memory, pos, query_pos)
        >>> print(output.shape)
        torch.Size([1, 100, 256])
    """

    def __init__(
        self,
        d_model: int = 256,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        pos_enc_at_attn: bool = False,
        pos_enc_at_cross_attn_keys: bool = True,
        pos_enc_at_cross_attn_queries: bool = False,
    ):
        """Initializes a memory attention layer with self-attention, cross-attention, and feedforward components."""
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.dropout_value = dropout
        self.self_attn = RoPEAttention(embedding_dim=256, num_heads=1, downsample_rate=1)
        self.cross_attn_image = RoPEAttention(
            rope_k_repeat=True,
            embedding_dim=256,
            num_heads=1,
            downsample_rate=1,
            kv_in_dim=64,
        )

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU()

        # Where to add pos enc
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt, query_pos):
        """Performs self-attention on input tensor using positional encoding and RoPE attention mechanism."""
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        tgt2 = self.self_attn(q, k, v=tgt2)
        tgt = tgt + self.dropout1(tgt2)
        return tgt

    def _forward_ca(self, tgt, memory, query_pos, pos, num_k_exclude_rope=0):
        """Performs cross-attention between target and memory tensors using RoPEAttention mechanism."""
        kwds = {}
        if num_k_exclude_rope > 0:
            assert isinstance(self.cross_attn_image, RoPEAttention)
            kwds = {"num_k_exclude_rope": num_k_exclude_rope}

        # Cross-Attention
        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn_image(
            q=tgt2 + query_pos if self.pos_enc_at_cross_attn_queries else tgt2,
            k=memory + pos if self.pos_enc_at_cross_attn_keys else memory,
            v=memory,
            **kwds,
        )
        tgt = tgt + self.dropout2(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """Processes input tensors using self-attention, cross-attention, and MLP for memory-based attention."""
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(tgt, memory, query_pos, pos, num_k_exclude_rope)
        # MLP
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class MemoryAttention(nn.Module):
    """
    Memory attention module for processing sequential data with self and cross-attention mechanisms.

    This class implements a multi-layer attention mechanism that combines self-attention and cross-attention
    for processing sequential data, particularly useful in transformer-like architectures.

    Attributes:
        d_model (int): The dimension of the model's hidden state.
        layers (nn.ModuleList): A list of MemoryAttentionLayer modules.
        num_layers (int): The number of attention layers.
        norm (nn.LayerNorm): Layer normalization applied to the output.
        pos_enc_at_input (bool): Whether to apply positional encoding at the input.
        batch_first (bool): Whether the input tensors are in batch-first format.

    Methods:
        forward: Processes input tensors through the attention layers.

    Examples:
        >>> d_model = 256
        >>> layer = MemoryAttentionLayer(d_model)
        >>> attention = MemoryAttention(d_model, pos_enc_at_input=True, layer=layer, num_layers=3)
        >>> curr = torch.randn(10, 32, d_model)  # (seq_len, batch_size, d_model)
        >>> memory = torch.randn(20, 32, d_model)  # (mem_len, batch_size, d_model)
        >>> curr_pos = torch.randn(10, 32, d_model)
        >>> memory_pos = torch.randn(20, 32, d_model)
        >>> output = attention(curr, memory, curr_pos, memory_pos)
        >>> print(output.shape)
        torch.Size([10, 32, 256])
    """

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = True,  # Do layers expect batch first input?
    ):
        """Initializes MemoryAttention module with layers and normalization for attention processing."""
        super().__init__()
        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.batch_first = batch_first

    def forward(
        self,
        curr: torch.Tensor,  # self-attention inputs
        memory: torch.Tensor,  # cross-attention inputs
        curr_pos: Optional[Tensor] = None,  # pos_enc for self-attention inputs
        memory_pos: Optional[Tensor] = None,  # pos_enc for cross-attention inputs
        num_obj_ptr_tokens: int = 0,  # number of object pointer *tokens*
    ):
        """Processes input tensors through multiple attention layers, applying self and cross-attention mechanisms."""
        if isinstance(curr, list):
            assert isinstance(curr_pos, list)
            assert len(curr) == len(curr_pos) == 1
            curr, curr_pos = (
                curr[0],
                curr_pos[0],
            )

        assert curr.shape[1] == memory.shape[1], "Batch size must be the same for curr and memory"

        output = curr
        if self.pos_enc_at_input and curr_pos is not None:
            output = output + 0.1 * curr_pos

        if self.batch_first:
            # Convert to batch first
            output = output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)

        for layer in self.layers:
            kwds = {}
            if isinstance(layer.cross_attn_image, RoPEAttention):
                kwds = {"num_k_exclude_rope": num_obj_ptr_tokens}

            output = layer(
                tgt=output,
                memory=memory,
                pos=memory_pos,
                query_pos=curr_pos,
                **kwds,
            )
        normed_output = self.norm(output)

        if self.batch_first:
            # Convert back to seq first
            normed_output = normed_output.transpose(0, 1)
            curr_pos = curr_pos.transpose(0, 1)

        return normed_output
