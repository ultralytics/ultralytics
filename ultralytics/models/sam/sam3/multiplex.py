# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

"""SAM 3.1 Object Multiplex bucket management.

Object Multiplex groups tracked objects into fixed-capacity buckets so the memory encoder, memory
attention, and mask decoder run once per bucket (batch = num_buckets) instead of once per object.
``MultiplexState`` maps between the data space (num_objects, C, ...) and the multiplex space
(num_buckets, multiplex_count, C, ...) via precomputed partial-permutation matmuls.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

# Special slot values
_PADDING_NUM = -1  # empty slot in a bucket
_REMOVED_NUM = -1116  # slot whose object was removed (kept distinct from empty)


class MultiplexState:
    """Bucket-assignment state for Object Multiplex tracking.

    Records which object index occupies which slot of which bucket, and converts tensors between
    the data space and the multiplex space.

    Attributes:
        assignments (list[list[int]]): Per-bucket lists of object indices; ``_PADDING_NUM`` marks
            empty slots and ``_REMOVED_NUM`` marks removed objects.
        num_buckets (int): Number of buckets.
        multiplex_count (int): Number of slots per bucket.
        total_valid_entries (int): Number of live objects (non-negative slots).
        object_ids (list[int] | None): Optional external ids mapped 1:1 to valid entries.
        mux_matrix (torch.Tensor): (num_buckets * multiplex_count, total_valid_entries) matrix.
        demux_matrix (torch.Tensor): (total_valid_entries, num_buckets * multiplex_count) matrix.
    """

    def __init__(
        self,
        assignments: list[list[int]],
        device: torch.device,
        dtype: torch.dtype,
        allowed_bucket_capacity: int,
        object_ids: list[int] | None = None,
    ):
        """Initialize the state from per-bucket object index assignments.

        Args:
            assignments (list[list[int]]): One list per bucket with object indices in
                [0, num_objects) or the special values ``_PADDING_NUM`` / ``_REMOVED_NUM``.
            device (torch.device): Device for the mux/demux matrices.
            dtype (torch.dtype): Dtype for the mux/demux matrices.
            allowed_bucket_capacity (int): Maximum non-padding entries per bucket.
            object_ids (list[int] | None): Optional external object ids for bookkeeping.
        """
        self.device = device
        self.dtype = dtype
        self.allowed_bucket_capacity = allowed_bucket_capacity
        self._initialize_assignments(assignments, object_ids=object_ids)

    def _initialize_assignments(self, assignments: list[list[int]], object_ids: list[int] | None = None):
        """Validate assignments and rebuild the derived counters and transition matrices."""
        self.assignments = assignments
        self.num_buckets = len(assignments)
        if self.num_buckets == 0:
            raise ValueError("No buckets found in the state")

        self.multiplex_count = len(assignments[0])
        assert all(len(bucket) == self.multiplex_count for bucket in assignments)

        self.total_valid_entries = sum(sum(1 for x in bucket if x >= 0) for bucket in assignments)
        self.total_non_padding_entries = sum(sum(1 for x in bucket if x != _PADDING_NUM) for bucket in assignments)

        self.object_ids = object_ids
        if self.object_ids is not None:
            assert len(self.object_ids) == self.total_valid_entries, "object_ids must map 1:1 to the valid entries"

        seen = set()
        for bucket in assignments:
            non_padding = sum(1 for x in bucket if x != _PADDING_NUM)
            assert non_padding <= self.allowed_bucket_capacity, f"{non_padding} > {self.allowed_bucket_capacity}"
            for obj_idx in bucket:
                if obj_idx >= 0:
                    assert obj_idx < self.total_non_padding_entries and obj_idx not in seen
                    seen.add(obj_idx)

        self._precompute_transition_matrices(self.device, self.dtype)

    @property
    def available_slots(self) -> int:
        """Number of free slots across all buckets."""
        return self.num_buckets * self.allowed_bucket_capacity - self.total_non_padding_entries

    def find_next_batch_of_available_indices(self, num_objects: int, allow_new_buckets: bool = False) -> list[int]:
        """Return the consecutive object indices the next ``num_objects`` additions will receive."""
        assert num_objects > 0
        if not allow_new_buckets:
            assert self.available_slots >= num_objects, f"not enough slots {self.available_slots} < {num_objects}"
        return list(range(self.total_valid_entries, self.total_valid_entries + num_objects))

    def add_objects(
        self,
        object_indices: list[int],
        object_ids: list[int] | None = None,
        allow_new_buckets: bool = False,
        prefer_new_buckets: bool = False,
    ):
        """Add new objects, filling empty slots first and creating new buckets when allowed.

        Args:
            object_indices (list[int]): Sorted indices continuing the existing sequence.
            object_ids (list[int] | None): External ids matching ``object_indices``.
            allow_new_buckets (bool): Permit creating new buckets when existing slots run out.
            prefer_new_buckets (bool): Place all new objects in fresh buckets (requires
                ``allow_new_buckets``).
        """
        if len(object_indices) == 0:
            return
        object_indices = object_indices.copy()
        assert (object_ids is None) == (self.object_ids is None), "object_ids must be consistently given or omitted"
        if object_ids is not None:
            assert len(object_ids) == len(object_indices)
            object_ids = object_ids.copy()

        num_new_objects = len(object_indices)
        assert object_indices == sorted(object_indices), "object_indices must be sorted"
        object_indices.reverse()  # pop from the end
        if object_ids is not None:
            object_ids.reverse()
        if prefer_new_buckets:
            assert allow_new_buckets, "prefer_new_buckets requires allow_new_buckets"

        def _pop_next():
            idx = object_indices.pop()
            if object_ids is not None and self.object_ids is not None:
                self.object_ids.append(object_ids.pop())
            return idx

        if not prefer_new_buckets:  # fill empty slots in existing buckets first
            for bucket in self.assignments:
                for i in range(self.allowed_bucket_capacity):
                    if bucket[i] == _PADDING_NUM:
                        bucket[i] = _pop_next()
                        if len(object_indices) == 0:
                            break
                if len(object_indices) == 0:
                    break

        if len(object_indices) > 0 and not allow_new_buckets:
            raise ValueError(f"Cannot place objects {list(reversed(object_indices))} without creating new buckets")

        while len(object_indices) > 0:  # create new buckets for the remainder
            new_bucket = [_PADDING_NUM] * self.multiplex_count
            for i in range(self.allowed_bucket_capacity):
                if len(object_indices) == 0:
                    break
                new_bucket[i] = _pop_next()
            self.assignments.append(new_bucket)

        original = self.total_valid_entries
        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        assert self.total_valid_entries == original + num_new_objects

    def remove_objects(self, object_indices: list[int], strict: bool = True) -> list[int]:
        """Remove objects, drop fully-empty buckets, and remap remaining indices to be sequential.

        Args:
            object_indices (list[int]): Object indices to remove.
            strict (bool): Raise if an index is not found.

        Returns:
            (list[int]): Indices of the buckets that were kept, in original order.
        """
        object_indices = object_indices.copy()
        for bucket in self.assignments:
            for slot_idx, obj_id in enumerate(bucket):
                if obj_id in object_indices:
                    bucket[slot_idx] = _REMOVED_NUM
                    object_indices.remove(obj_id)
        if strict:
            assert len(object_indices) == 0, f"Failed to remove objects: {object_indices}"

        buckets_to_keep = [
            i
            for i, bucket in enumerate(self.assignments)
            if not all(obj_id in (_PADDING_NUM, _REMOVED_NUM) for obj_id in bucket)
        ]
        self.assignments = [self.assignments[i] for i in buckets_to_keep]

        if len(buckets_to_keep) == 0:  # state invalidated
            self.assignments = None
            if self.object_ids is not None:
                self.object_ids = []
            return buckets_to_keep

        # Remap surviving object indices to a dense sequential range
        sorted_ids = sorted({obj_id for bucket in self.assignments for obj_id in bucket if obj_id >= 0})
        id_mapping = {old: new for new, old in enumerate(sorted_ids)}
        for bucket in self.assignments:
            for i, obj_id in enumerate(bucket):
                if obj_id >= 0:
                    bucket[i] = id_mapping[obj_id]
        if self.object_ids is not None:
            new_object_ids = [None] * len(sorted_ids)
            for old_idx, new_idx in id_mapping.items():
                new_object_ids[new_idx] = self.object_ids[old_idx]
            assert not any(obj_id is None for obj_id in new_object_ids)
            self.object_ids = new_object_ids

        self._initialize_assignments(self.assignments, object_ids=self.object_ids)
        return buckets_to_keep

    def _precompute_transition_matrices(self, device: torch.device, dtype: torch.dtype):
        """Build the partial-permutation matrices used by mux/demux."""
        self.mux_matrix = torch.zeros(
            self.num_buckets * self.multiplex_count, self.total_valid_entries, device=device, dtype=dtype
        )
        self.demux_matrix = torch.zeros(
            self.total_valid_entries, self.num_buckets * self.multiplex_count, device=device, dtype=dtype
        )
        for i in range(self.num_buckets):
            for j in range(self.multiplex_count):
                object_idx = self.assignments[i][j]
                if object_idx >= 0:
                    self.mux_matrix[i * self.multiplex_count + j, object_idx] = 1.0
                    self.demux_matrix[object_idx, i * self.multiplex_count + j] = 1.0

    def mux(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (total_valid_entries, ...) to (num_buckets, multiplex_count, ...); padding slots are zeros."""
        assert x.shape[0] == self.total_valid_entries, f"{x.shape[0]} != {self.total_valid_entries}"
        result = self.mux_matrix @ x.reshape(x.shape[0], -1)
        return result.view((self.num_buckets, self.multiplex_count) + x.shape[1:])

    def demux(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (num_buckets, multiplex_count, ...) back to (total_valid_entries, ...)."""
        assert x.shape[:2] == (self.num_buckets, self.multiplex_count)
        result = self.demux_matrix @ x.reshape(self.num_buckets * self.multiplex_count, -1)
        return result.view((self.total_valid_entries,) + x.shape[2:])

    def get_valid_object_mask(self) -> torch.Tensor:
        """Return a (num_buckets, multiplex_count) bool mask of occupied slots."""
        return (self.mux_matrix.sum(dim=1) > 0).reshape(self.num_buckets, self.multiplex_count)

    def get_all_valid_object_idx(self) -> set[int]:
        """Return the set of live internal object indices."""
        return {obj_idx for bucket in self.assignments for obj_idx in bucket if obj_idx >= 0}


class MultiplexController(nn.Module):
    """Create MultiplexStates with the configured bucket capacity (no learnable weights)."""

    def __init__(self, multiplex_count: int, eval_multiplex_count: int = -1):
        """Initialize with bucket size ``multiplex_count`` and an optional eval-time capacity."""
        super().__init__()
        self.multiplex_count = multiplex_count
        self.eval_multiplex_count = eval_multiplex_count if eval_multiplex_count >= 0 else multiplex_count
        assert self.multiplex_count >= 1

    @property
    def allowed_bucket_capacity(self) -> int:
        """Bucket capacity for the current mode."""
        return self.multiplex_count if self.training else self.eval_multiplex_count

    def get_state(
        self,
        num_valid_entries: int,
        device: torch.device,
        dtype: torch.dtype,
        random: bool = True,
        object_ids: list[int] | None = None,
    ) -> MultiplexState:
        """Assign ``num_valid_entries`` objects to buckets and return the resulting state."""
        capacity = self.allowed_bucket_capacity
        num_buckets = math.ceil(num_valid_entries / capacity)
        ids = torch.randperm(num_valid_entries, dtype=torch.int64) if random else torch.arange(num_valid_entries)
        total = num_buckets * capacity
        if ids.shape[0] < total:
            ids = torch.cat([ids, torch.tensor([_PADDING_NUM] * (total - ids.shape[0]))])
        assignments = [
            ids[i * capacity : (i + 1) * capacity].tolist() + [_PADDING_NUM] * (self.multiplex_count - capacity)
            for i in range(num_buckets)
        ]
        return MultiplexState(assignments, device, dtype, allowed_bucket_capacity=capacity, object_ids=object_ids)


def functional_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    num_heads: int,
    num_k_exclude_rope: int = 0,
    freqs_cis: torch.Tensor | None = None,
    rope_k_repeat: bool = False,
) -> torch.Tensor:
    """Multi-head attention with rotary encoding applied inside (projections live in the caller).

    Args:
        q (torch.Tensor): Queries (B, N, C).
        k (torch.Tensor): Keys (B, M, C); batch may be 1 for broadcast.
        v (torch.Tensor): Values (B, M, C).
        num_heads (int): Number of attention heads.
        num_k_exclude_rope (int): Trailing key tokens excluded from rotary encoding (object pointers).
        freqs_cis (torch.Tensor | None): Complex rotary table matching the query grid.
        rope_k_repeat (bool): Repeat the table along keys to cover multiple memory frames.

    Returns:
        (torch.Tensor): Attention output (B, N, C).
    """
    from ..modules.utils import apply_rotary_enc

    b, n, cq = q.shape
    _, m, ck = k.shape
    _, _, cv = v.shape
    q = q.reshape(b, n, num_heads, cq // num_heads).transpose(1, 2)
    k = k.reshape(k.shape[0], m, num_heads, ck // num_heads).transpose(1, 2)
    v = v.reshape(v.shape[0], m, num_heads, cv // num_heads).transpose(1, 2)

    if freqs_cis is not None:
        num_k_rope = k.size(-2) - num_k_exclude_rope
        q, k[:, :, :num_k_rope] = apply_rotary_enc(
            q, k[:, :, :num_k_rope], freqs_cis, repeat_freqs_k=rope_k_repeat
        )

    out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(1, 2).reshape(b, n, cv)


class SimpleRoPEAttention(nn.Module):
    """Rotary-encoded attention without q/k/v/out projections (the decoupled layer owns them)."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout_p: float = 0.0,
        rope_theta: float = 10000.0,
        rope_k_repeat: bool = False,
        feat_sizes: tuple[int, int] = (64, 64),
    ):
        """Initialize with a precomputed axial rotary table for the given feature grid."""
        super().__init__()
        from functools import partial

        from ..modules.utils import compute_axial_cis

        self.num_heads = num_heads
        self.dropout_p = dropout_p  # inference-only port; dropout is inactive
        self.compute_cis = partial(compute_axial_cis, dim=d_model // num_heads, theta=rope_theta)
        self.freqs_cis = self.compute_cis(end_x=feat_sizes[0], end_y=feat_sizes[1])
        self.rope_k_repeat = rope_k_repeat

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0) -> torch.Tensor:
        """Apply rotary encoding and attention; recompute the table if the grid changed."""
        w = h = int(math.sqrt(q.shape[-2]))
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=w, end_y=h).to(q.device)
        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat
        return functional_attention(
            q,
            k,
            v,
            num_heads=self.num_heads,
            num_k_exclude_rope=num_k_exclude_rope,
            freqs_cis=self.freqs_cis,
            rope_k_repeat=self.rope_k_repeat,
        )


class DecoupledTransformerDecoderLayerv2(nn.Module):
    """Memory-attention layer with decoupled image/object streams (SAM 3.1 Object Multiplex).

    Cross-attention queries and keys are sums of separate projections of the image features and the
    object (mask memory) features, which lets one bucket of objects share a single image stream.
    """

    def __init__(
        self,
        activation: str,
        d_model: int,
        num_heads: int,
        dim_feedforward: int,
        dropout: float,
        pos_enc_at_attn: bool,
        pos_enc_at_cross_attn_keys: bool,
        pos_enc_at_cross_attn_queries: bool,
        pre_norm: bool,
        self_attention_rope: SimpleRoPEAttention,
        cross_attention_rope: SimpleRoPEAttention,
    ):
        """Initialize projections, feedforward, norms, and the rope attention submodules."""
        super().__init__()
        assert pre_norm, "only the pre-norm variant is used by SAM 3.1"
        self.d_model = d_model
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward

        self.self_attn_q_proj = nn.Linear(d_model, d_model)
        self.self_attn_k_proj = nn.Linear(d_model, d_model)
        self.self_attn_v_proj = nn.Linear(d_model, d_model)
        self.self_attn_out_proj = nn.Linear(d_model, d_model)

        self.cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.cross_attn_k_proj = nn.Linear(d_model, d_model)
        self.cross_attn_v_proj = nn.Linear(d_model, d_model)
        self.cross_attn_out_proj = nn.Linear(d_model, d_model)

        self.image_cross_attn_q_proj = nn.Linear(d_model, d_model)
        self.image_cross_attn_k_proj = nn.Linear(d_model, d_model)

        self.self_attention_rope = self_attention_rope
        self.cross_attention_rope = cross_attention_rope

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        assert activation == "gelu", f"unsupported activation {activation}"
        self.activation = F.gelu
        self.pos_enc_at_attn = pos_enc_at_attn
        self.pos_enc_at_cross_attn_queries = pos_enc_at_cross_attn_queries
        self.pos_enc_at_cross_attn_keys = pos_enc_at_cross_attn_keys

    def _forward_sa(self, tgt: torch.Tensor, query_pos: torch.Tensor | None) -> torch.Tensor:
        """Self-attention over the object stream."""
        tgt2 = self.norm1(tgt)
        q = k = tgt2 + query_pos if self.pos_enc_at_attn else tgt2
        out = self.self_attention_rope(self.self_attn_q_proj(q), self.self_attn_k_proj(k), self.self_attn_v_proj(tgt2))
        return tgt + self.dropout1(self.self_attn_out_proj(out))

    def _forward_ca(
        self,
        image: torch.Tensor,
        tgt: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        query_pos: torch.Tensor | None,
        memory_image_pos: torch.Tensor | None,
        num_k_exclude_rope: int = 0,
    ) -> torch.Tensor:
        """Cross-attention: decoupled image + object projections on both queries and keys."""
        tgt2 = self.norm2(tgt)
        q = self.image_cross_attn_q_proj(image) + self.cross_attn_q_proj(tgt2)
        if self.pos_enc_at_cross_attn_queries:
            q = q + query_pos
        k = self.image_cross_attn_k_proj(memory_image) + self.cross_attn_k_proj(memory)
        if self.pos_enc_at_cross_attn_keys:
            k = k + memory_image_pos
        v = self.cross_attn_v_proj(memory)
        out = self.cross_attention_rope(q, k, v, num_k_exclude_rope=num_k_exclude_rope)
        return tgt + self.dropout2(self.cross_attn_out_proj(out))

    def forward(
        self,
        image: torch.Tensor,
        tgt: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        image_pos: torch.Tensor | None = None,
        query_pos: torch.Tensor | None = None,
        memory_image_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        num_k_exclude_rope: int = 0,
    ):
        """Run self-attention, decoupled cross-attention, and the feedforward block."""
        tgt = self._forward_sa(tgt, query_pos)
        tgt = self._forward_ca(image, tgt, memory_image, memory, query_pos, memory_image_pos, num_k_exclude_rope)
        tgt2 = self.norm3(tgt)
        tgt = tgt + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(tgt2)))))
        return image, tgt


class TransformerEncoderDecoupledCrossAttention(nn.Module):
    """Stack of decoupled memory-attention layers fusing bucket features with the memory bank."""

    def __init__(
        self,
        d_model: int,
        pos_enc_at_input: bool,
        layer: nn.Module,
        num_layers: int,
        batch_first: bool = False,
        use_image_in_output: bool = True,
    ):
        """Initialize cloned layers and the output norm."""
        super().__init__()
        import copy

        self.d_model = d_model
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(d_model)
        self.pos_enc_at_input = pos_enc_at_input
        self.use_image_in_output = use_image_in_output
        self.batch_first = batch_first

    def forward(
        self,
        image: torch.Tensor,
        src: torch.Tensor,
        memory_image: torch.Tensor,
        memory: torch.Tensor,
        image_pos: torch.Tensor | None = None,
        src_pos: torch.Tensor | None = None,
        memory_image_pos: torch.Tensor | None = None,
        memory_pos: torch.Tensor | None = None,
        num_obj_ptr_tokens: int = 0,
    ) -> dict:
        """Fuse current bucket features with spatial memory + image memory + object pointers.

        All tensors arrive sequence-first (L, B, C) and are transposed when batch_first.
        memory_image is zero-padded over the trailing object-pointer tokens (which have no
        image stream), with their temporal encodings appended to memory_image_pos.
        """
        assert src.shape[1] == memory.shape[1], "Batch size must be the same for src and memory"

        output = src
        if self.pos_enc_at_input and src_pos is not None:
            output = output + 0.1 * src_pos

        if self.batch_first:
            output = output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)
            image = image.transpose(0, 1)
            memory = memory.transpose(0, 1)
            memory_pos = memory_pos.transpose(0, 1)
            memory_image = memory_image.transpose(0, 1)
            memory_image_pos = memory_image_pos.transpose(0, 1)

        if memory_image.shape[1] != memory.shape[1]:
            assert (memory.shape[1] - memory_image.shape[1]) == num_obj_ptr_tokens
            memory_image = torch.cat(
                [
                    memory_image,
                    torch.zeros(
                        (memory_image.shape[0], num_obj_ptr_tokens) + memory_image.shape[2:],
                        dtype=memory_image.dtype,
                        device=memory_image.device,
                    ),
                ],
                dim=1,
            )
            if memory_image_pos is not None:
                assert (memory_pos.shape[1] - memory_image_pos.shape[1]) == num_obj_ptr_tokens
                memory_image_pos = torch.cat([memory_image_pos, memory_pos[0:1, -num_obj_ptr_tokens:]], dim=1)

        for layer in self.layers:
            image, output = layer(
                image=image,
                tgt=output,
                memory_image=memory_image,
                memory=memory,
                image_pos=image_pos,
                query_pos=src_pos,
                memory_image_pos=memory_image_pos,
                memory_pos=memory_pos,
                num_k_exclude_rope=num_obj_ptr_tokens,
            )

        normed_output = self.norm(output + image) if self.use_image_in_output else self.norm(output)

        if self.batch_first:
            normed_output = normed_output.transpose(0, 1)
            src_pos = src_pos.transpose(0, 1)

        return {"memory": normed_output, "pos_embed": src_pos}


class MultiplexMaskDecoder(nn.Module):
    """Mask decoder that predicts masks for all slots of a bucket in one transformer pass.

    Ported from SAM 3.1 in the configuration shipped with sam3.1_multiplex.pt: separate per-slot
    iou/object-score tokens, ``multimask_outputs_only`` (3 mask tokens per slot, no single-mask
    token), and multimask tokens used for object pointers. The unreachable single-mask/stability
    fallback of the original is not ported.

    The token layout fed to the two-way transformer is
    [obj_score_token x M, iou_token x M, mask_tokens x (M * num_multimask_outputs)] with
    M = multiplex_count.
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        multiplex_count: int,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid: bool = False,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
        **kwargs,  # accepts (unused) dynamic_multimask_via_stability extra args for config parity
    ):
        """Initialize tokens, upscaling path, hypernetwork MLPs, and prediction heads."""
        from ultralytics.nn.modules import MLP, LayerNorm2d

        super().__init__()
        assert num_multimask_outputs > 0
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.multiplex_count = multiplex_count
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_output_per_object = num_multimask_outputs  # multimask_outputs_only=True
        self.num_mask_tokens = multiplex_count * self.num_mask_output_per_object
        self.pred_obj_scores = pred_obj_scores
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

        self.iou_token = nn.Embedding(multiplex_count, transformer_dim)
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(multiplex_count, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.use_high_res_features = use_high_res_features
        if use_high_res_features:
            self.conv_s0 = nn.Conv2d(transformer_dim, transformer_dim // 8, kernel_size=1, stride=1)
            self.conv_s1 = nn.Conv2d(transformer_dim, transformer_dim // 4, kernel_size=1, stride=1)

        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_output_per_object)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_output_per_object,
            iou_head_depth,
            sigmoid=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        multimask_output: bool,
        high_res_features: list[torch.Tensor] | None = None,
        extra_per_object_embeddings: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict per-slot masks for each bucket.

        Args:
            image_embeddings (torch.Tensor): Bucket image embeddings (B, C, H, W).
            image_pe (torch.Tensor): Positional encoding with the shape of image_embeddings (1, C, H, W).
            multimask_output (bool): Must be True (multimask_outputs_only configuration).
            high_res_features (list[torch.Tensor] | None): Two high-resolution feature maps.
            extra_per_object_embeddings (torch.Tensor | None): (B, multiplex_count, C) additions to
                the mask tokens (e.g. output suppression embeddings).

        Returns:
            (dict[str, torch.Tensor]): masks (B, M, num_multimask, H*4, W*4), iou_pred (B, M,
                num_multimask), sam_tokens_out (B, M, num_multimask, C), object_score_logits (B, M, 1).
        """
        assert multimask_output, "MultiplexMaskDecoder is configured with multimask_outputs_only=True"
        B = image_embeddings.shape[0]

        token_list = []
        if self.pred_obj_scores:
            token_list.append(self.obj_score_token.weight)
        token_list.append(self.iou_token.weight)
        tokens = torch.cat(token_list, dim=0).unsqueeze(0).expand(B, -1, -1)

        if extra_per_object_embeddings is not None:
            mask_tokens = self.mask_tokens.weight.view(
                1, self.multiplex_count, self.num_mask_output_per_object, -1
            ).expand(B, -1, -1, -1)
            mask_tokens = (mask_tokens + extra_per_object_embeddings.unsqueeze(2)).flatten(1, 2)
        else:
            mask_tokens = self.mask_tokens.weight.unsqueeze(0).expand(B, -1, -1)
        tokens = torch.cat([tokens, mask_tokens], dim=1)

        assert image_pe.size(0) == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = image_embeddings.shape

        hs, src = self.transformer(image_embeddings, pos_src, tokens)

        s = 0
        if self.pred_obj_scores:
            obj_score_token_out = hs[:, s : s + self.multiplex_count, :]
            s += self.multiplex_count
        iou_token_out = hs[:, s : s + self.multiplex_count, :]
        s += self.multiplex_count
        mask_tokens_out = hs[:, s : s + self.num_mask_tokens, :]

        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        mask_tokens_out = mask_tokens_out.view(B, self.multiplex_count, self.num_mask_output_per_object, -1)
        hyper_in = torch.stack(
            [
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, :, i, :])
                for i in range(self.num_mask_output_per_object)
            ],
            dim=2,
        )

        b, c, h, w = upscaled_embedding.shape
        masks = torch.bmm(hyper_in.flatten(1, 2), upscaled_embedding.view(b, c, h * w)).view(
            b, self.multiplex_count, self.num_mask_output_per_object, h, w
        )
        iou_pred = self.iou_prediction_head(iou_token_out).view(
            b, self.multiplex_count, self.num_mask_output_per_object
        )
        if self.pred_obj_scores:
            object_score_logits = self.pred_obj_score_head(obj_score_token_out)
        else:
            object_score_logits = 10.0 * iou_pred.new_ones(b, self.multiplex_count, 1)

        # sam_tokens_out: multimask tokens (use_multimask_token_for_obj_ptr with multimask_outputs_only)
        return {
            "masks": masks,
            "iou_pred": iou_pred,
            "sam_tokens_out": mask_tokens_out,
            "object_score_logits": object_score_logits,
        }
