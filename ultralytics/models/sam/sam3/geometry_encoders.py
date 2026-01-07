# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torchvision

from ultralytics.nn.modules.utils import _get_clones
from ultralytics.utils.ops import xywh2xyxy


def is_right_padded(mask: torch.Tensor):
    """Given a padding mask (following pytorch convention, 1s for padded values), returns whether the padding is on the
    right or not.
    """
    return (mask.long() == torch.sort(mask.long(), dim=-1)[0]).all()


def concat_padded_sequences(seq1, mask1, seq2, mask2, return_index: bool = False):
    """
    Concatenates two right-padded sequences, such that the resulting sequence
    is contiguous and also right-padded.

    Following pytorch's convention, tensors are sequence first, and the mask are
    batch first, with 1s for padded values.

    :param seq1: A tensor of shape (seq1_length, batch_size, hidden_size).
    :param mask1: A tensor of shape (batch_size, seq1_length).
    :param seq2: A tensor of shape (seq2_length, batch_size,  hidden_size).
    :param mask2: A tensor of shape (batch_size, seq2_length).
    :param return_index: If True, also returns the index of the ids of the element of seq2
        in the concatenated sequence. This can be used to retrieve the elements of seq2
    :return: A tuple (concatenated_sequence, concatenated_mask) if return_index is False,
        otherwise (concatenated_sequence, concatenated_mask, index).
    """
    seq1_length, batch_size, hidden_size = seq1.shape
    seq2_length, batch_size, hidden_size = seq2.shape

    assert batch_size == seq1.size(1) == seq2.size(1) == mask1.size(0) == mask2.size(0)
    assert hidden_size == seq1.size(2) == seq2.size(2)
    assert seq1_length == mask1.size(1)
    assert seq2_length == mask2.size(1)

    torch._assert(is_right_padded(mask1), "Mask is not right padded")
    torch._assert(is_right_padded(mask2), "Mask is not right padded")

    actual_seq1_lengths = (~mask1).sum(dim=-1)
    actual_seq2_lengths = (~mask2).sum(dim=-1)

    final_lengths = actual_seq1_lengths + actual_seq2_lengths
    max_length = seq1_length + seq2_length
    concatenated_mask = (
        torch.arange(max_length, device=seq2.device)[None].repeat(batch_size, 1) >= final_lengths[:, None]
    )

    # (max_len, batch_size, hidden_size)
    concatenated_sequence = torch.zeros((max_length, batch_size, hidden_size), device=seq2.device, dtype=seq2.dtype)
    concatenated_sequence[:seq1_length, :, :] = seq1

    # At this point, the element of seq1 are in the right place
    # We just need to shift the elements of seq2

    index = torch.arange(seq2_length, device=seq2.device)[:, None].repeat(1, batch_size)
    index = index + actual_seq1_lengths[None]

    concatenated_sequence = concatenated_sequence.scatter(0, index[:, :, None].expand(-1, -1, hidden_size), seq2)

    if return_index:
        return concatenated_sequence, concatenated_mask, index

    return concatenated_sequence, concatenated_mask


class Prompt:
    """Utility class to manipulate geometric prompts.

    We expect the sequences in pytorch convention, that is sequence first, batch second The dimensions are expected as
    follows: box_embeddings shape: N_boxes x B x C_box box_mask shape: B x N_boxes. Can be None if nothing is masked out
    point_embeddings shape: N_points x B x C_point point_mask shape: B x N_points. Can be None if nothing is masked out
    mask_embeddings shape: N_masks x B x 1 x H_mask x W_mask mask_mask shape: B x N_masks. Can be None if nothing is
    masked out

    We also store positive/negative labels. These tensors are also stored batch-first If they are None, we'll assume
    positive labels everywhere box_labels: long tensor of shape N_boxes x B point_labels: long tensor of shape N_points
    x B mask_labels: long tensor of shape N_masks x B
    """

    def __init__(self, box_embeddings=None, box_mask=None, box_labels=None):
        """Initialize the Prompt object."""
        # Check for null prompt
        # Check for null prompt
        if box_embeddings is None:
            self.box_embeddings = None
            self.box_labels = None
            self.box_mask = None
            return

        # Get sequence length, batch size, and device
        box_seq_len = box_embeddings.shape[0]
        bs = box_embeddings.shape[1]
        device = box_embeddings.device

        # Initialize labels and attention mask if not provided
        if box_labels is None:
            box_labels = torch.ones(box_seq_len, bs, device=device, dtype=torch.long)
        if box_mask is None:
            box_mask = torch.zeros(bs, box_seq_len, device=device, dtype=torch.bool)

        # Dimension checks
        assert list(box_embeddings.shape[:2]) == [box_seq_len, bs], (
            f"Wrong dimension for box embeddings. Expected [{box_seq_len}, {bs}, *] got {box_embeddings.shape}"
        )
        assert box_embeddings.shape[-1] == 4, (
            f"Expected box embeddings to have 4 coordinates, got {box_embeddings.shape[-1]}"
        )
        assert list(box_mask.shape) == [bs, box_seq_len], (
            f"Wrong dimension for box mask. Expected [{bs}, {box_seq_len}] got {box_mask.shape}"
        )
        assert list(box_labels.shape) == [box_seq_len, bs], (
            f"Wrong dimension for box labels. Expected [{box_seq_len}, {bs}] got {box_labels.shape}"
        )

        # Device checks
        assert box_embeddings.device == device, (
            f"Expected box embeddings to be on device {device}, got {box_embeddings.device}"
        )
        assert box_mask.device == device, f"Expected box mask to be on device {device}, got {box_mask.device}"
        assert box_labels.device == device, f"Expected box labels to be on device {device}, got {box_labels.device}"

        self.box_embeddings = box_embeddings
        self.box_mask = box_mask
        self.box_labels = box_labels

    def append_boxes(self, boxes, labels=None, mask=None):
        """Append box prompts to existing prompts.

        Args:
            boxes: Tensor of shape (N_new_boxes, B, 4) with normalized box coordinates
            labels: Optional tensor of shape (N_new_boxes, B) with positive/negative labels
            mask: Optional tensor of shape (B, N_new_boxes) for attention mask
        """
        if self.box_embeddings is None:
            # First boxes - initialize
            self.box_embeddings = boxes
            bs = boxes.shape[1]
            box_seq_len = boxes.shape[0]

            if labels is None:
                labels = torch.ones(box_seq_len, bs, device=boxes.device, dtype=torch.long)
            if mask is None:
                mask = torch.zeros(bs, box_seq_len, device=boxes.device, dtype=torch.bool)

            self.box_labels = labels
            self.box_mask = mask
            return

        # Append to existing boxes
        bs = self.box_embeddings.shape[1]
        assert boxes.shape[1] == bs, f"Batch size mismatch: expected {bs}, got {boxes.shape[1]}"

        if labels is None:
            labels = torch.ones(boxes.shape[0], bs, device=boxes.device, dtype=torch.long)
        if mask is None:
            mask = torch.zeros(bs, boxes.shape[0], dtype=torch.bool, device=boxes.device)

        assert list(boxes.shape[:2]) == list(labels.shape[:2]), (
            f"Shape mismatch between boxes {boxes.shape} and labels {labels.shape}"
        )

        # Concatenate using the helper function
        self.box_labels, _ = concat_padded_sequences(
            self.box_labels.unsqueeze(-1), self.box_mask, labels.unsqueeze(-1), mask
        )
        self.box_labels = self.box_labels.squeeze(-1)
        self.box_embeddings, self.box_mask = concat_padded_sequences(self.box_embeddings, self.box_mask, boxes, mask)


class SequenceGeometryEncoder(nn.Module):
    """Encoder for geometric box prompts. Assumes boxes are passed in the "normalized CxCyWH" format.

    Boxes can be encoded with any of the three possibilities:
    - direct projection: linear projection from coordinate space to d_model
    - pooling: RoI align features from the backbone
    - pos encoder: position encoding of the box center

    These three options are mutually compatible and will be summed if multiple are selected.

    As an alternative, boxes can be encoded as two corner points (top-left and bottom-right).

    The encoded sequence can be further processed with a transformer.
    """

    def __init__(
        self,
        encode_boxes_as_points: bool,
        boxes_direct_project: bool,
        boxes_pool: bool,
        boxes_pos_enc: bool,
        d_model: int,
        pos_enc,
        num_layers: int,
        layer: nn.Module,
        roi_size: int = 7,
        add_cls: bool = True,
        add_post_encode_proj: bool = True,
        use_act_ckpt: bool = False,
    ):
        """Initialize the SequenceGeometryEncoder."""
        super().__init__()

        self.d_model = d_model
        self.pos_enc = pos_enc
        self.encode_boxes_as_points = encode_boxes_as_points
        self.roi_size = roi_size

        # Label embeddings: 2 labels if encoding as boxes (pos/neg)
        # 6 labels if encoding as points (regular pos/neg, top-left pos/neg, bottom-right pos/neg)
        num_labels = 6 if self.encode_boxes_as_points else 2
        self.label_embed = torch.nn.Embedding(num_labels, self.d_model)

        # CLS token for pooling
        self.cls_embed = None
        if add_cls:
            self.cls_embed = torch.nn.Embedding(1, self.d_model)

        # Point encoding (used when encode_boxes_as_points is True)
        if encode_boxes_as_points:
            self.points_direct_project = nn.Linear(2, self.d_model)
            self.points_pool_project = None
            self.points_pos_enc_project = None
        else:
            # Box encoding modules
            assert boxes_direct_project or boxes_pos_enc or boxes_pool, "Error: need at least one way to encode boxes"
            self.points_direct_project = None
            self.points_pool_project = None
            self.points_pos_enc_project = None

            self.boxes_direct_project = None
            self.boxes_pool_project = None
            self.boxes_pos_enc_project = None

            if boxes_direct_project:
                self.boxes_direct_project = nn.Linear(4, self.d_model)
            if boxes_pool:
                self.boxes_pool_project = nn.Conv2d(self.d_model, self.d_model, self.roi_size)
            if boxes_pos_enc:
                self.boxes_pos_enc_project = nn.Linear(self.d_model + 2, self.d_model)

        self.final_proj = None
        if add_post_encode_proj:
            self.final_proj = nn.Linear(self.d_model, self.d_model)
            self.norm = nn.LayerNorm(self.d_model)

        self.img_pre_norm = nn.Identity()
        if self.points_pool_project is not None or self.boxes_pool_project is not None:
            self.img_pre_norm = nn.LayerNorm(self.d_model)

        self.encode = None
        if num_layers > 0:
            assert add_cls, "It's currently highly recommended to add a CLS when using a transformer"
            self.encode = _get_clones(layer, num_layers)
            self.encode_norm = nn.LayerNorm(self.d_model)

        self.use_act_ckpt = use_act_ckpt

    def _encode_points(self, points, points_mask, points_labels, img_feats):
        """Encode points (used when boxes are converted to corner points)."""
        # Direct projection of coordinates
        points_embed = self.points_direct_project(points.to(img_feats.dtype))

        # Add label embeddings
        type_embed = self.label_embed(points_labels.long())
        return type_embed + points_embed, points_mask

    def _encode_boxes(self, boxes, boxes_mask, boxes_labels, img_feats: torch.Tensor):
        """Encode boxes using configured encoding methods."""
        boxes_embed = None
        n_boxes, bs = boxes.shape[:2]

        if self.boxes_direct_project is not None:
            proj = self.boxes_direct_project(boxes.to(img_feats.dtype))
            boxes_embed = proj

        if self.boxes_pool_project is not None:
            H, W = img_feats.shape[-2:]

            # Convert boxes to xyxy format and denormalize
            boxes_xyxy = xywh2xyxy(boxes.to(img_feats.dtype))
            scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype)
            scale = scale.to(device=boxes_xyxy.device, non_blocking=True)
            scale = scale.view(1, 1, 4)
            boxes_xyxy = boxes_xyxy * scale

            # RoI align
            sampled = torchvision.ops.roi_align(img_feats, boxes_xyxy.transpose(0, 1).unbind(0), self.roi_size)
            assert list(sampled.shape) == [
                bs * n_boxes,
                self.d_model,
                self.roi_size,
                self.roi_size,
            ]
            proj = self.boxes_pool_project(sampled)
            proj = proj.view(bs, n_boxes, self.d_model).transpose(0, 1)

            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        if self.boxes_pos_enc_project is not None:
            cx, cy, w, h = boxes.unbind(-1)
            enc = self.pos_enc.encode_boxes(cx.flatten(), cy.flatten(), w.flatten(), h.flatten())
            enc = enc.view(boxes.shape[0], boxes.shape[1], enc.shape[-1])

            proj = self.boxes_pos_enc_project(enc.to(img_feats.dtype))
            if boxes_embed is None:
                boxes_embed = proj
            else:
                boxes_embed = boxes_embed + proj

        # Add label embeddings
        type_embed = self.label_embed(boxes_labels.long())
        return type_embed + boxes_embed, boxes_mask

    def forward(self, geo_prompt: Prompt, img_feats, img_sizes, img_pos_embeds=None):
        """Encode geometric box prompts.

        Args:
            geo_prompt: Prompt object containing box embeddings, masks, and labels
            img_feats: List of image features from backbone
            img_sizes: List of (H, W) tuples for each feature level
            img_pos_embeds: Optional position embeddings for image features

        Returns:
            Tuple of (encoded_embeddings, attention_mask)
        """
        boxes = geo_prompt.box_embeddings
        boxes_mask = geo_prompt.box_mask
        boxes_labels = geo_prompt.box_labels

        seq_first_img_feats = img_feats[-1]  # [H*W, B, C]
        seq_first_img_pos_embeds = (
            img_pos_embeds[-1] if img_pos_embeds is not None else torch.zeros_like(seq_first_img_feats)
        )

        # Prepare image features for pooling if needed
        if self.points_pool_project or self.boxes_pool_project:
            assert len(img_feats) == len(img_sizes)
            cur_img_feat = img_feats[-1]
            cur_img_feat = self.img_pre_norm(cur_img_feat)
            H, W = img_sizes[-1]
            assert cur_img_feat.shape[0] == H * W
            N, C = cur_img_feat.shape[-2:]
            # Reshape to NxCxHxW
            cur_img_feat = cur_img_feat.permute(1, 2, 0)
            cur_img_feat = cur_img_feat.view(N, C, H, W)
            img_feats = cur_img_feat

        if self.encode_boxes_as_points:
            # Convert boxes to corner points
            assert boxes is not None and boxes.shape[-1] == 4

            boxes_xyxy = xywh2xyxy(boxes)
            top_left, bottom_right = boxes_xyxy.split(split_size=2, dim=-1)

            # Adjust labels for corner points (offset by 2 and 4)
            labels_tl = boxes_labels + 2
            labels_br = boxes_labels + 4

            # Concatenate top-left and bottom-right points
            points = torch.cat([top_left, bottom_right], dim=0)
            points_labels = torch.cat([labels_tl, labels_br], dim=0)
            points_mask = torch.cat([boxes_mask, boxes_mask], dim=1)

            final_embeds, final_mask = self._encode_points(
                points=points,
                points_mask=points_mask,
                points_labels=points_labels,
                img_feats=img_feats,
            )
        else:
            # Encode boxes directly
            final_embeds, final_mask = self._encode_boxes(
                boxes=boxes,
                boxes_mask=boxes_mask,
                boxes_labels=boxes_labels,
                img_feats=img_feats,
            )

        bs = final_embeds.shape[1]
        assert final_mask.shape[0] == bs

        # Add CLS token if configured
        if self.cls_embed is not None:
            cls = self.cls_embed.weight.view(1, 1, self.d_model).repeat(1, bs, 1)
            cls_mask = torch.zeros(bs, 1, dtype=final_mask.dtype, device=final_mask.device)
            final_embeds, final_mask = concat_padded_sequences(final_embeds, final_mask, cls, cls_mask)

        # Final projection
        if self.final_proj is not None:
            final_embeds = self.norm(self.final_proj(final_embeds))

        # Transformer encoding layers
        if self.encode is not None:
            for lay in self.encode:
                final_embeds = lay(
                    tgt=final_embeds,
                    memory=seq_first_img_feats,
                    tgt_key_padding_mask=final_mask,
                    pos=seq_first_img_pos_embeds,
                )
            final_embeds = self.encode_norm(final_embeds)

        return final_embeds, final_mask
