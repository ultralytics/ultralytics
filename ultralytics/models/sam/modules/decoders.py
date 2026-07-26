# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch
from torch import nn

from ultralytics.nn.modules import MLP, LayerNorm2d


class MaskDecoder(nn.Module):
    """Decoder module for generating masks and their associated quality scores using a transformer architecture.

    This class predicts masks given image and prompt embeddings, utilizing a transformer to process the inputs and
    generate mask predictions along with their quality scores.

    Attributes:
        transformer_dim (int): Channel dimension for the transformer module.
        transformer (nn.Module): Transformer module used for mask prediction.
        num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
        iou_token (nn.Embedding): Embedding for the IoU token.
        num_mask_tokens (int): Number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for the mask tokens.
        output_upscaling (nn.Sequential): Neural network sequence for upscaling the output.
        output_hypernetworks_mlps (nn.ModuleList): Hypernetwork MLPs for generating masks.
        iou_prediction_head (nn.Module): MLP for predicting mask quality.

    Methods:
        forward: Predict masks given image and prompt embeddings.
        predict_masks: Internal method for mask prediction.

    Examples:
        >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
        >>> masks, iou_pred = decoder(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, multimask_output=True
        ... )
        >>> print(f"Predicted masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """Initialize the MaskDecoder module for generating masks and their associated quality scores.

        Args:
            transformer_dim (int): Channel dimension for the transformer module.
            transformer (nn.Module): Transformer module used for mask prediction.
            num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
            activation (type[nn.Module]): Type of activation to use when upscaling masks.
            iou_head_depth (int): Depth of the MLP used to predict mask quality.
            iou_head_hidden_dim (int): Hidden dimension of the MLP used to predict mask quality.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): Embeddings from the image encoder.
            image_pe (torch.Tensor): Positional encoding with the shape of image_embeddings.
            sparse_prompt_embeddings (torch.Tensor): Embeddings of the points and boxes.
            dense_prompt_embeddings (torch.Tensor): Embeddings of the mask inputs.
            multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
            masks (torch.Tensor): Batched predicted masks.
            iou_pred (torch.Tensor): Batched predictions of mask quality.

        Examples:
            >>> decoder = MaskDecoder(transformer_dim=256, transformer=transformer_module)
            >>> image_emb = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_emb = torch.rand(1, 2, 256)
            >>> dense_emb = torch.rand(1, 256, 64, 64)
            >>> masks, iou_pred = decoder(image_emb, image_pe, sparse_emb, dense_emb, multimask_output=True)
            >>> print(f"Masks shape: {masks.shape}, IoU predictions shape: {iou_pred.shape}")
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        mask_slice = slice(1, None) if multimask_output else slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict masks and quality scores using image and prompt embeddings via transformer architecture."""
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: list[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


class SAM2MaskDecoder(nn.Module):
    """Transformer-based decoder for predicting instance segmentation masks from image and prompt embeddings.

    This class extends the functionality of the MaskDecoder, incorporating additional features such as high-resolution
    feature processing, dynamic multimask output, and object score prediction.

    Attributes:
        transformer_dim (int): Channel dimension of the transformer.
        transformer (nn.Module): Transformer used to predict masks.
        num_multimask_outputs (int): Number of masks to predict when disambiguating masks.
        iou_token (nn.Embedding): Embedding for IOU token.
        num_mask_tokens (int): Total number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for mask tokens.
        pred_obj_scores (bool): Whether to predict object scores.
        obj_score_token (nn.Embedding): Embedding for object score token.
        use_multimask_token_for_obj_ptr (bool): Whether to use multimask token for object pointer.
        output_upscaling (nn.Sequential): Upscaling layers for output.
        use_high_res_features (bool): Whether to use high-resolution features.
        conv_s0 (nn.Conv2d): Convolutional layer for high-resolution features (s0).
        conv_s1 (nn.Conv2d): Convolutional layer for high-resolution features (s1).
        output_hypernetworks_mlps (nn.ModuleList): List of MLPs for output hypernetworks.
        iou_prediction_head (MLP): MLP for IOU prediction.
        pred_obj_score_head (nn.Linear | MLP): Linear layer or MLP for object score prediction.
        dynamic_multimask_via_stability (bool): Whether to use dynamic multimask via stability.
        dynamic_multimask_stability_delta (float): Delta value for dynamic multimask stability.
        dynamic_multimask_stability_thresh (float): Threshold for dynamic multimask stability.

    Methods:
        forward: Predict masks given image and prompt embeddings.
        predict_masks: Predict instance segmentation masks from image and prompt embeddings.
        _get_stability_scores: Compute mask stability scores based on IoU between thresholds.
        _dynamic_multimask_via_stability: Dynamically select the most stable mask output.

    Examples:
        >>> image_embeddings = torch.rand(1, 256, 64, 64)
        >>> image_pe = torch.rand(1, 256, 64, 64)
        >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
        >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
        >>> decoder = SAM2MaskDecoder(256, transformer)
        >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
        ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
        ... )
    """

    def __init__(
        self,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
        use_high_res_features: bool = False,
        iou_prediction_use_sigmoid=False,
        dynamic_multimask_via_stability=False,
        dynamic_multimask_stability_delta=0.05,
        dynamic_multimask_stability_thresh=0.98,
        pred_obj_scores: bool = False,
        pred_obj_scores_mlp: bool = False,
        use_multimask_token_for_obj_ptr: bool = False,
    ) -> None:
        """Initialize the SAM2MaskDecoder module for predicting instance segmentation masks.

        This decoder extends the functionality of MaskDecoder, incorporating additional features such as high-resolution
        feature processing, dynamic multimask output, and object score prediction.

        Args:
            transformer_dim (int): Channel dimension of the transformer.
            transformer (nn.Module): Transformer used to predict masks.
            num_multimask_outputs (int): Number of masks to predict when disambiguating masks.
            activation (type[nn.Module]): Type of activation to use when upscaling masks.
            iou_head_depth (int): Depth of the MLP used to predict mask quality.
            iou_head_hidden_dim (int): Hidden dimension of the MLP used to predict mask quality.
            use_high_res_features (bool): Whether to use high-resolution features.
            iou_prediction_use_sigmoid (bool): Whether to use sigmoid for IOU prediction.
            dynamic_multimask_via_stability (bool): Whether to use dynamic multimask via stability.
            dynamic_multimask_stability_delta (float): Delta value for dynamic multimask stability.
            dynamic_multimask_stability_thresh (float): Threshold for dynamic multimask stability.
            pred_obj_scores (bool): Whether to predict object scores.
            pred_obj_scores_mlp (bool): Whether to use MLP for object score prediction.
            use_multimask_token_for_obj_ptr (bool): Whether to use multimask token for object pointer.
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.num_multimask_outputs = num_multimask_outputs

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.num_mask_tokens = num_multimask_outputs + 1
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.pred_obj_scores = pred_obj_scores
        if self.pred_obj_scores:
            self.obj_score_token = nn.Embedding(1, transformer_dim)
        self.use_multimask_token_for_obj_ptr = use_multimask_token_for_obj_ptr

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
            [MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3) for _ in range(self.num_mask_tokens)]
        )

        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
            sigmoid=iou_prediction_use_sigmoid,
        )
        if self.pred_obj_scores:
            self.pred_obj_score_head = nn.Linear(transformer_dim, 1)
            if pred_obj_scores_mlp:
                self.pred_obj_score_head = MLP(transformer_dim, transformer_dim, 1, 3)

        # When outputting a single mask, optionally we can dynamically fall back to the best
        # multimask output token if the single mask output token gives low stability scores.
        self.dynamic_multimask_via_stability = dynamic_multimask_via_stability
        self.dynamic_multimask_stability_delta = dynamic_multimask_stability_delta
        self.dynamic_multimask_stability_thresh = dynamic_multimask_stability_thresh

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        multimask_output: bool,
        repeat_image: bool,
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): Embeddings from the image encoder with shape (B, C, H, W).
            image_pe (torch.Tensor): Positional encoding with the shape of image_embeddings (B, C, H, W).
            sparse_prompt_embeddings (torch.Tensor): Embeddings of the points and boxes with shape (B, N, C).
            dense_prompt_embeddings (torch.Tensor): Embeddings of the mask inputs with shape (B, C, H, W).
            multimask_output (bool): Whether to return multiple masks or a single mask.
            repeat_image (bool): Flag to repeat the image embeddings.
            high_res_features (list[torch.Tensor] | None, optional): Optional high-resolution features.

        Returns:
            masks (torch.Tensor): Batched predicted masks with shape (B, N, H, W).
            iou_pred (torch.Tensor): Batched predictions of mask quality with shape (B, N).
            sam_tokens_out (torch.Tensor): Batched SAM token for mask output with shape (B, N, C).
            object_score_logits (torch.Tensor): Batched object score logits with shape (B, 1).

        Examples:
            >>> image_embeddings = torch.rand(1, 256, 64, 64)
            >>> image_pe = torch.rand(1, 256, 64, 64)
            >>> sparse_prompt_embeddings = torch.rand(1, 2, 256)
            >>> dense_prompt_embeddings = torch.rand(1, 256, 64, 64)
            >>> decoder = SAM2MaskDecoder(256, transformer)
            >>> masks, iou_pred, sam_tokens_out, obj_score_logits = decoder.forward(
            ...     image_embeddings, image_pe, sparse_prompt_embeddings, dense_prompt_embeddings, True, False
            ... )
        """
        masks, iou_pred, mask_tokens_out, object_score_logits = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            repeat_image=repeat_image,
            high_res_features=high_res_features,
        )

        # Select the correct mask or masks for output
        if multimask_output:
            masks = masks[:, 1:, :, :]
            iou_pred = iou_pred[:, 1:]
        elif self.dynamic_multimask_via_stability and not self.training:
            masks, iou_pred = self._dynamic_multimask_via_stability(masks, iou_pred)
        else:
            masks = masks[:, 0:1, :, :]
            iou_pred = iou_pred[:, 0:1]

        if multimask_output and self.use_multimask_token_for_obj_ptr:
            sam_tokens_out = mask_tokens_out[:, 1:]  # [b, 3, c] shape
        else:
            # Take the mask output token. Here we *always* use the token for single mask output.
            # At test time, even if we track after 1-click (and using multimask_output=True),
            # we still take the single mask token here. The rationale is that we always track
            # after multiple clicks during training, so the past tokens seen during training
            # are always the single mask token (and we'll let it be the object-memory token).
            sam_tokens_out = mask_tokens_out[:, 0:1]  # [b, 1, c] shape

        return masks, iou_pred, sam_tokens_out, object_score_logits

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
        repeat_image: bool,
        high_res_features: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict instance segmentation masks from image and prompt embeddings using a transformer."""
        # Concatenate output tokens
        s = 0
        if self.pred_obj_scores:
            output_tokens = torch.cat(
                [
                    self.obj_score_token.weight,
                    self.iou_token.weight,
                    self.mask_tokens.weight,
                ],
                dim=0,
            )
            s = 1
        else:
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.shape[0], -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # Expand per-image data in batch direction to be per-mask
        if repeat_image:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            assert image_embeddings.shape[0] == tokens.shape[0]
            src = image_embeddings
        src = src + dense_prompt_embeddings
        assert image_pe.shape[0] == 1, "image_pe should have size 1 in batch dim (from `get_dense_pe()`)"
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        iou_token_out = hs[:, s, :]
        mask_tokens_out = hs[:, s + 1 : (s + 1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)
        if not self.use_high_res_features or high_res_features is None:
            upscaled_embedding = self.output_upscaling(src)
        else:
            dc1, ln1, act1, dc2, act2 = self.output_upscaling
            feat_s0, feat_s1 = high_res_features
            upscaled_embedding = act1(ln1(dc1(src) + feat_s1))
            upscaled_embedding = act2(dc2(upscaled_embedding) + feat_s0)

        hyper_in_list: list[torch.Tensor] = [
            self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]) for i in range(self.num_mask_tokens)
        ]
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)
        if self.pred_obj_scores:
            assert s == 1
            object_score_logits = self.pred_obj_score_head(hs[:, 0, :])
        else:
            # Obj scores logits - default to 10.0, i.e. assuming the object is present, sigmoid(10)=1
            object_score_logits = 10.0 * iou_pred.new_ones(iou_pred.shape[0], 1)

        return masks, iou_pred, mask_tokens_out, object_score_logits

    def _get_stability_scores(self, mask_logits):
        """Compute mask stability scores based on IoU between upper and lower thresholds."""
        mask_logits = mask_logits.flatten(-2)
        area_i = torch.sum(mask_logits > self.dynamic_multimask_stability_delta, dim=-1).float()
        area_u = torch.sum(mask_logits > -self.dynamic_multimask_stability_delta, dim=-1).float()
        return torch.where(area_u > 0, area_i / area_u, 1.0)

    def _dynamic_multimask_via_stability(self, all_mask_logits, all_iou_scores):
        """Dynamically select the most stable mask output based on stability scores and IoU predictions.

        This method is used when outputting a single mask. If the stability score from the current single-mask output
        (based on output token 0) falls below a threshold, it instead selects from multi-mask outputs (based on output
        tokens 1-3) the mask with the highest predicted IoU score. This ensures a valid mask for both clicking and
        tracking scenarios.

        Args:
            all_mask_logits (torch.Tensor): Logits for all predicted masks, shape (B, N, H, W) where B is batch size, N
                is number of masks (typically 4), and H, W are mask dimensions.
            all_iou_scores (torch.Tensor): Predicted IoU scores for all masks, shape (B, N).

        Returns:
            mask_logits_out (torch.Tensor): Selected mask logits, shape (B, 1, H, W).
            iou_scores_out (torch.Tensor): Selected IoU scores, shape (B, 1).

        Examples:
            >>> decoder = SAM2MaskDecoder(...)
            >>> all_mask_logits = torch.rand(2, 4, 256, 256)  # 2 images, 4 masks each
            >>> all_iou_scores = torch.rand(2, 4)
            >>> mask_logits, iou_scores = decoder._dynamic_multimask_via_stability(all_mask_logits, all_iou_scores)
            >>> print(mask_logits.shape, iou_scores.shape)
            torch.Size([2, 1, 256, 256]) torch.Size([2, 1])
        """
        # The best mask from multimask output tokens (1~3)
        multimask_logits = all_mask_logits[:, 1:, :, :]
        multimask_iou_scores = all_iou_scores[:, 1:]
        best_scores_inds = torch.argmax(multimask_iou_scores, dim=-1)
        batch_inds = torch.arange(multimask_iou_scores.shape[0], device=all_iou_scores.device)
        best_multimask_logits = multimask_logits[batch_inds, best_scores_inds]
        best_multimask_logits = best_multimask_logits.unsqueeze(1)
        best_multimask_iou_scores = multimask_iou_scores[batch_inds, best_scores_inds]
        best_multimask_iou_scores = best_multimask_iou_scores.unsqueeze(1)

        # The mask from singlemask output token 0 and its stability score
        singlemask_logits = all_mask_logits[:, 0:1, :, :]
        singlemask_iou_scores = all_iou_scores[:, 0:1]
        stability_scores = self._get_stability_scores(singlemask_logits)
        is_stable = stability_scores >= self.dynamic_multimask_stability_thresh

        # Dynamically fall back to best multimask output upon low stability scores.
        mask_logits_out = torch.where(
            is_stable[..., None, None].expand_as(singlemask_logits),
            singlemask_logits,
            best_multimask_logits,
        )
        iou_scores_out = torch.where(
            is_stable.expand_as(singlemask_iou_scores),
            singlemask_iou_scores,
            best_multimask_iou_scores,
        )
        return mask_logits_out, iou_scores_out
