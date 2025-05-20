# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from typing import List, Optional, Tuple, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules import LayerNorm2d

from .blocks import (
    Block,
    CXBlock,
    Fuser,
    MaskDownSampler,
    MultiScaleBlock,
    PatchEmbed,
    PositionEmbeddingRandom,
    PositionEmbeddingSine,
)


class ImageEncoderViT(nn.Module):
    """
    An image encoder using Vision Transformer (ViT) architecture for encoding images into a compact latent space.

    This class processes images by splitting them into patches, applying transformer blocks, and generating a final
    encoded representation through a neck module.

    Attributes:
        img_size (int): Dimension of input images, assumed to be square.
        patch_embed (PatchEmbed): Module for patch embedding.
        pos_embed (nn.Parameter | None): Absolute positional embedding for patches.
        blocks (nn.ModuleList): List of transformer blocks for processing patch embeddings.
        neck (nn.Sequential): Neck module to further process the output.

    Methods:
        forward: Processes input through patch embedding, positional embedding, blocks, and neck.

    Examples:
        >>> import torch
        >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
        >>> input_image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(input_image)
        >>> print(output.shape)
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        qkv_bias: bool = True,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        act_layer: Type[nn.Module] = nn.GELU,
        use_abs_pos: bool = True,
        use_rel_pos: bool = False,
        rel_pos_zero_init: bool = True,
        window_size: int = 0,
        global_attn_indexes: Tuple[int, ...] = (),
    ) -> None:
        """
        Initialize an ImageEncoderViT instance for encoding images using Vision Transformer architecture.

        Args:
            img_size (int): Input image size, assumed to be square.
            patch_size (int): Size of image patches.
            in_chans (int): Number of input image channels.
            embed_dim (int): Dimension of patch embeddings.
            depth (int): Number of transformer blocks.
            num_heads (int): Number of attention heads in each block.
            mlp_ratio (float): Ratio of MLP hidden dimension to embedding dimension.
            out_chans (int): Number of output channels from the neck module.
            qkv_bias (bool): If True, adds learnable bias to query, key, value projections.
            norm_layer (Type[nn.Module]): Type of normalization layer to use.
            act_layer (Type[nn.Module]): Type of activation layer to use.
            use_abs_pos (bool): If True, uses absolute positional embeddings.
            use_rel_pos (bool): If True, adds relative positional embeddings to attention maps.
            rel_pos_zero_init (bool): If True, initializes relative positional parameters to zero.
            window_size (int): Size of attention window for windowed attention blocks.
            global_attn_indexes (Tuple[int, ...]): Indices of blocks that use global attention.

        Examples:
            >>> encoder = ImageEncoderViT(img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12)
            >>> input_image = torch.randn(1, 3, 224, 224)
            >>> output = encoder(input_image)
            >>> print(output.shape)
        """
        super().__init__()
        self.img_size = img_size

        self.patch_embed = PatchEmbed(
            kernel_size=(patch_size, patch_size),
            stride=(patch_size, patch_size),
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.pos_embed: Optional[nn.Parameter] = None
        if use_abs_pos:
            # Initialize absolute positional embedding with pretrain image size.
            self.pos_embed = nn.Parameter(torch.zeros(1, img_size // patch_size, img_size // patch_size, embed_dim))

        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                act_layer=act_layer,
                use_rel_pos=use_rel_pos,
                rel_pos_zero_init=rel_pos_zero_init,
                window_size=window_size if i not in global_attn_indexes else 0,
                input_size=(img_size // patch_size, img_size // patch_size),
            )
            self.blocks.append(block)

        self.neck = nn.Sequential(
            nn.Conv2d(
                embed_dim,
                out_chans,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
            nn.Conv2d(
                out_chans,
                out_chans,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(out_chans),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through patch embedding, positional embedding, transformer blocks, and neck module."""
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            pos_embed = (
                F.interpolate(self.pos_embed.permute(0, 3, 1, 2), scale_factor=self.img_size / 1024).permute(0, 2, 3, 1)
                if self.img_size != 1024
                else self.pos_embed
            )
            x = x + pos_embed
        for blk in self.blocks:
            x = blk(x)
        return self.neck(x.permute(0, 3, 1, 2))


class PromptEncoder(nn.Module):
    """
    Encodes different types of prompts for input to SAM's mask decoder, producing sparse and dense embeddings.

    Attributes:
        embed_dim (int): Dimension of the embeddings.
        input_image_size (Tuple[int, int]): Size of the input image as (H, W).
        image_embedding_size (Tuple[int, int]): Spatial size of the image embedding as (H, W).
        pe_layer (PositionEmbeddingRandom): Module for random position embedding.
        num_point_embeddings (int): Number of point embeddings for different types of points.
        point_embeddings (nn.ModuleList): List of point embeddings.
        not_a_point_embed (nn.Embedding): Embedding for points that are not part of any label.
        mask_input_size (Tuple[int, int]): Size of the input mask.
        mask_downscaling (nn.Sequential): Neural network for downscaling the mask.
        no_mask_embed (nn.Embedding): Embedding for cases where no mask is provided.

    Methods:
        get_dense_pe: Returns the positional encoding used to encode point prompts.
        forward: Embeds different types of prompts, returning both sparse and dense embeddings.

    Examples:
        >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
        >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
        >>> boxes = torch.rand(1, 2, 2)
        >>> masks = torch.rand(1, 1, 256, 256)
        >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
        >>> print(sparse_embeddings.shape, dense_embeddings.shape)
        torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
    """

    def __init__(
        self,
        embed_dim: int,
        image_embedding_size: Tuple[int, int],
        input_image_size: Tuple[int, int],
        mask_in_chans: int,
        activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Initialize the PromptEncoder module for encoding various types of prompts.

        Args:
            embed_dim (int): The dimension of the embeddings.
            image_embedding_size (Tuple[int, int]): The spatial size of the image embedding as (H, W).
            input_image_size (Tuple[int, int]): The padded size of the input image as (H, W).
            mask_in_chans (int): The number of hidden channels used for encoding input masks.
            activation (Type[nn.Module]): The activation function to use when encoding input masks.

        Examples:
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_embeddings, dense_embeddings = prompt_encoder(points, boxes, masks)
            >>> print(sparse_embeddings.shape, dense_embeddings.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.input_image_size = input_image_size
        self.image_embedding_size = image_embedding_size
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for _ in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)
        self.not_a_point_embed = nn.Embedding(1, embed_dim)

        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans // 4),
            activation(),
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),
            LayerNorm2d(mask_in_chans),
            activation(),
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Return the dense positional encoding used for encoding point prompts.

        Generate a positional encoding for a dense set of points matching the shape of the image
        encoding. The encoding is used to provide spatial information to the model when processing point prompts.

        Returns:
            (torch.Tensor): Positional encoding tensor with shape (1, embed_dim, H, W), where H and W are the
                height and width of the image embedding size, respectively.

        Examples:
            >>> prompt_encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> dense_pe = prompt_encoder.get_dense_pe()
            >>> print(dense_pe.shape)
            torch.Size([1, 256, 64, 64])
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(self, points: torch.Tensor, labels: torch.Tensor, pad: bool) -> torch.Tensor:
        """Embed point prompts by applying positional encoding and label-specific embeddings."""
        points = points + 0.5  # Shift to center of pixel
        if pad:
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)
            points = torch.cat([points, padding_point], dim=1)
            labels = torch.cat([labels, padding_label], dim=1)
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight
        point_embedding[labels == 1] += self.point_embeddings[1].weight
        point_embedding[labels == 2] += self.point_embeddings[2].weight
        point_embedding[labels == 3] += self.point_embeddings[3].weight
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embed box prompts by applying positional encoding and adding corner embeddings."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embed mask inputs by downscaling and processing through convolutional layers."""
        return self.mask_downscaling(masks)

    @staticmethod
    def _get_batch_size(
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        """Get the batch size of the output given the batch size of the input prompts."""
        if points is not None:
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        """Return the device of the first point embedding's weight tensor."""
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        points: Optional[Tuple[torch.Tensor, torch.Tensor]],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embed different types of prompts, returning both sparse and dense embeddings.

        Args:
            points (Tuple[torch.Tensor, torch.Tensor] | None): Point coordinates and labels to embed. The first
                tensor contains coordinates with shape (B, N, 2), and the second tensor contains labels with
                shape (B, N).
            boxes (torch.Tensor | None): Boxes to embed with shape (B, M, 2, 2), where M is the number of boxes.
            masks (torch.Tensor | None): Masks to embed with shape (B, 1, H, W).

        Returns:
            (Tuple[torch.Tensor, torch.Tensor]): A tuple containing:
                - sparse_embeddings (torch.Tensor): Sparse embeddings for points and boxes with shape (B, N, embed_dim).
                - dense_embeddings (torch.Tensor): Dense embeddings for masks of shape (B, embed_dim, embed_H, embed_W).

        Examples:
            >>> encoder = PromptEncoder(256, (64, 64), (1024, 1024), 16)
            >>> points = (torch.rand(1, 5, 2), torch.randint(0, 4, (1, 5)))
            >>> boxes = torch.rand(1, 2, 2, 2)
            >>> masks = torch.rand(1, 1, 256, 256)
            >>> sparse_emb, dense_emb = encoder(points, boxes, masks)
            >>> print(sparse_emb.shape, dense_emb.shape)
            torch.Size([1, 7, 256]) torch.Size([1, 256, 64, 64])
        """
        bs = self._get_batch_size(points, boxes, masks)
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        if points is not None:
            coords, labels = points
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )

        return sparse_embeddings, dense_embeddings


class MemoryEncoder(nn.Module):
    """
    Encode pixel features and masks into a memory representation for efficient image segmentation.

    This class processes pixel-level features and masks, fusing them to generate encoded memory representations
    suitable for downstream tasks in image segmentation models like SAM (Segment Anything Model).

    Attributes:
        mask_downsampler (MaskDownSampler): Module for downsampling input masks.
        pix_feat_proj (nn.Conv2d): Convolutional layer for projecting pixel features.
        fuser (Fuser): Module for fusing pixel features and masks.
        position_encoding (PositionEmbeddingSine): Module for adding positional encoding to features.
        out_proj (nn.Module): Output projection layer, either nn.Identity or nn.Conv2d.

    Methods:
        forward: Process input pixel features and masks to generate encoded memory representations.

    Examples:
        >>> import torch
        >>> encoder = MemoryEncoder(out_dim=256, in_dim=256)
        >>> pix_feat = torch.randn(1, 256, 64, 64)
        >>> masks = torch.randn(1, 1, 64, 64)
        >>> encoded_feat, pos = encoder(pix_feat, masks)
        >>> print(encoded_feat.shape, pos.shape)
        torch.Size([1, 256, 64, 64]) torch.Size([1, 128, 64, 64])
    """

    def __init__(
        self,
        out_dim,
        in_dim=256,  # in_dim of pix_feats
    ):
        """
        Initialize the MemoryEncoder for encoding pixel features and masks into memory representations.

        This encoder processes pixel-level features and masks, fusing them to generate encoded memory representations
        suitable for downstream tasks in image segmentation models like SAM (Segment Anything Model).

        Args:
            out_dim (int): Output dimension of the encoded features.
            in_dim (int): Input dimension of the pixel features. Default is 256.

        Examples:
            >>> encoder = MemoryEncoder(out_dim=256, in_dim=256)
            >>> pix_feat = torch.randn(1, 256, 64, 64)
            >>> masks = torch.randn(1, 1, 64, 64)
            >>> encoded_feat, pos = encoder(pix_feat, masks)
            >>> print(encoded_feat.shape, pos.shape)
            torch.Size([1, 256, 64, 64]) torch.Size([1, 128, 64, 64])
        """
        super().__init__()

        self.mask_downsampler = MaskDownSampler(kernel_size=3, stride=2, padding=1)

        self.pix_feat_proj = nn.Conv2d(in_dim, in_dim, kernel_size=1)
        self.fuser = Fuser(CXBlock(dim=256), num_layers=2)
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=64)
        self.out_proj = nn.Identity()
        if out_dim != in_dim:
            self.out_proj = nn.Conv2d(in_dim, out_dim, kernel_size=1)

    def forward(
        self,
        pix_feat: torch.Tensor,
        masks: torch.Tensor,
        skip_mask_sigmoid: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Process pixel features and masks to generate encoded memory representations for segmentation."""
        if not skip_mask_sigmoid:
            masks = F.sigmoid(masks)
        masks = self.mask_downsampler(masks)

        # Fuse pix_feats and downsampled masks, in case the visual features are on CPU, cast them to CUDA
        pix_feat = pix_feat.to(masks.device)

        x = self.pix_feat_proj(pix_feat)
        x = x + masks
        x = self.fuser(x)
        x = self.out_proj(x)

        pos = self.position_encoding(x).to(x.dtype)

        return {"vision_features": x, "vision_pos_enc": [pos]}


class ImageEncoder(nn.Module):
    """
    Encode images using a trunk-neck architecture, producing multiscale features and positional encodings.

    This class combines a trunk network for feature extraction with a neck network for feature refinement
    and positional encoding generation. It can optionally discard the lowest resolution features.

    Attributes:
        trunk (nn.Module): The trunk network for initial feature extraction.
        neck (nn.Module): The neck network for feature refinement and positional encoding generation.
        scalp (int): Number of lowest resolution feature levels to discard.

    Methods:
        forward: Process the input image through the trunk and neck networks.

    Examples:
        >>> trunk = SomeTrunkNetwork()
        >>> neck = SomeNeckNetwork()
        >>> encoder = ImageEncoder(trunk, neck, scalp=1)
        >>> image = torch.randn(1, 3, 224, 224)
        >>> output = encoder(image)
        >>> print(output.keys())
        dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
    """

    def __init__(
        self,
        trunk: nn.Module,
        neck: nn.Module,
        scalp: int = 0,
    ):
        """
        Initialize the ImageEncoder with trunk and neck networks for feature extraction and refinement.

        This encoder combines a trunk network for feature extraction with a neck network for feature refinement
        and positional encoding generation. It can optionally discard the lowest resolution features.

        Args:
            trunk (nn.Module): The trunk network for initial feature extraction.
            neck (nn.Module): The neck network for feature refinement and positional encoding generation.
            scalp (int): Number of lowest resolution feature levels to discard.

        Examples:
            >>> trunk = SomeTrunkNetwork()
            >>> neck = SomeNeckNetwork()
            >>> encoder = ImageEncoder(trunk, neck, scalp=1)
            >>> image = torch.randn(1, 3, 224, 224)
            >>> output = encoder(image)
            >>> print(output.keys())
            dict_keys(['vision_features', 'vision_pos_enc', 'backbone_fpn'])
        """
        super().__init__()
        self.trunk = trunk
        self.neck = neck
        self.scalp = scalp
        assert self.trunk.channel_list == self.neck.backbone_channel_list, (
            f"Channel dims of trunk {self.trunk.channel_list} and neck {self.neck.backbone_channel_list} do not match."
        )

    def forward(self, sample: torch.Tensor):
        """Encode input through patch embedding, positional embedding, transformer blocks, and neck module."""
        features, pos = self.neck(self.trunk(sample))
        if self.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.scalp], pos[: -self.scalp]

        src = features[-1]
        return {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }


class FpnNeck(nn.Module):
    """
    A Feature Pyramid Network (FPN) neck variant for multiscale feature fusion in object detection models.

    This FPN variant removes the output convolution and uses bicubic interpolation for feature resizing,
    similar to ViT positional embedding interpolation.

    Attributes:
        position_encoding (PositionEmbeddingSine): Sinusoidal positional encoding module.
        convs (nn.ModuleList): List of convolutional layers for each backbone level.
        backbone_channel_list (List[int]): List of channel dimensions from the backbone.
        fpn_interp_model (str): Interpolation mode for FPN feature resizing.
        fuse_type (str): Type of feature fusion, either 'sum' or 'avg'.
        fpn_top_down_levels (List[int]): Levels to have top-down features in outputs.

    Methods:
        forward: Perform forward pass through the FPN neck.

    Examples:
        >>> backbone_channels = [64, 128, 256, 512]
        >>> fpn_neck = FpnNeck(256, backbone_channels)
        >>> inputs = [torch.rand(1, c, 32, 32) for c in backbone_channels]
        >>> outputs, positions = fpn_neck(inputs)
        >>> print(len(outputs), len(positions))
        4 4
    """

    def __init__(
        self,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """
        Initializes a modified Feature Pyramid Network (FPN) neck.

        This FPN variant removes the output convolution and uses bicubic interpolation for feature resizing,
        similar to ViT positional embedding interpolation.

        Args:
            d_model (int): Dimension of the model.
            backbone_channel_list (List[int]): List of channel dimensions from the backbone.
            kernel_size (int): Kernel size for the convolutional layers.
            stride (int): Stride for the convolutional layers.
            padding (int): Padding for the convolutional layers.
            fpn_interp_model (str): Interpolation mode for FPN feature resizing.
            fuse_type (str): Type of feature fusion, either 'sum' or 'avg'.
            fpn_top_down_levels (Optional[List[int]]): Levels to have top-down features in outputs.

        Examples:
            >>> backbone_channels = [64, 128, 256, 512]
            >>> fpn_neck = FpnNeck(256, backbone_channels)
            >>> print(fpn_neck)
        """
        super().__init__()
        self.position_encoding = PositionEmbeddingSine(num_pos_feats=256)
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in {"sum", "avg"}
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):
        """
        Performs forward pass through the Feature Pyramid Network (FPN) neck.

        This method processes a list of input tensors from the backbone through the FPN, applying lateral connections
        and top-down feature fusion. It generates output feature maps and corresponding positional encodings.

        Args:
            xs (List[torch.Tensor]): List of input tensors from the backbone, each with shape (B, C, H, W).

        Returns:
            (Tuple[List[torch.Tensor], List[torch.Tensor]]): A tuple containing:
                - out (List[torch.Tensor]): List of output feature maps after FPN processing, each with shape
                  (B, d_model, H, W).
                - pos (List[torch.Tensor]): List of positional encodings corresponding to each output feature map.

        Examples:
            >>> fpn_neck = FpnNeck(d_model=256, backbone_channel_list=[64, 128, 256, 512])
            >>> inputs = [torch.rand(1, c, 32, 32) for c in [64, 128, 256, 512]]
            >>> outputs, positions = fpn_neck(inputs)
            >>> print(len(outputs), len(positions))
            4 4
        """
        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(None if self.fpn_interp_model == "nearest" else False),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


class Hiera(nn.Module):
    """
    Hierarchical vision transformer for efficient multiscale feature extraction in image processing tasks.

    This class implements a Hiera model, which is a hierarchical vision transformer architecture designed for
    efficient multiscale feature extraction. It uses a series of transformer blocks organized into stages,
    with optional pooling and global attention mechanisms.

    Attributes:
        window_spec (Tuple[int, ...]): Window sizes for each stage.
        q_stride (Tuple[int, int]): Downsampling stride between stages.
        stage_ends (List[int]): Indices of the last block in each stage.
        q_pool_blocks (List[int]): Indices of blocks where pooling is applied.
        return_interm_layers (bool): Whether to return intermediate layer outputs.
        patch_embed (PatchEmbed): Module for patch embedding.
        global_att_blocks (Tuple[int, ...]): Indices of blocks with global attention.
        window_pos_embed_bkg_spatial_size (Tuple[int, int]): Spatial size for window positional embedding background.
        pos_embed (nn.Parameter): Positional embedding for the background.
        pos_embed_window (nn.Parameter): Positional embedding for the window.
        blocks (nn.ModuleList): List of MultiScaleBlock modules.
        channel_list (List[int]): List of output channel dimensions for each stage.

    Methods:
        _get_pos_embed: Generate positional embeddings by interpolating and combining window and background embeddings.
        forward: Perform the forward pass through the Hiera model.

    Examples:
        >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output_features = model(input_tensor)
        >>> for feat in output_features:
        ...     print(feat.shape)
    """

    def __init__(
        self,
        embed_dim: int = 96,  # initial embed dim
        num_heads: int = 1,  # initial number of heads
        drop_path_rate: float = 0.0,  # stochastic depth
        q_pool: int = 3,  # number of q_pool stages
        q_stride: Tuple[int, int] = (2, 2),  # downsample stride bet. stages
        stages: Tuple[int, ...] = (2, 3, 16, 3),  # blocks per stage
        dim_mul: float = 2.0,  # dim_mul factor at stage shift
        head_mul: float = 2.0,  # head_mul factor at stage shift
        window_pos_embed_bkg_spatial_size: Tuple[int, int] = (14, 14),
        # window size per stage, when not using global att.
        window_spec: Tuple[int, ...] = (
            8,
            4,
            14,
            7,
        ),
        # global attn in these blocks
        global_att_blocks: Tuple[int, ...] = (
            12,
            16,
            20,
        ),
        return_interm_layers=True,  # return feats from every stage
    ):
        """
        Initialize a Hiera model, a hierarchical vision transformer for efficient multiscale feature extraction.

        Hiera is a hierarchical vision transformer architecture designed for efficient multiscale feature extraction
        in image processing tasks. It uses a series of transformer blocks organized into stages, with optional
        pooling and global attention mechanisms.

        Args:
            embed_dim (int): Initial embedding dimension for the model.
            num_heads (int): Initial number of attention heads.
            drop_path_rate (float): Stochastic depth rate.
            q_pool (int): Number of query pooling stages.
            q_stride (Tuple[int, int]): Downsampling stride between stages.
            stages (Tuple[int, ...]): Number of blocks per stage.
            dim_mul (float): Dimension multiplier factor at stage transitions.
            head_mul (float): Head multiplier factor at stage transitions.
            window_pos_embed_bkg_spatial_size (Tuple[int, int]): Spatial size for window positional embedding background.
            window_spec (Tuple[int, ...]): Window sizes for each stage when not using global attention.
            global_att_blocks (Tuple[int, ...]): Indices of blocks that use global attention.
            return_interm_layers (bool): Whether to return intermediate layer outputs.

        Examples:
            >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> output_features = model(input_tensor)
            >>> for feat in output_features:
            ...     print(feat.shape)
        """
        super().__init__()

        assert len(stages) == len(window_spec)
        self.window_spec = window_spec

        depth = sum(stages)
        self.q_stride = q_stride
        self.stage_ends = [sum(stages[:i]) - 1 for i in range(1, len(stages) + 1)]
        assert 0 <= q_pool <= len(self.stage_ends[:-1])
        self.q_pool_blocks = [x + 1 for x in self.stage_ends[:-1]][:q_pool]
        self.return_interm_layers = return_interm_layers

        self.patch_embed = PatchEmbed(
            embed_dim=embed_dim,
            kernel_size=(7, 7),
            stride=(4, 4),
            padding=(3, 3),
        )
        # Which blocks have global att?
        self.global_att_blocks = global_att_blocks

        # Windowed positional embedding (https://arxiv.org/abs/2311.05613)
        self.window_pos_embed_bkg_spatial_size = window_pos_embed_bkg_spatial_size
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, *self.window_pos_embed_bkg_spatial_size))
        self.pos_embed_window = nn.Parameter(torch.zeros(1, embed_dim, self.window_spec[0], self.window_spec[0]))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        cur_stage = 1
        self.blocks = nn.ModuleList()

        for i in range(depth):
            dim_out = embed_dim
            # lags by a block, so first block of
            # next stage uses an initial window size
            # of previous stage and final window size of current stage
            window_size = self.window_spec[cur_stage - 1]

            if self.global_att_blocks is not None:
                window_size = 0 if i in self.global_att_blocks else window_size

            if i - 1 in self.stage_ends:
                dim_out = int(embed_dim * dim_mul)
                num_heads = int(num_heads * head_mul)
                cur_stage += 1

            block = MultiScaleBlock(
                dim=embed_dim,
                dim_out=dim_out,
                num_heads=num_heads,
                drop_path=dpr[i],
                q_stride=self.q_stride if i in self.q_pool_blocks else None,
                window_size=window_size,
            )

            embed_dim = dim_out
            self.blocks.append(block)

        self.channel_list = (
            [self.blocks[i].dim_out for i in self.stage_ends[::-1]]
            if return_interm_layers
            else [self.blocks[-1].dim_out]
        )

    def _get_pos_embed(self, hw: Tuple[int, int]) -> torch.Tensor:
        """Generate positional embeddings by interpolating and combining window and background embeddings."""
        h, w = hw
        window_embed = self.pos_embed_window
        pos_embed = F.interpolate(self.pos_embed, size=(h, w), mode="bicubic")
        pos_embed = pos_embed + window_embed.tile([x // y for x, y in zip(pos_embed.shape, window_embed.shape)])
        pos_embed = pos_embed.permute(0, 2, 3, 1)
        return pos_embed

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Perform forward pass through Hiera model, extracting multiscale features from input images.

        Args:
            x (torch.Tensor): Input tensor with shape (B, C, H, W) representing a batch of images.

        Returns:
            (List[torch.Tensor]): List of feature maps at different scales, each with shape (B, C_i, H_i, W_i), where
                C_i is the channel dimension and H_i, W_i are the spatial dimensions at scale i. The list is ordered
                from highest resolution (fine features) to lowest resolution (coarse features) if return_interm_layers
                is True, otherwise contains only the final output.

        Examples:
            >>> model = Hiera(embed_dim=96, num_heads=1, stages=(2, 3, 16, 3))
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> output_features = model(input_tensor)
            >>> for feat in output_features:
            ...     print(feat.shape)
        """
        x = self.patch_embed(x)
        # x: (B, H, W, C)

        # Add pos embed
        x = x + self._get_pos_embed(x.shape[1:3])

        outputs = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if (i == self.stage_ends[-1]) or (i in self.stage_ends and self.return_interm_layers):
                feats = x.permute(0, 3, 1, 2)
                outputs.append(feats)

        return outputs
