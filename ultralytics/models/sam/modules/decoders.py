# Ultralytics YOLO 🚀, AGPL-3.0 license


import torch
from torch import nn
from torch.nn import functional as F

from ultralytics.nn.modules import LayerNorm2d


class MaskDecoder(nn.Module):
    """
    Decoder module for generating masks and their associated quality scores, using a transformer architecture to predict
    masks given image and prompt embeddings.

    Attributes:
        transformer_dim (int): Channel dimension for the transformer module.
        transformer (nn.Module): The transformer module used for mask prediction.
        num_multimask_outputs (int): Number of masks to predict for disambiguating masks.
        iou_token (nn.Embedding): Embedding for the IoU token.
        num_mask_tokens (int): Number of mask tokens.
        mask_tokens (nn.Embedding): Embedding for the mask tokens.
        output_upscaling (nn.Sequential): Neural network sequence for upscaling the output.
        output_hypernetworks_mlps (nn.ModuleList): Hypernetwork MLPs for generating masks.
        iou_prediction_head (nn.Module): MLP for predicting mask quality.
    """

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
        num_multimask_outputs: int = 3,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = 3,
        iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a transformer architecture.

        Args:
            transformer_dim (int): the channel dimension of the transformer module
            transformer (nn.Module): the transformer used to predict masks
            num_multimask_outputs (int): the number of masks to predict when disambiguating masks
            activation (nn.Module): the type of activation to use when upscaling masks
            iou_head_depth (int): the depth of the MLP used to predict mask quality
            iou_head_hidden_dim (int): the hidden dimension of the MLP used to predict mask quality
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
        """
        Predict masks given image and prompt embeddings.

        Args:
            image_embeddings (torch.Tensor): the embeddings from the image encoder
            image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
            sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
            dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
            multimask_output (bool): Whether to return multiple masks or a single mask.

        Returns:
            torch.Tensor: batched predicted masks
            torch.Tensor: batched predictions of mask quality
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

        # Prepare output
        return masks, iou_pred

    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Predicts masks.

        See 'forward' for more details.
        """
        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
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


class MLP(nn.Module):
    """
    MLP (Multi-Layer Perceptron) model lightly adapted from
    https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        """
        Initializes the MLP (Multi-Layer Perceptron) model.

        Args:
            input_dim (int): The dimensionality of the input features.
            hidden_dim (int): The dimensionality of the hidden layers.
            output_dim (int): The dimensionality of the output layer.
            num_layers (int): The number of hidden layers.
            sigmoid_output (bool, optional): Whether to apply a sigmoid activation to the output layer. Defaults to False.
        """
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        """Executes feedforward within the neural network module and applies activation."""
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x
