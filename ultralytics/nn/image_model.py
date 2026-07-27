# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import abstractmethod

import torch
from PIL import Image
from torch import nn

from ultralytics.utils.torch_utils import smart_inference_mode


class ImageModel(nn.Module):
    """Abstract base class for image encoding models.

    Define the interface for image encoding models used in vision-language tasks. Subclasses must implement the
    encode_image method to provide image encoding functionality.

    Methods:
        encode_image: Encode image inputs into normalized feature vectors.
    """

    def __init__(self):
        """Initialize the ImageModel base class."""
        super().__init__()

    @abstractmethod
    def encode_image(self, image, dtype):
        """Encode image inputs into normalized feature vectors."""


class MobileCLIPImageTS(ImageModel):
    """Load a TorchScript traced MobileCLIP2 image encoder (https://arxiv.org/abs/2508.20691).

    Mirror of MobileCLIPTS (text_model.py) for image encoding. Load a TorchScript image encoder via
    attempt_download_asset and call self.encoder(image) to get L2-normalized embeddings. Normalization is built into the
    TorchScript model, matching the MobileCLIPTS pattern.

    Attributes:
        encoder (torch.jit.ScriptModule): The loaded TorchScript MobileCLIP image encoder.
        image_preprocess (transforms.Compose): Preprocessing pipeline for PIL images.
        device (torch.device): Device where the model is loaded.

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> image_encoder = MobileCLIPImageTS(device=device, weight="mobileclip2_s4_image.ts")
        >>> features = image_encoder.encode_image(torch.randn(1, 3, 256, 256).to(device))
    """

    def __init__(
        self,
        device: torch.device,
        weight: str = "mobileclip2_s4_image.ts",
        imgsz: int = 256,
        interpolation: str = "bilinear",
    ):
        """Initialize the MobileCLIP TorchScript image encoder.

        Args:
            device (torch.device): Device to load the model on.
            weight (str): Path to the TorchScript model weights.
            imgsz (int): Input image size for preprocessing.
            interpolation (str): Resize interpolation mode ('bilinear' for S4, 'bicubic' for L/14).
        """
        super().__init__()
        from torchvision import transforms

        from ultralytics.utils.downloads import attempt_download_asset

        interp = transforms.InterpolationMode.BICUBIC if interpolation == "bicubic" else transforms.InterpolationMode.BILINEAR
        self.encoder = torch.jit.load(attempt_download_asset(weight), map_location=device)
        self.device = device
        self.image_preprocess = transforms.Compose(
            [
                transforms.Resize(imgsz, interpolation=interp, antialias=True),
                transforms.CenterCrop(imgsz),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
                ),
            ]
        )

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Encode images into normalized feature vectors.

        Args:
            image (PIL.Image | torch.Tensor): Image input. PIL images are auto-preprocessed with CLIP-standard ImageNet
                normalization (not Ultralytics identity normalization).
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized image feature vectors with L2 normalization applied.

        Examples:
            >>> model = MobileCLIPImageTS(device=torch.device("cpu"))
            >>> features = model.encode_image(torch.randn(1, 3, 256, 256))
            >>> features.shape
            torch.Size([1, 768])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        # NOTE: no need to do normalization here as it's embedded in the torchscript model
        return self.encoder(image).to(dtype)


def build_image_model(variant: str, device: torch.device = None) -> ImageModel:
    """Build an image encoding model based on the specified variant.

    Args:
        variant (str): Model variant in format "mobileclip2:size" (e.g., "mobileclip2:s4").
        device (torch.device, optional): Device to load the model on.

    Returns:
        (ImageModel): Instantiated image encoding model.

    Examples:
        >>> model = build_image_model("mobileclip2:s4", device=torch.device("cpu"))
    """
    base, size = variant.split(":")
    if base == "mobileclip2":
        configs = {
            "s4": ("mobileclip2_s4_image.ts", 256, "bilinear"),
            "l14": ("mobileclip2_l14_image.ts", 224, "bicubic"),
        }
        if size not in configs:
            raise ValueError(f"Unrecognized mobileclip2 variant '{size}'. Supported: {list(configs)}")
        weight, imgsz, interp = configs[size]
        return MobileCLIPImageTS(device, weight=weight, imgsz=imgsz, interpolation=interp)
    raise ValueError(f"Unrecognized image model '{base}'. Supported: 'mobileclip2'.")
