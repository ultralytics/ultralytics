# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image

from ultralytics.utils import checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip


class TextModel(nn.Module):
    """
    Abstract base class for text encoding models.

    This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement
    the tokenize and encode_text methods to provide text tokenization and encoding functionality.

    Methods:
        tokenize: Convert input texts to tokens for model processing.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    def __init__(self):
        """Initialize the TextModel base class."""
        super().__init__()

    @abstractmethod
    def tokenize(self, texts):
        """Convert input texts to tokens for model processing."""
        pass

    @abstractmethod
    def encode_text(self, texts, dtype):
        """Encode tokenized texts into normalized feature vectors."""
        pass


class CLIP(TextModel):
    """
    Implements OpenAI's CLIP (Contrastive Language-Image Pre-training) text encoder.

    This class provides a text encoder based on OpenAI's CLIP model, which can convert text into feature vectors
    that are aligned with corresponding image features in a shared embedding space.

    Attributes:
        model (clip.model.CLIP): The loaded CLIP model.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to CLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> clip_model = CLIP(size="ViT-B/32", device=device)
        >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> text_features = clip_model.encode_text(tokens)
        >>> print(text_features.shape)
    """

    def __init__(self, size: str, device: torch.device) -> None:
        """
        Initialize the CLIP text encoder.

        This class implements the TextModel interface using OpenAI's CLIP model for text encoding. It loads
        a pre-trained CLIP model of the specified size and prepares it for text encoding tasks.

        Args:
            size (str): Model size identifier (e.g., 'ViT-B/32').
            device (torch.device): Device to load the model on.

        Examples:
            >>> import torch
            >>> clip_model = CLIP("ViT-B/32", device=torch.device("cuda:0"))
            >>> text_features = clip_model.encode_text(["a photo of a cat", "a photo of a dog"])
        """
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: str | list[str]) -> torch.Tensor:
        """
        Convert input texts to CLIP tokens.

        Args:
            texts (str | list[str]): Input text or list of texts to tokenize.

        Returns:
            (torch.Tensor): Tokenized text tensor with shape (batch_size, context_length) ready for model processing.

        Examples:
            >>> model = CLIP("ViT-B/32", device="cpu")
            >>> tokens = model.tokenize("a photo of a cat")
            >>> print(tokens.shape)  # torch.Size([1, 77])
        """
        return clip.tokenize(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Encode tokenized texts into normalized feature vectors.

        This method processes tokenized text inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            texts (torch.Tensor): Tokenized text inputs, typically created using the tokenize() method.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with unit length (L2 norm = 1).

        Examples:
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = clip_model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])
        """
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats

    @smart_inference_mode()
    def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Encode preprocessed images into normalized feature vectors.

        This method processes preprocessed image inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            image (PIL.Image | torch.Tensor): Preprocessed image input. If a PIL Image is provided, it will be
                converted to a tensor using the model's image preprocessing function.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized image feature vectors with unit length (L2 norm = 1).

        Examples:
            >>> from ultralytics.nn.text_model import CLIP
            >>> from PIL import Image
            >>> clip_model = CLIP("ViT-B/32", device="cuda")
            >>> image = Image.open("path/to/image.jpg")
            >>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
            >>> features = clip_model.encode_image(image_tensor)
            >>> features.shape
            torch.Size([1, 512])
        """
        if isinstance(image, Image.Image):
            image = self.image_preprocess(image).unsqueeze(0).to(self.device)
        img_feats = self.model.encode_image(image).to(dtype)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)
        return img_feats


class MobileCLIP(TextModel):
    """
    Implement Apple's MobileCLIP text encoder for efficient text encoding.

    This class implements the TextModel interface using Apple's MobileCLIP model, providing efficient text encoding
    capabilities for vision-language tasks with reduced computational requirements compared to standard CLIP models.

    Attributes:
        model (mobileclip.model.MobileCLIP): The loaded MobileCLIP model.
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.
        config_size_map (dict): Mapping from size identifiers to model configuration names.

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIP(size="s0", device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size: str, device: torch.device) -> None:
        """
        Initialize the MobileCLIP text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model for efficient text encoding.

        Args:
            size (str): Model size identifier (e.g., 's0', 's1', 's2', 'b', 'blt').
            device (torch.device): Device to load the model on.

        Examples:
            >>> import torch
            >>> model = MobileCLIP("s0", device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
        """
        try:
            import warnings

            # Suppress 'timm.models.layers is deprecated, please import via timm.layers' warning from mobileclip usage
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import mobileclip
        except ImportError:
            # Ultralytics fork preferred since Apple MobileCLIP repo has incorrect version of torchvision
            checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
            import mobileclip

        super().__init__()
        config = self.config_size_map[size]
        file = f"mobileclip_{size}.pt"
        if not Path(file).is_file():
            from ultralytics import download

            download(f"https://docs-assets.developer.apple.com/ml-research/datasets/mobileclip/{file}")
        self.model = mobileclip.create_model_and_transforms(f"mobileclip_{config}", pretrained=file, device=device)[0]
        self.tokenizer = mobileclip.get_tokenizer(f"mobileclip_{config}")
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """
        Convert input texts to MobileCLIP tokens.

        Args:
            texts (list[str]): List of text strings to tokenize.

        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

        Examples:
            >>> model = MobileCLIP("s0", "cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        """
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.

        Examples:
            >>> model = MobileCLIP("s0", device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # Actual dimension depends on model size
        """
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


class MobileCLIPTS(TextModel):
    """
    Load a TorchScript traced version of MobileCLIP.

    This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format, providing
    efficient text encoding capabilities for vision-language tasks with optimized inference performance.

    Attributes:
        encoder (torch.jit.ScriptModule): The loaded TorchScript MobileCLIP text encoder.
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.

    Examples:
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> text_encoder = MobileCLIPTS(device=device)
        >>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> features = text_encoder.encode_text(tokens)
    """

    def __init__(self, clip_weight_name="mobileclip_blt.ts", device: torch.device="cuda"):
        """
        Initialize the MobileCLIP TorchScript text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for
        efficient text encoding with optimized inference performance.

        Args:
            device (torch.device): Device to load the model on.

        Examples:
            >>> model = MobileCLIPTS(device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
        """
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset(clip_weight_name), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device

    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """
        Convert input texts to MobileCLIP tokens.

        Args:
            texts (list[str]): List of text strings to tokenize.

        Returns:
            (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

        Examples:
            >>> model = MobileCLIPTS("cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        """
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype, optional): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors with L2 normalization applied.

        Examples:
            >>> model = MobileCLIPTS(device="cpu")
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
            >>> features.shape
            torch.Size([2, 512])  # Actual dimension depends on model size
        """
        # NOTE: no need to do normalization here as it's embedded in the torchscript model
        return self.encoder(texts).to(dtype)

import os,sys
import open_clip
import copy


class OpenCLIP(TextModel):

    def __init__(self,device,  pretrained_path="mobileclip2_b.ts"):
        """
        Initializes the model, tokenizer, and preprocessing steps.
        """
        super().__init__()

        import os
        pretrained_path= os.path.join("./", pretrained_path)
        if not os.path.exists(pretrained_path):
            raise Exception("Please download the mobileclip2_b.pt from")

        pretrained_name= os.path.basename(pretrained_path)
        model_name={"mobileclip2_b.pt":"MobileCLIP2-B"}[pretrained_name]

        model_kwargs = {}
        if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
            model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained_path, **model_kwargs
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # Set up model for inference
        self.model.eval()
        self.model = self.reparameterize_model(self.model)

        # Move model to GPU if available
        self.device = device
        self.model.to(self.device)
        print(f"Model '{model_name}' loaded on {self.device}.")


    def tokenize(self, texts: list[str]) -> torch.Tensor:
        """
        Tokenizes input texts into model-compatible format.

        Args:
            texts (list[str]): List of text strings to tokenize.

        Returns:
            (torch.Tensor): Tokenized text inputs.
        """


        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:

        text_features = self.model.encode_text(texts)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def reparameterize_model(self,model: torch.nn.Module) -> nn.Module:
        """Method returns a model where a multi-branched structure
            used in training is re-parameterized into a single branch
            for inference.

        Args:
            model: MobileOne model in train mode.

        Returns:
            MobileOne model in inference mode.
        """
        # Avoid editing original graph
        model = copy.deepcopy(model)
        for module in model.modules():
            if hasattr(module, "reparameterize"):
                module.reparameterize()
        return model
    
def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    """
    Build a text encoding model based on the specified variant.

    Args:
        variant (str): Model variant in format "base:size" (e.g., "clip:ViT-B/32" or "mobileclip:s0").
        device (torch.device, optional): Device to load the model on.

    Returns:
        (TextModel): Instantiated text encoding model.

    Examples:
        >>> model = build_text_model("clip:ViT-B/32", device=torch.device("cuda"))
        >>> model = build_text_model("mobileclip:s0", device=torch.device("cpu"))
    """
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device=device)
    elif base == "mobileclip":
        return MobileCLIPTS(clip_weight_name="mobileclip_blt.ts",device=device)
    elif base == "mobileclip2":
        return MobileCLIPTS(clip_weight_name="mobileclip2_b.ts",device=device)
    else:
        raise ValueError(f"Unrecognized base model: '{base}'. Supported base models: 'clip', 'mobileclip', 'mobileclip2'.")