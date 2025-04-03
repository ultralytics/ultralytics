# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils import LOGGER, checks
from ultralytics.utils.torch_utils import smart_inference_mode

try:
    import clip
except ImportError:
    checks.check_requirements("git+https://github.com/ultralytics/CLIP.git")
    import clip

try:
    import warnings

    # Suppress 'timm.models.layers is deprecated, please import via timm.layers' warning from mobileclip usage
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        import mobileclip
except ImportError:
    # MobileCLIP repo has an incorrect version of torchvision as dependency
    # Manually install other dependencies first and install mobileclip with "--no-deps" flag
    checks.check_requirements(["open-clip-torch>=2.20.0", "timm>=0.9.5"])
    checks.check_requirements("git+https://github.com/apple/ml-mobileclip.git", cmds="--no-deps")
    import mobileclip


class TextModel(nn.Module):
    """
    Abstract base class for text encoding models.

    This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement
    the tokenize and encode_text methods.

    Methods:
        tokenize: Convert input texts to tokens.
        encode_text: Encode tokenized texts into feature vectors.
    """

    def __init__(self):
        """Initialize the TextModel base class."""
        super().__init__()

    @abstractmethod
    def tokenize(texts):
        """Convert input texts to tokens for model processing."""
        pass

    @abstractmethod
    def encode_text(texts, dtype):
        """Encode tokenized texts into normalized feature vectors."""
        pass


class CLIP(TextModel):
    """
    OpenAI CLIP text encoder implementation.

    This class implements the TextModel interface using OpenAI's CLIP model for text encoding.

    Attributes:
        model (clip.model.CLIP): The loaded CLIP model.
        device (torch.device): Device where the model is loaded.

    Methods:
        tokenize: Convert input texts to CLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    def __init__(self, size, device):
        """
        Initialize the CLIP text encoder.

        Args:
            size (str): Model size identifier (e.g., 'ViT-B/32').
            device (torch.device): Device to load the model on.
        """
        super().__init__()
        self.model = clip.load(size, device=device)[0]
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts):
        """Convert input texts to CLIP tokens."""
        return clip.tokenize(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        """
        Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors.
        """
        txt_feats = self.model.encode_text(texts).to(dtype)
        txt_feats = txt_feats / txt_feats.norm(p=2, dim=-1, keepdim=True)
        return txt_feats


class MobileCLIP(TextModel):
    """
    Apple MobileCLIP text encoder implementation.

    This class implements the TextModel interface using Apple's MobileCLIP model for efficient text encoding.

    Attributes:
        model (mobileclip.model.MobileCLIP): The loaded MobileCLIP model.
        tokenizer (callable): Tokenizer function for processing text inputs.
        device (torch.device): Device where the model is loaded.
        config_size_map (dict): Mapping from size identifiers to model configuration names.

    Methods:
        tokenize: Convert input texts to MobileCLIP tokens.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    config_size_map = {"s0": "s0", "s1": "s1", "s2": "s2", "b": "b", "blt": "b"}

    def __init__(self, size, device):
        """
        Initialize the MobileCLIP text encoder.

        Args:
            size (str): Model size identifier (e.g., 's0', 's1', 's2', 'b', 'blt').
            device (torch.device): Device to load the model on.
        """
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

    def tokenize(self, texts):
        """Convert input texts to MobileCLIP tokens."""
        return self.tokenizer(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        """
        Encode tokenized texts into normalized feature vectors.

        Args:
            texts (torch.Tensor): Tokenized text inputs.
            dtype (torch.dtype): Data type for output features.

        Returns:
            (torch.Tensor): Normalized text feature vectors.
        """
        text_features = self.model.encode_text(texts).to(dtype)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features


def build_text_model(variant, device=None):
    """
    Build a text encoding model based on the specified variant.

    Args:
        variant (str): Model variant in format "base:size" (e.g., "clip:ViT-B/32" or "mobileclip:s0").
        device (torch.device, optional): Device to load the model on.

    Returns:
        (TextModel): Instantiated text encoding model.

    Raises:
        ValueError: If the specified variant is not supported.
    """
    base, size = variant.split(":")
    if base == "clip":
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIP(size, device)
    else:
        raise ValueError(f"Unrecognized base model: '{base}'. Supported base models: 'clip', 'mobileclip'.")
