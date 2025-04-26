# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.utils import checks
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
    # Ultralytics fork preferred since Apple MobileCLIP repo has incorrect version of torchvision
    checks.check_requirements("git+https://github.com/ultralytics/mobileclip.git")
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
        >>> from ultralytics.models.sam import CLIP
        >>> import torch
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> clip_model = CLIP(size="ViT-B/32", device=device)
        >>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> text_features = clip_model.encode_text(tokens)
        >>> print(text_features.shape)
    """

    def __init__(self, size, device):
        """
        Initialize the CLIP text encoder.

        This class implements the TextModel interface using OpenAI's CLIP model for text encoding. It loads
        a pre-trained CLIP model of the specified size and prepares it for text encoding tasks.

        Args:
            size (str): Model size identifier (e.g., 'ViT-B/32').
            device (torch.device): Device to load the model on.

        Examples:
            >>> import torch
            >>> from ultralytics.models.sam.modules.clip import CLIP
            >>> clip_model = CLIP("ViT-B/32", device=torch.device("cuda:0"))
            >>> text_features = clip_model.encode_text(["a photo of a cat", "a photo of a dog"])
        """
        super().__init__()
        self.model = clip.load(size, device=device)[0]
        self.to(device)
        self.device = device
        self.eval()

    def tokenize(self, texts):
        """
        Convert input texts to CLIP tokens.

        Args:
            texts (str | List[str]): Input text or list of texts to tokenize.

        Returns:
            (torch.Tensor): Tokenized text tensor with shape (batch_size, context_length) ready for model processing.

        Examples:
            >>> model = CLIP("ViT-B/32", device="cpu")
            >>> tokens = model.tokenize("a photo of a cat")
            >>> print(tokens.shape)  # torch.Size([1, 77])
        """
        return clip.tokenize(texts).to(self.device)

    @smart_inference_mode()
    def encode_text(self, texts, dtype=torch.float32):
        """
        Encode tokenized texts into normalized feature vectors.

        This method processes tokenized text inputs through the CLIP model to generate feature vectors, which are then
        normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

        Args:
            texts (torch.Tensor): Tokenized text inputs, typically created using the tokenize() method.
            dtype (torch.dtype, optional): Data type for output features. Default is torch.float32.

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


class MobileCLIP(TextModel):
    """
    Implement Apple's MobileCLIP text encoder for efficient text encoding.

    This class implements the TextModel interface using Apple's MobileCLIP model, providing efficient text encoding
    capabilities for vision-language tasks.

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

    def __init__(self, size, device):
        """
        Initialize the MobileCLIP text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model for efficient text encoding.

        Args:
            size (str): Model size identifier (e.g., 's0', 's1', 's2', 'b', 'blt').
            device (torch.device): Device to load the model on.

        Examples:
            >>> from ultralytics.nn.modules import MobileCLIP
            >>> import torch
            >>> model = MobileCLIP("s0", device=torch.device("cpu"))
            >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
            >>> features = model.encode_text(tokens)
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
    def encode_text(self, texts, dtype=torch.float32):
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


def build_text_model(variant, device=None):
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
        return CLIP(size, device)
    elif base == "mobileclip":
        return MobileCLIP(size, device)
    else:
        raise ValueError(f"Unrecognized base model: '{base}'. Supported base models: 'clip', 'mobileclip'.")
