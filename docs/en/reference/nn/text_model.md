---
description: Documentation for text encoding models in Ultralytics YOLOE, supporting both OpenAI CLIP and Apple MobileCLIP implementations for vision-language tasks.
keywords: YOLOE, text encoding, CLIP, MobileCLIP, TextModel, vision-language models, embeddings, Ultralytics, deep learning
---

# Reference for `ultralytics/nn/text_model.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-class">Classes</span>"

        - [`TextModel`](#ultralytics.nn.text_model.TextModel)
        - [`CLIP`](#ultralytics.nn.text_model.CLIP)
        - [`MobileCLIP`](#ultralytics.nn.text_model.MobileCLIP)
        - [`MobileCLIPTS`](#ultralytics.nn.text_model.MobileCLIPTS)

    === "<span class="doc-kind doc-kind-method">Methods</span>"

        - [`TextModel.tokenize`](#ultralytics.nn.text_model.TextModel.tokenize)
        - [`TextModel.encode_text`](#ultralytics.nn.text_model.TextModel.encode_text)
        - [`CLIP.tokenize`](#ultralytics.nn.text_model.CLIP.tokenize)
        - [`CLIP.encode_text`](#ultralytics.nn.text_model.CLIP.encode_text)
        - [`CLIP.encode_image`](#ultralytics.nn.text_model.CLIP.encode_image)
        - [`MobileCLIP.tokenize`](#ultralytics.nn.text_model.MobileCLIP.tokenize)
        - [`MobileCLIP.encode_text`](#ultralytics.nn.text_model.MobileCLIP.encode_text)
        - [`MobileCLIPTS.tokenize`](#ultralytics.nn.text_model.MobileCLIPTS.tokenize)
        - [`MobileCLIPTS.encode_text`](#ultralytics.nn.text_model.MobileCLIPTS.encode_text)

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`build_text_model`](#ultralytics.nn.text_model.build_text_model)


## Class `ultralytics.nn.text_model.TextModel` {#ultralytics.nn.text\_model.TextModel}

```python
TextModel(self)
```

**Bases:** `nn.Module`

Abstract base class for text encoding models.

This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement the tokenize and encode_text methods to provide text tokenization and encoding functionality.

**Methods**

| Name | Description |
| --- | --- |
| [`encode_text`](#ultralytics.nn.text_model.TextModel.encode_text) | Encode tokenized texts into normalized feature vectors. |
| [`tokenize`](#ultralytics.nn.text_model.TextModel.tokenize) | Convert input texts to tokens for model processing. |

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L22-L45"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class TextModel(nn.Module):
    """Abstract base class for text encoding models.

    This class defines the interface for text encoding models used in vision-language tasks. Subclasses must implement
    the tokenize and encode_text methods to provide text tokenization and encoding functionality.

    Methods:
        tokenize: Convert input texts to tokens for model processing.
        encode_text: Encode tokenized texts into normalized feature vectors.
    """

    def __init__(self):
        """Initialize the TextModel base class."""
        super().__init__()
```
</details>

<br>

### Method `ultralytics.nn.text_model.TextModel.encode_text` {#ultralytics.nn.text\_model.TextModel.encode\_text}

```python
def encode_text(self, texts, dtype)
```

Encode tokenized texts into normalized feature vectors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` |  |  | *required* |
| `dtype` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L43-L45"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@abstractmethod
def encode_text(self, texts, dtype):
    """Encode tokenized texts into normalized feature vectors."""
    pass
```
</details>

<br>

### Method `ultralytics.nn.text_model.TextModel.tokenize` {#ultralytics.nn.text\_model.TextModel.tokenize}

```python
def tokenize(self, texts)
```

Convert input texts to tokens for model processing.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L38-L40"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@abstractmethod
def tokenize(self, texts):
    """Convert input texts to tokens for model processing."""
    pass
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.text_model.CLIP` {#ultralytics.nn.text\_model.CLIP}

```python
CLIP(self, size: str, device: torch.device) -> None
```

**Bases:** `TextModel`

Implements OpenAI's CLIP (Contrastive Language-Image Pre-training) text encoder.

This class provides a text encoder based on OpenAI's CLIP model, which can convert text into feature vectors that are aligned with corresponding image features in a shared embedding space.

This class implements the TextModel interface using OpenAI's CLIP model for text encoding. It loads a pre-trained CLIP model of the specified size and prepares it for text encoding tasks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `size` | `str` | Model size identifier (e.g., 'ViT-B/32'). | *required* |
| `device` | `torch.device` | Device to load the model on. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `clip.model.CLIP` | The loaded CLIP model. |
| `image_preprocess` | `callable` | Preprocessing transform for images. |
| `device` | `torch.device` | Device where the model is loaded. |

**Methods**

| Name | Description |
| --- | --- |
| [`encode_image`](#ultralytics.nn.text_model.CLIP.encode_image) | Encode images into normalized feature vectors. |
| [`encode_text`](#ultralytics.nn.text_model.CLIP.encode_text) | Encode tokenized texts into normalized feature vectors. |
| [`tokenize`](#ultralytics.nn.text_model.CLIP.tokenize) | Convert input texts to CLIP tokens. |

**Examples**

```python
>>> import torch
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> clip_model = CLIP(size="ViT-B/32", device=device)
>>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
>>> text_features = clip_model.encode_text(tokens)
>>> print(text_features.shape)
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L48-L162"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class CLIP(TextModel):
    """Implements OpenAI's CLIP (Contrastive Language-Image Pre-training) text encoder.

    This class provides a text encoder based on OpenAI's CLIP model, which can convert text into feature vectors that
    are aligned with corresponding image features in a shared embedding space.

    Attributes:
        model (clip.model.CLIP): The loaded CLIP model.
        image_preprocess (callable): Preprocessing transform for images.
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
        """Initialize the CLIP text encoder.

        This class implements the TextModel interface using OpenAI's CLIP model for text encoding. It loads a
        pre-trained CLIP model of the specified size and prepares it for text encoding tasks.

        Args:
            size (str): Model size identifier (e.g., 'ViT-B/32').
            device (torch.device): Device to load the model on.
        """
        super().__init__()
        self.model, self.image_preprocess = clip.load(size, device=device)
        self.to(device)
        self.device = device
        self.eval()
```
</details>

<br>

### Method `ultralytics.nn.text_model.CLIP.encode_image` {#ultralytics.nn.text\_model.CLIP.encode\_image}

```python
def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor
```

Encode images into normalized feature vectors.

This method processes image inputs through the CLIP model to generate feature vectors, which are then normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `image` | `PIL.Image | torch.Tensor` | Image input as a PIL Image or preprocessed tensor. If a PIL Image is<br>    provided, it will be converted to a tensor using the model's image preprocessing function. | *required* |
| `dtype` | `torch.dtype, optional` | Data type for output features. | `torch.float32` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Normalized image feature vectors with unit length (L2 norm = 1). |

**Examples**

```python
>>> from ultralytics.nn.text_model import CLIP
>>> from PIL import Image
>>> clip_model = CLIP("ViT-B/32", device="cuda")
>>> image = Image.open("path/to/image.jpg")
>>> image_tensor = clip_model.image_preprocess(image).unsqueeze(0).to("cuda")
>>> features = clip_model.encode_image(image_tensor)
>>> features.shape
torch.Size([1, 512])
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L134-L162"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def encode_image(self, image: Image.Image | torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Encode images into normalized feature vectors.

    This method processes image inputs through the CLIP model to generate feature vectors, which are then
    normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

    Args:
        image (PIL.Image | torch.Tensor): Image input as a PIL Image or preprocessed tensor. If a PIL Image is
            provided, it will be converted to a tensor using the model's image preprocessing function.
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
```
</details>

<br>

### Method `ultralytics.nn.text_model.CLIP.encode_text` {#ultralytics.nn.text\_model.CLIP.encode\_text}

```python
def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor
```

Encode tokenized texts into normalized feature vectors.

This method processes tokenized text inputs through the CLIP model to generate feature vectors, which are then normalized to unit length. These normalized vectors can be used for text-image similarity comparisons.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` | `torch.Tensor` | Tokenized text inputs, typically created using the tokenize() method. | *required* |
| `dtype` | `torch.dtype, optional` | Data type for output features. | `torch.float32` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Normalized text feature vectors with unit length (L2 norm = 1). |

**Examples**

```python
>>> clip_model = CLIP("ViT-B/32", device="cuda")
>>> tokens = clip_model.tokenize(["a photo of a cat", "a photo of a dog"])
>>> features = clip_model.encode_text(tokens)
>>> features.shape
torch.Size([2, 512])
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L109-L131"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Encode tokenized texts into normalized feature vectors.

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
```
</details>

<br>

### Method `ultralytics.nn.text_model.CLIP.tokenize` {#ultralytics.nn.text\_model.CLIP.tokenize}

```python
def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor
```

Convert input texts to CLIP tokens.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` | `str | list[str]` | Input text or list of texts to tokenize. | *required* |
| `truncate` | `bool, optional` | Whether to trim texts that exceed CLIP's context length. Defaults to True to<br>    avoid RuntimeError from overly long inputs while still allowing explicit opt-out. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Tokenized text tensor with shape (batch_size, context_length) ready for model processing. |

**Examples**

```python
>>> model = CLIP("ViT-B/32", device="cpu")
>>> tokens = model.tokenize("a photo of a cat")
>>> print(tokens.shape)  # torch.Size([1, 77])
>>> strict_tokens = model.tokenize("a photo of a cat", truncate=False)  # Enforce strict length checks
>>> print(strict_tokens.shape)  # Same shape/content as tokens since prompt less than 77 tokens
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L88-L106"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def tokenize(self, texts: str | list[str], truncate: bool = True) -> torch.Tensor:
    """Convert input texts to CLIP tokens.

    Args:
        texts (str | list[str]): Input text or list of texts to tokenize.
        truncate (bool, optional): Whether to trim texts that exceed CLIP's context length. Defaults to True to
            avoid RuntimeError from overly long inputs while still allowing explicit opt-out.

    Returns:
        (torch.Tensor): Tokenized text tensor with shape (batch_size, context_length) ready for model processing.

    Examples:
        >>> model = CLIP("ViT-B/32", device="cpu")
        >>> tokens = model.tokenize("a photo of a cat")
        >>> print(tokens.shape)  # torch.Size([1, 77])
        >>> strict_tokens = model.tokenize("a photo of a cat", truncate=False)  # Enforce strict length checks
        >>> print(strict_tokens.shape)  # Same shape/content as tokens since prompt less than 77 tokens
    """
    return clip.tokenize(texts, truncate=truncate).to(self.device)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.text_model.MobileCLIP` {#ultralytics.nn.text\_model.MobileCLIP}

```python
MobileCLIP(self, size: str, device: torch.device) -> None
```

**Bases:** `TextModel`

Implement Apple's MobileCLIP text encoder for efficient text encoding.

This class implements the TextModel interface using Apple's MobileCLIP model, providing efficient text encoding capabilities for vision-language tasks with reduced computational requirements compared to standard CLIP models.

This class implements the TextModel interface using Apple's MobileCLIP model for efficient text encoding.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `size` | `str` | Model size identifier (e.g., 's0', 's1', 's2', 'b', 'blt'). | *required* |
| `device` | `torch.device` | Device to load the model on. | *required* |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `model` | `mobileclip.model.MobileCLIP` | The loaded MobileCLIP model. |
| `tokenizer` | `callable` | Tokenizer function for processing text inputs. |
| `device` | `torch.device` | Device where the model is loaded. |
| `config_size_map` | `dict` | Mapping from size identifiers to model configuration names. |

**Methods**

| Name | Description |
| --- | --- |
| [`encode_text`](#ultralytics.nn.text_model.MobileCLIP.encode_text) | Encode tokenized texts into normalized feature vectors. |
| [`tokenize`](#ultralytics.nn.text_model.MobileCLIP.tokenize) | Convert input texts to MobileCLIP tokens. |

**Examples**

```python
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> text_encoder = MobileCLIP(size="s0", device=device)
>>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
>>> features = text_encoder.encode_text(tokens)
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L165-L254"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class MobileCLIP(TextModel):
    """Implement Apple's MobileCLIP text encoder for efficient text encoding.

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
        """Initialize the MobileCLIP text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model for efficient text encoding.

        Args:
            size (str): Model size identifier (e.g., 's0', 's1', 's2', 'b', 'blt').
            device (torch.device): Device to load the model on.
        """
        try:
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
```
</details>

<br>

### Method `ultralytics.nn.text_model.MobileCLIP.encode_text` {#ultralytics.nn.text\_model.MobileCLIP.encode\_text}

```python
def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor
```

Encode tokenized texts into normalized feature vectors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` | `torch.Tensor` | Tokenized text inputs. | *required* |
| `dtype` | `torch.dtype, optional` | Data type for output features. | `torch.float32` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Normalized text feature vectors with L2 normalization applied. |

**Examples**

```python
>>> model = MobileCLIP("s0", device="cpu")
>>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
>>> features = model.encode_text(tokens)
>>> features.shape
torch.Size([2, 512])  # Actual dimension depends on model size
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L235-L254"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Encode tokenized texts into normalized feature vectors.

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
```
</details>

<br>

### Method `ultralytics.nn.text_model.MobileCLIP.tokenize` {#ultralytics.nn.text\_model.MobileCLIP.tokenize}

```python
def tokenize(self, texts: list[str]) -> torch.Tensor
```

Convert input texts to MobileCLIP tokens.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` | `list[str]` | List of text strings to tokenize. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Tokenized text inputs with shape (batch_size, sequence_length). |

**Examples**

```python
>>> model = MobileCLIP("s0", "cpu")
>>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L219-L232"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def tokenize(self, texts: list[str]) -> torch.Tensor:
    """Convert input texts to MobileCLIP tokens.

    Args:
        texts (list[str]): List of text strings to tokenize.

    Returns:
        (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

    Examples:
        >>> model = MobileCLIP("s0", "cpu")
        >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
    """
    return self.tokenizer(texts).to(self.device)
```
</details>


<br><br><hr><br>

## Class `ultralytics.nn.text_model.MobileCLIPTS` {#ultralytics.nn.text\_model.MobileCLIPTS}

```python
MobileCLIPTS(self, device: torch.device, weight: str = "mobileclip_blt.ts")
```

**Bases:** `TextModel`

Load a TorchScript traced version of MobileCLIP.

This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format, providing efficient text encoding capabilities for vision-language tasks with optimized inference performance.

This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for efficient text encoding with optimized inference performance.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `device` | `torch.device` | Device to load the model on. | *required* |
| `weight` | `str` | Path to the TorchScript model weights. | `"mobileclip_blt.ts"` |

**Attributes**

| Name | Type | Description |
| --- | --- | --- |
| `encoder` | `torch.jit.ScriptModule` | The loaded TorchScript MobileCLIP text encoder. |
| `tokenizer` | `callable` | Tokenizer function for processing text inputs. |
| `device` | `torch.device` | Device where the model is loaded. |

**Methods**

| Name | Description |
| --- | --- |
| [`encode_text`](#ultralytics.nn.text_model.MobileCLIPTS.encode_text) | Encode tokenized texts into normalized feature vectors. |
| [`tokenize`](#ultralytics.nn.text_model.MobileCLIPTS.tokenize) | Convert input texts to MobileCLIP tokens. |

**Examples**

```python
>>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
>>> text_encoder = MobileCLIPTS(device=device)
>>> tokens = text_encoder.tokenize(["a photo of a cat", "a photo of a dog"])
>>> features = text_encoder.encode_text(tokens)
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L257-L335"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
class MobileCLIPTS(TextModel):
    """Load a TorchScript traced version of MobileCLIP.

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

    def __init__(self, device: torch.device, weight: str = "mobileclip_blt.ts"):
        """Initialize the MobileCLIP TorchScript text encoder.

        This class implements the TextModel interface using Apple's MobileCLIP model in TorchScript format for efficient
        text encoding with optimized inference performance.

        Args:
            device (torch.device): Device to load the model on.
            weight (str): Path to the TorchScript model weights.
        """
        super().__init__()
        from ultralytics.utils.downloads import attempt_download_asset

        self.encoder = torch.jit.load(attempt_download_asset(weight), map_location=device)
        self.tokenizer = clip.clip.tokenize
        self.device = device
```
</details>

<br>

### Method `ultralytics.nn.text_model.MobileCLIPTS.encode_text` {#ultralytics.nn.text\_model.MobileCLIPTS.encode\_text}

```python
def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor
```

Encode tokenized texts into normalized feature vectors.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` | `torch.Tensor` | Tokenized text inputs. | *required* |
| `dtype` | `torch.dtype, optional` | Data type for output features. | `torch.float32` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Normalized text feature vectors with L2 normalization applied. |

**Examples**

```python
>>> model = MobileCLIPTS(device="cpu")
>>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
>>> features = model.encode_text(tokens)
>>> features.shape
torch.Size([2, 512])  # Actual dimension depends on model size
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L317-L335"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
@smart_inference_mode()
def encode_text(self, texts: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Encode tokenized texts into normalized feature vectors.

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
```
</details>

<br>

### Method `ultralytics.nn.text_model.MobileCLIPTS.tokenize` {#ultralytics.nn.text\_model.MobileCLIPTS.tokenize}

```python
def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor
```

Convert input texts to MobileCLIP tokens.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `texts` | `list[str]` | List of text strings to tokenize. | *required* |
| `truncate` | `bool, optional` | Whether to trim texts that exceed the tokenizer context length. Defaults to True,<br>    matching CLIP's behavior to prevent runtime failures on long captions. | `True` |

**Returns**

| Type | Description |
| --- | --- |
| `torch.Tensor` | Tokenized text inputs with shape (batch_size, sequence_length). |

**Examples**

```python
>>> model = MobileCLIPTS(device=torch.device("cpu"))
>>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
>>> strict_tokens = model.tokenize(
...     ["a very long caption"], truncate=False
... )  # RuntimeError if exceeds 77-token
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L296-L314"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def tokenize(self, texts: list[str], truncate: bool = True) -> torch.Tensor:
    """Convert input texts to MobileCLIP tokens.

    Args:
        texts (list[str]): List of text strings to tokenize.
        truncate (bool, optional): Whether to trim texts that exceed the tokenizer context length. Defaults to True,
            matching CLIP's behavior to prevent runtime failures on long captions.

    Returns:
        (torch.Tensor): Tokenized text inputs with shape (batch_size, sequence_length).

    Examples:
        >>> model = MobileCLIPTS(device=torch.device("cpu"))
        >>> tokens = model.tokenize(["a photo of a cat", "a photo of a dog"])
        >>> strict_tokens = model.tokenize(
        ...     ["a very long caption"], truncate=False
        ... )  # RuntimeError if exceeds 77-token
    """
    return self.tokenizer(texts, truncate=truncate).to(self.device)
```
</details>


<br><br><hr><br>

## Function `ultralytics.nn.text_model.build_text_model` {#ultralytics.nn.text\_model.build\_text\_model}

```python
def build_text_model(variant: str, device: torch.device = None) -> TextModel
```

Build a text encoding model based on the specified variant.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `variant` | `str` | Model variant in format "base:size" (e.g., "clip:ViT-B/32" or "mobileclip:s0"). | *required* |
| `device` | `torch.device, optional` | Device to load the model on. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `TextModel` | Instantiated text encoding model. |

**Examples**

```python
>>> model = build_text_model("clip:ViT-B/32", device=torch.device("cuda"))
>>> model = build_text_model("mobileclip:s0", device=torch.device("cpu"))
```

<details>
<summary>Source code in <code>ultralytics/nn/text_model.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/text_model.py#L338-L360"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def build_text_model(variant: str, device: torch.device = None) -> TextModel:
    """Build a text encoding model based on the specified variant.

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
        return MobileCLIPTS(device)
    elif base == "mobileclip2":
        return MobileCLIPTS(device, weight="mobileclip2_b.ts")
    else:
        raise ValueError(f"Unrecognized base model '{base}'. Supported models are 'clip', 'mobileclip', 'mobileclip2'.")
```
</details>

<br><br>
