---
comments: true
description: Explore SAM 3 LiteText, a lightweight drop-in replacement for the SAM 3 text encoder using MobileCLIP student distillation. Up to 88% fewer text-encoder parameters with comparable segmentation accuracy.
keywords: SAM 3 LiteText, SAM3 lightweight, MobileCLIP text encoder, knowledge distillation, efficient segmentation, student encoder, EfficientSAM3, concept segmentation, text-prompt segmentation, Ultralytics
---

# SAM 3 LiteText: Lightweight Text Encoder for SAM 3

**SAM 3 LiteText** replaces the heavy 353 M-parameter CLIP ViT-L text encoder in [SAM 3](sam-3.md) with a compact [MobileCLIP](https://arxiv.org/abs/2311.17049)-based student encoder trained via knowledge distillation. The ViT-H image encoder and all other components are unchanged, making LiteText a **drop-in replacement** that uses the same `SAM3SemanticPredictor` interface.

| Model variant     | Student backbone | Text-encoder params | Reduction vs SAM 3 |
| ----------------- | ---------------- | ------------------: | ------------------: |
| sam3-litetext-s0  | MobileCLIP-S0    |             42.5 M  |              ~88 %  |
| sam3-litetext-s1  | MobileCLIP-S1    |             63.5 M  |              ~82 %  |
| sam3-litetext-l   | MobileCLIP2-L    |            123.8 M  |              ~65 %  |

## Overview

Standard SAM 3 bundles a CLIP ViT-L text encoder (353 M parameters). On memory-constrained hardware or when running many concurrent inference streams, the text encoder alone can be the bottleneck. SAM 3 LiteText addresses this by substituting a **MobileCLIP transformer** trained to mimic the CLIP ViT-L output via knowledge distillation — while keeping the powerful ViT-H image backbone intact.

The three variants trade parameter count against segmentation quality:

- **S0** (42.5 M) — smallest; uses the MCT (RepMixer) architecture for further efficiency.
- **S1** (63.5 M) — standard transformer; good balance of speed and quality.
- **L** (123.8 M) — largest student encoder; closest to the original CLIP quality.

!!! note "Paper"

    SAM 3 LiteText is described in _EfficientSAM3: A Lightweight Text-Prompted Segment Anything Model_ ([arXiv:2602.12173](https://arxiv.org/abs/2602.12173)).

## Installation

SAM 3 LiteText is included in Ultralytics from the same version as SAM 3. Install or upgrade with:

```bash
pip install -U ultralytics
```

!!! warning "Model Weights"

    LiteText weights are **not automatically downloaded**. Download them from the
    [HuggingFace `Simon7108528/EfficientSAM3` — `sam3_litetext/` folder](https://huggingface.co/Simon7108528/EfficientSAM3/tree/main/sam3_litetext).

    | File name                                       | Variant | Context | Size   |
    | ----------------------------------------------- | ------- | ------: | -----: |
    | `sam3_litetext_mobileclip_s0_ctx16.pt`          | S0      |      16 | 2.2 GB |
    | `sam3_litetext_mobileclip_s0_ctx32.pt`          | S0      |      32 | 2.2 GB |
    | `sam3_litetext_mobileclip_s1_ctx16.pt`          | S1      |      16 | 2.3 GB |
    | `sam3_litetext_mobileclip_s1_ctx32.pt`          | S1      |      32 | 2.3 GB |
    | `sam3_litetext_mobileclip2_l_ctx16.pt`          | L       |      16 | 2.5 GB |
    | `sam3_litetext_mobileclip2_l_ctx32.pt`          | L       |      32 | 2.5 GB |

    Download a checkpoint with the `huggingface_hub` package:

    ```python
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id="Simon7108528/EfficientSAM3",
        filename="sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt",
    )
    print(path)  # local cache path, pass directly to SAM3SemanticPredictor
    ```

    Or with `wget`:

    ```bash
    wget https://huggingface.co/Simon7108528/EfficientSAM3/resolve/main/sam3_litetext/sam3_litetext_mobileclip_s0_ctx16.pt
    ```

## How to Use SAM 3 LiteText

The interface is identical to standard SAM 3. Simply provide the path to a LiteText checkpoint — the correct backbone and context length are auto-detected from the filename.

### Image Segmentation with Text Prompts

!!! example "Segment with a text prompt"

    === "Python"

        ```python
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model="sam3_litetext_mobileclip_s0_ctx16.pt",  # or full path
            half=False,
            save=True,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        results = predictor(source="image.jpg", text=["dog"])

        for r in results:
            print(f"Found {len(r.masks)} masks")
        ```

### Multiple text classes

!!! example "Segment multiple concept classes"

    === "Python"

        ```python
        from ultralytics.models.sam.predict import SAM3SemanticPredictor

        overrides = dict(
            conf=0.25,
            task="segment",
            mode="predict",
            model="sam3_litetext_mobileclip_s0_ctx16.pt",
            save=True,
        )
        predictor = SAM3SemanticPredictor(overrides=overrides)
        results = predictor(source="image.jpg", text=["cat", "dog", "person"])

        for r in results:
            print(f"Found {len(r.boxes)} detections across {len(predictor.model.names)} classes")
        ```

### Choosing the right checkpoint

| Use case                              | Recommended variant |
| ------------------------------------- | ------------------- |
| Fastest inference / edge deployment   | S0 ctx16            |
| Balanced quality vs. speed            | S1 ctx16            |
| Best segmentation accuracy            | L ctx16             |
| Long text descriptions (> 10 tokens)  | any ctx32 variant   |

## Architecture

### How auto-detection works

When you call `SAM3SemanticPredictor` with a LiteText checkpoint path, `build_sam3_image_model` uses two private helpers to configure the model automatically:

- **`_detect_litetext_backbone(path)`** — inspects `os.path.basename(path)` for keywords like `litetext-s0`, `efficient_sam3_text_s0`, or `mobileclip_s0` and returns the variant string (`"S0"`, `"S1"`, or `"L"`). Returns `None` for standard SAM 3 checkpoints.
- **`_detect_litetext_context_length(path)`** — looks for `ctx16` or `ctx32` in the basename (defaults to 16).

Both helpers operate on the **basename only**, so they are not confused by directory names that contain `litetext`.

### MobileCLIP text transformer (`mobile_clip.py`)

The backbone is a `MobileCLIPTextTransformer` with three variants:

| Variant | `model_name` | Layers | Heads | Hidden dim | Architecture notes          |
| ------- | ------------ | -----: | ----: | ---------: | --------------------------- |
| S0      | `mct`        |      4 |     8 |        512 | RepMixer bookend blocks     |
| S1      | `base`       |     12 |     8 |        512 | Standard multi-head attention |
| L       | `base`       |     12 |    12 |        768 | Standard multi-head attention |

The `mct` architecture (S0) wraps the first and last two transformer layers with **RepMixer** blocks — a re-parameterisable depthwise-separable structure that is fast at inference after calling `reparameterize()`.

### TextStudentEncoder (`text_encoder_student.py`)

`TextStudentEncoder` is the SAM 3–compatible wrapper around `MobileCLIPTextTransformer`:

- Tokenises input strings with `clip.tokenize()` from the standard OpenAI CLIP BPE vocabulary — the same vocabulary used by the CLIP ViT-L baseline, so no additional tokenizer dependencies are needed.
- Projects the transformer output from MobileCLIP's hidden dimension (512 / 768) to SAM 3's `d_model=256` via a single linear layer.
- Returns `(text_attention_mask, text_memory, input_embeds)` matching the `VETextEncoder` convention so the rest of the SAM 3 pipeline is unchanged.

## Comparison with Standard SAM 3

| Property                      | SAM 3 (standard)    | SAM 3 LiteText S0    | SAM 3 LiteText S1    | SAM 3 LiteText L     |
| ----------------------------- | ------------------- | -------------------- | -------------------- | -------------------- |
| Text encoder                  | CLIP ViT-L          | MobileCLIP-S0 (MCT)  | MobileCLIP-S1        | MobileCLIP2-L        |
| Text encoder params           | 353 M               | 42.5 M (−88 %)       | 63.5 M (−82 %)       | 123.8 M (−65 %)      |
| Image encoder                 | ViT-H (unchanged)   | ViT-H (unchanged)    | ViT-H (unchanged)    | ViT-H (unchanged)    |
| Predictor class               | SAM3SemanticPredictor | SAM3SemanticPredictor | SAM3SemanticPredictor | SAM3SemanticPredictor |
| API change needed             | —                   | None                 | None                 | None                 |

!!! tip "When to use LiteText"

    - **Memory-limited environments**: Running on a 10–12 GB GPU where the 353 M CLIP encoder leaves little room for larger batch sizes.
    - **Multi-stream inference**: Sharing a single GPU across several concurrent video streams.
    - **Rapid prototyping**: Faster checkpoint loading and lower idle VRAM during development.

## Citation

```bibtex
@article{zeng2025efficientsam3,
  title     = {EfficientSAM3: A Lightweight Text-Prompted Segment Anything Model},
  author    = {Zeng, Simon and others},
  journal   = {arXiv preprint arXiv:2602.12173},
  year      = {2025},
}
```

## FAQ

??? question "How is LiteText different from SAM 3?"

    Only the text encoder is changed. The ViT-H image encoder, detection transformer, segmentation head, and all other components are identical to SAM 3. LiteText checkpoints use the same `SAM3SemanticPredictor` interface.

??? question "Do I need to specify the backbone or context length manually?"

    No. Both are auto-detected from the checkpoint filename. You can override context length by passing `litetext_context_length` to `build_sam3_image_model()` directly.

??? question "Can I use LiteText for video tracking?"

    Yes. `SAM3VideoSemanticPredictor` inherits from `SAM3SemanticPredictor` and uses the same `build_sam3_image_model` path, so LiteText checkpoints work for video tracking without any code changes.

??? question "Why does the S0 MCT architecture use RepMixer blocks?"

    RepMixer blocks use a depthwise-separable structure that can be **re-parameterised** (fused) at inference time into a single depthwise convolution, reducing latency. Call `model.backbone.language_backbone.reparameterize()` after loading weights if you need maximum throughput.
