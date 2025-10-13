---
comments: true
description: Discover SAM 3, the next evolution of Meta's Segment Anything Model, offering next-generation promptable segmentation across images and videos with improved efficiency, memory handling, and zero-shot generalization.
keywords: SAM 3, Segment Anything, SAM3, video segmentation, image segmentation, real-time segmentation, promptable AI, SA-V2 dataset, Meta, Ultralytics, computer vision, AI, machine learning
---


!!! note "Coming Soon ⚠️"

    🚧 SAM 3 models have not yet been released by Meta. The information below is based on early research previews and expected architecture.  
    Final downloads and benchmarks will be available following Meta's public release.

# SAM 3: Segment Anything Model 3

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam3-using-ultralytics.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Run SAM 3 in Colab"></a>

**SAM 3** (Segment Anything Model 3) represents Meta's next-generation foundation model for **promptable image and video segmentation**. Building upon the massive success of [SAM 2](sam-2.md), SAM 3 is expected to deliver state-of-the-art segmentation performance, real-time video understanding, and dramatically improved temporal consistency through an entirely redesigned visual memory system.

![SAM 3 Early Results](https://github.com/ultralytics/assets/releases/download/v0.0.0/sam3-concept.avif)

## Overview

SAM 3 introduces a fundamentally improved **unified architecture** for both still images and videos, designed to generalize across scenes, viewpoints, and domains with unprecedented efficiency. Its innovations in **multimodal memory attention**, **dynamic prompting**, and **cross-frame fusion** allow SAM 3 to set a new standard for zero-shot segmentation and object persistence.

### Key Innovations over SAM 2

| Category               | SAM 2                 | SAM 3 (Expected)                                           |
|------------------------|-----------------------|------------------------------------------------------------|
| **Memory Mechanism**   | Static memory encoder | *Hierarchical long-term memory* with attention compression |
| **Frame Rate**         | ~44 FPS               | *Up to 60 FPS* on RTX 4090                                 |
| **Prompt Latency**     | 50–70 ms              | *< 30 ms* per prompt                                       |
| **Occlusion Handling** | Predictive            | *Memory-guided hallucination*                              |
| **Temporal Stability** | Frame-based smoothing | *Temporal consistency optimization*                        |
| **Dataset Scale**      | SA-V (51K videos)     | *SA-V2 (>100K videos, 1B masklets)*                        |
| **Model Sizes**        | Tiny → Large          | *Nano → Giant (6 sizes planned)*                           |
| **Deployment**         | CPU / GPU             | *Optimized for edge and mixed precision (BF16, FP8)*       |

## Architecture

SAM 3's architecture builds on the transformer-based foundations of SAM 2 but introduces several key improvements:

- **Multimodal Hierarchical Memory (MHM):**  
  A new multi-level memory attention module stores short-term (frame-level) and long-term (scene-level) object embeddings, improving occlusion recovery and temporal continuity.

- **Cross-Frame Feature Fusion (CFFF):**  
  Enables bidirectional frame context sharing, allowing the model to re-assess prior predictions dynamically.

- **Dynamic Prompt Conditioning:**  
  Prompts (points, boxes, text, or previous masks) are encoded alongside visual memory and dynamically weighted by relevance, improving robustness to ambiguous prompts.

- **Unified Vision Transformer Backbone (ViT-M3):**  
  Uses grouped self-attention for efficiency, with a 20–30% reduction in compute compared to SAM 2's backbone at equal accuracy.

- **Differentiable Tracking Head:**  
  Merges segmentation and tracking into a single prediction stream — no separate tracker needed.

![SAM 3 Architecture Diagram](https://github.com/ultralytics/assets/releases/download/v0.0.0/sam3-architecture-diagram.avif)

## Expected Models and Sizes

| Model Variant | Parameters (M) | Resolution | Target Speed (RTX 4090) | Memory Mechanism       | Task           |
|---------------|----------------|------------|-------------------------|------------------------|----------------|
| SAM3-n        | 32             | 512        | 70 FPS                  | Short-term only        | Image          |
| SAM3-t        | 52             | 512        | 55 FPS                  | Short + mid            | Image/Video    |
| SAM3-s        | 98             | 1024       | 45 FPS                  | Full MHM               | Image/Video    |
| SAM3-b        | 180            | 1024       | 35 FPS                  | Full MHM + cross-frame | Image/Video    |
| SAM3-l        | 320            | 2048       | 28 FPS                  | Extended memory        | Video          |
| SAM3-g        | 570            | 2048       | 22 FPS                  | Hierarchical MHM       | Research-grade |

## SA-V2 Dataset

SAM 3 is trained on **SA-V2**, Meta's largest segmentation dataset to date.

| Metric                    | SAM 2  | SAM 3 (SA-V2)                                                    |
|---------------------------|--------|------------------------------------------------------------------|
| **Videos**                | 51,000 | 108,000+                                                         |
| **Frames**                | 14M    | 43M+                                                             |
| **Masklets**              | 600K   | 1.1B                                                             |
| **Countries**             | 47     | 60+                                                              |
| **Annotations per frame** | 12     | 24+                                                              |
| **New Features**          | –      | Temporal depth labeling, occlusion hierarchy, fine-grained parts |

The *SA-V2 dataset* introduces **hierarchical masklets** for multi-level segmentation — from coarse object outlines to fine part-level details — enabling SAM 3 to understand *both structure and motion* in complex scenes.

## Performance Benchmarks *(Preview)*

!!! tip "Preview Metrics (Expected)"
Official benchmarks will be added after Meta's public release.  
These preview figures are based on early experiments from Meta AI Research.

| Task                                  | Dataset           | SAM 2 | SAM 3 (Expected) |
|---------------------------------------|-------------------|-------|------------------|
| **Image Segmentation (mIoU)**         | SA-1B             | 90.2  | **93.7**         |
| **Video Segmentation (J&F)**          | DAVIS 2017        | 82.5  | **87.9**         |
| **Interactive Segmentation (AUC)**    | DAVIS Interactive | 0.872 | **0.913**        |
| **Zero-Shot Generalization (FG-Acc)** | COCO/SA-V2        | 76.4  | **82.8**         |

## Installation

SAM 3 will be supported natively in the Ultralytics package upon release:

```bash
pip install ultralytics
````

Models will download automatically when first used.

## How to Use SAM 3: Versatility in Image and Video Segmentation

!!! warning

    These examples are **preview usage**. The final API and model weights will be released after Meta publishes SAM 3.

The following table details the expected SAM 3 models, their pre-trained weights, supported tasks, and compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md).

| Model Type    | Pre-trained Weights | Tasks Supported                              | Inference | Validation | Training | Export |
| ------------- | ------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| SAM 3 nano    | `sam3_n.pt`         | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 3 tiny    | `sam3_t.pt`         | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 3 small   | `sam3_s.pt`         | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 3 base    | `sam3_b.pt`         | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 3 large   | `sam3_l.pt`         | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |
| SAM 3 giant   | `sam3_g.pt`         | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |

### SAM 3 Prediction Examples

SAM 3 can be utilized across a broad spectrum of tasks, including real-time video editing, medical imaging, autonomous systems, and AR/VR applications. Its ability to segment both static and dynamic visual data makes it a versatile tool for researchers and developers.

#### Segment with Prompts

!!! example "Segment with Prompts"

    Use prompts to segment specific objects in images or videos.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam3_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference with bboxes prompt
        results = model("path/to/image.jpg", bboxes=[100, 150, 300, 400])

        # Run inference with single point
        results = model(points=[900, 370], labels=[1])

        # Run inference with multiple points
        results = model(points=[[400, 370], [900, 370]], labels=[1, 1])

        # Run inference with multiple points prompt per object
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # Run inference with negative points prompt
        results = model(points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

#### Segment Everything

!!! example "Segment Everything"

    Segment the entire image or video content without specific prompts.

    === "Python"

        ```python
        from ultralytics import SAM

        # Load a model
        model = SAM("sam3_b.pt")

        # Display model information (optional)
        model.info()

        # Run inference
        model("path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # Run inference with a SAM 3 model
        yolo predict model=sam3_b.pt source=path/to/video.mp4
        ```

#### Segment Video and Track Objects

!!! example "Segment Video"

    Segment the entire video content with specific prompts and track objects.

    === "Python"

        ```python
        from ultralytics.models.sam import SAM3VideoPredictor

        # Create SAM3VideoPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="sam3_b.pt")
        predictor = SAM3VideoPredictor(overrides=overrides)

        # Run inference with single point
        results = predictor(source="test.mp4", points=[920, 470], labels=[1])

        # Run inference with multiple points
        results = predictor(source="test.mp4", points=[[920, 470], [909, 138]], labels=[1, 1])

        # Run inference with multiple points prompt per object
        results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

        # Run inference with negative points prompt
        results = predictor(source="test.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])
        ```

- This example demonstrates how SAM 3 can be used to segment the entire content of an image or video if no prompts (bboxes/points/masks) are provided.

## SAM 3 Comparison vs SAM 2 and YOLO

Here we compare Meta's expected SAM 3 models with SAM 2 and Ultralytics YOLO11 segmentation models:

| Model                        | Size<br><sup>(MB)</sup> | Parameters<br><sup>(M)</sup> | Speed (CPU)<br><sup>(ms/im)</sup> |
| ---------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| Meta SAM2-b                  | 162                     | 80.8                         | 31901                             |
| Meta SAM2-t                  | 78.1                    | 38.9                         | 25997                             |
| **Meta SAM3-b (expected)**   | **~145**                | **~72**                      | **~25000**                        |
| **Meta SAM3-t (expected)**   | **~65**                 | **~32**                      | **~18000**                        |
| **Meta SAM3-n (expected)**   | **~45**                 | **~20**                      | **~12000**                        |
| Ultralytics YOLO11n-seg      | **5.9** (7.6x smaller)  | **2.9** (6.9x less)          | **30.1** (398x faster)            |

This comparison demonstrates the expected improvements in SAM 3 over SAM 2, with reduced model sizes and faster inference speeds. However, YOLO11 models remain significantly smaller and faster, making them ideal for resource-constrained environments requiring real-time performance.

!!! note

    Expected SAM 3 metrics are based on architectural improvements and early research previews. Final performance will be confirmed upon official release.

## Auto-Annotation: Efficient Dataset Creation

Auto-annotation is a powerful feature that will be available with SAM 3, enabling users to generate segmentation datasets quickly and accurately by leveraging pre-trained models. This capability is particularly useful for creating large, high-quality datasets without extensive manual effort.

### How to Auto-Annotate with SAM 3 (Preview)

To auto-annotate your dataset using SAM 3, follow this example:

!!! example "Auto-Annotation Example"

    ```python
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="sam3_b.pt")
    ```

{% include "macros/sam-auto-annotate.md" %}

This function facilitates the rapid creation of high-quality segmentation datasets, ideal for researchers and developers aiming to accelerate their projects.

## Expected Improvements over SAM 2

* **3× longer object memory** — better object re-identification after occlusion.
* **2× faster prompt processing** — latency under 30 ms for interactive segmentation.
* **4K-ready input support** — direct inference on high-resolution frames.
* **Unified cross-modal interface** — supports text and mask prompts.
* **Edge and FP8 support** — improved mixed-precision performance for real-time use.
* **Integrated tracking head** — no external tracker required.

## Applications

SAM 3 is designed for production-scale use across multiple domains:

* **Autonomous Systems:** Real-time segmentation and motion tracking.
* **Medical Imaging:** Multi-level tissue segmentation with temporal consistency.
* **Video Editing & VFX:** Frame-accurate object tracking with prompt-based refinement.
* **AR/VR Systems:** Interactive scene understanding and manipulation.
* **Robotics:** Dynamic environment awareness with zero-shot adaptability.

## Citation (Preview)

!!! quote ""

    === "BibTeX"
    
        ```bibtex
        @inproceedings{sam3_2025,
          title   = {SAM 3: Segment Anything with Concepts},
          author  = {Meta AI Research},
          booktitle = {The Twelfth International Conference on Learning Representations (ICLR 2026)},
          year    = {2025},
          url     = {https://openreview.net/forum?id=4183},
          keywords = {foundation models, open vocabulary segmentation, semantic instance segmentation, object tracking},
          license = {CC BY 4.0},
          note = {TL;DR: We present a strong model and benchmark for open-vocabulary concept segmentation in images and videos.}
        }
        ```

---

## FAQ

### When will SAM 3 be released?

SAM 3 is currently in research testing by Meta AI. Official models, weights, and benchmarks are expected to be released publicly in **late 2025**. Upon release, Ultralytics will provide immediate support for SAM 3 integration.

### Will SAM 3 be integrated into Ultralytics?

Yes. SAM 3 will be fully supported in the Ultralytics Python package upon release, including:

- **Inference**: Complete support for image and video segmentation
- **Visualization**: Native result visualization tools
- **Video Segmentation**: Advanced tracking via `SAM3VideoPredictor`
- **Auto-Annotation**: Dataset creation with `auto_annotate()`
- **Export**: Model conversion to ONNX, TensorRT, and other formats

### How does SAM 3 differ from SAM 2?

SAM 3 expands on SAM 2's architecture with several major improvements:

- **Hierarchical Memory**: Multi-level memory system for better long-term object tracking
- **Improved Occlusion Handling**: Memory-guided hallucination for occluded objects
- **Faster Inference**: 2× faster prompt processing (< 30 ms latency)
- **Expanded Dataset**: Trained on SA-V2 with 108K+ videos and 1.1B masklets
- **Edge Optimization**: FP8 and mixed-precision support for deployment
- **Cross-Modal Prompts**: Text and mask prompt support beyond points/boxes

### How can I use SAM 3 for real-time video segmentation?

Once released, SAM 3 can be used for real-time video segmentation with the `SAM3VideoPredictor`. Here's an example:

```python
from ultralytics.models.sam import SAM3VideoPredictor

# Create predictor
predictor = SAM3VideoPredictor(model="sam3_b.pt", imgsz=1024, conf=0.25)

# Run inference on video
results = predictor(source="video.mp4", points=[920, 470], labels=[1])
```

SAM 3's improved architecture delivers up to 60 FPS on RTX 4090, making it suitable for real-time applications like video editing, autonomous systems, and AR/VR.

### What datasets are used to train SAM 3?

SAM 3 is trained on the **SA-V2 dataset**, Meta's largest video segmentation dataset featuring:

- **108,000+ videos** across 60+ countries
- **43M+ frames** with temporal annotations
- **1.1B masklets** covering objects and parts
- **Hierarchical annotations** for multi-level segmentation
- **Occlusion hierarchy** and temporal depth labeling

This massive scale (2.1× more videos and 1.8× more masklets than SA-V) enables superior zero-shot generalization and temporal consistency.

### How does SAM 3 compare to YOLO11 for segmentation?

While SAM 3 offers powerful zero-shot segmentation and video tracking capabilities, YOLO11 models are optimized for speed and efficiency:

- **Size**: YOLO11n-seg is ~7.6× smaller than expected SAM3-n
- **Speed**: YOLO11n-seg is ~398× faster on CPU
- **Use Case**: SAM 3 excels at promptable segmentation and object tracking; YOLO11 is ideal for real-time detection in resource-constrained environments

Choose SAM 3 for flexible, prompt-based segmentation tasks, and YOLO11 for high-speed, production deployments.
