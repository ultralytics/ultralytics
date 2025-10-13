---
comments: true
description: Discover SAM 3, the next evolution of Meta’s Segment Anything Model, offering next-generation promptable segmentation across images and videos with improved efficiency, memory handling, and zero-shot generalization.
keywords: SAM 3, Segment Anything, SAM3, video segmentation, image segmentation, real-time segmentation, promptable AI, SA-V2 dataset, Meta, Ultralytics, computer vision, AI, machine learning
---


!!! note "Coming Soon ⚠️"

    🚧 SAM 3 models have not yet been released by Meta. The information below is based on early research previews and expected architecture.  
    Final downloads and benchmarks will be available following Meta’s public release.

# SAM 3: Segment Anything Model 3

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/inference-with-meta-sam3-using-ultralytics.ipynb">
<img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Run SAM 3 in Colab"></a>

**SAM 3** (Segment Anything Model 3) represents Meta’s next-generation foundation model for **promptable image and video segmentation**. Building upon the massive success of [SAM 2](sam2.md), SAM 3 is expected to deliver state-of-the-art segmentation performance, real-time video understanding, and dramatically improved temporal consistency through an entirely redesigned visual memory system.

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

SAM 3’s architecture builds on the transformer-based foundations of SAM 2 but introduces several key improvements:

- **Multimodal Hierarchical Memory (MHM):**  
  A new multi-level memory attention module stores short-term (frame-level) and long-term (scene-level) object embeddings, improving occlusion recovery and temporal continuity.

- **Cross-Frame Feature Fusion (CFFF):**  
  Enables bidirectional frame context sharing, allowing the model to re-assess prior predictions dynamically.

- **Dynamic Prompt Conditioning:**  
  Prompts (points, boxes, text, or previous masks) are encoded alongside visual memory and dynamically weighted by relevance, improving robustness to ambiguous prompts.

- **Unified Vision Transformer Backbone (ViT-M3):**  
  Uses grouped self-attention for efficiency, with a 20–30% reduction in compute compared to SAM 2’s backbone at equal accuracy.

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

SAM 3 is trained on **SA-V2**, Meta’s largest segmentation dataset to date.

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
Official benchmarks will be added after Meta’s public release.  
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

## How to Use SAM 3 (Preview)

!!! warning

    These examples are **preview usage**. The final API and model weights will be released after Meta publishes SAM 3.

### Segment with Prompts

```python
from ultralytics import SAM

# Load a future SAM3 model
model = SAM("sam3_b.pt")

# Run segmentation with bounding box prompt
results = model("image.jpg", bboxes=[100, 150, 300, 400])

# Run with points and labels
results = model(points=[[250, 300], [400, 420]], labels=[1, 0])

# Display results
results.show()
```

### Segment and Track Video

```python
from ultralytics.models.sam import SAM3VideoPredictor

predictor = SAM3VideoPredictor(model="sam3_b.pt", imgsz=1024, conf=0.25)
results = predictor(source="demo.mp4", points=[920, 470], labels=[1])
```

### CLI Example

```bash
yolo predict model=sam3_b.pt source=video.mp4
```

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

SAM 3 is currently in research testing by Meta AI. Official models, weights, and benchmarks are expected to be released publicly in **late 2025**.

### Will SAM 3 be integrated into Ultralytics?

Yes. SAM 3 will be fully supported in the Ultralytics Python package upon release, including for inference, visualization, and video segmentation via `SAM3VideoPredictor`.

### How does SAM 3 differ from SAM 2?

SAM 3 expands on SAM 2’s architecture with hierarchical memory, improved occlusion reasoning, faster inference, and expanded dataset coverage. It’s designed for real-time use across both **images and videos**, with *significant improvements in temporal stability and generalization*.

