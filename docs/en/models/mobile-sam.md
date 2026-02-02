---
comments: true
description: Discover MobileSAM, a lightweight and fast image segmentation model for mobile and edge applications. Compare its performance with SAM and YOLO models.
keywords: MobileSAM, image segmentation, lightweight model, fast segmentation, mobile applications, SAM, Tiny-ViT, YOLO, Ultralytics
---

![MobileSAM lightweight image segmentation model logo](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/logo2.png)

# Mobile Segment Anything (MobileSAM)

MobileSAM is a compact, efficient image segmentation model purpose-built for mobile and edge devices. Designed to bring the power of Meta's Segment Anything Model ([SAM](sam.md)) to environments with limited compute, MobileSAM delivers near-instant segmentation while maintaining compatibility with the original SAM pipeline. Whether you're developing real-time applications or lightweight deployments, MobileSAM provides impressive segmentation results with a fraction of the size and speed requirements of its predecessors.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/yXQPLMrNX2s"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Run Inference with MobileSAM using Ultralytics | Step-by-Step Guide 🎉
</p>

MobileSAM has been adopted in a variety of projects, including [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling), and [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D).

MobileSAM was trained on a single GPU using a 100k image dataset (1% of the original images) in less than a day. The training code will be released in the future.

## Available Models, Supported Tasks, and Operating Modes

The table below outlines the available MobileSAM model, its pretrained weights, supported tasks, and compatibility with different operating modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md). Supported modes are indicated by ✅ and unsupported modes by ❌.

| Model Type | Pretrained Weights                                                                            | Tasks Supported                              | Inference | Validation | Training | Export |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| MobileSAM  | [mobile_sam.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |

## MobileSAM Comparison vs YOLO

The following comparison highlights the differences between Meta's SAM variants, MobileSAM, and Ultralytics' smallest segmentation models, including [YOLO11n-seg](../models/yolo11.md):

| Model                                                                           | Size<br><sup>(MB)</sup> | Parameters<br><sup>(M)</sup> | Speed (CPU)<br><sup>(ms/im)</sup> |
| ------------------------------------------------------------------------------- | ----------------------- | ---------------------------- | --------------------------------- |
| Meta SAM-b                                                                      | 375                     | 93.7                         | 49401                             |
| Meta SAM2-b                                                                     | 162                     | 80.8                         | 31901                             |
| Meta SAM2-t                                                                     | 78.1                    | 38.9                         | 25997                             |
| MobileSAM                                                                       | 40.7                    | 10.1                         | 25381                             |
| FastSAM-s with YOLOv8 [backbone](https://www.ultralytics.com/glossary/backbone) | 23.7                    | 11.8                         | 55.9                              |
| Ultralytics YOLOv8n-seg                                                         | **6.7** (11.7x smaller) | **3.4** (11.4x less)         | **24.5** (1061x faster)           |
| Ultralytics YOLO11n-seg                                                         | **5.9** (13.2x smaller) | **2.9** (13.4x less)         | **30.1** (864x faster)            |

This comparison demonstrates the substantial differences in model size and speed between SAM variants and YOLO segmentation models. While SAM models offer unique automatic segmentation capabilities, YOLO models—especially YOLOv8n-seg and YOLO11n-seg—are significantly smaller, faster, and more computationally efficient.

Tests were conducted on a 2025 Apple M4 Pro with 24GB RAM using `torch==2.6.0` and `ultralytics==8.3.90`. To reproduce these results:

!!! example

    === "Python"

        ```python
        from ultralytics import ASSETS, SAM, YOLO, FastSAM

        # Profile SAM2-t, SAM2-b, SAM-b, MobileSAM
        for file in ["sam_b.pt", "sam2_b.pt", "sam2_t.pt", "mobile_sam.pt"]:
            model = SAM(file)
            model.info()
            model(ASSETS)

        # Profile FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model(ASSETS)

        # Profile YOLO models
        for file_name in ["yolov8n-seg.pt", "yolo11n-seg.pt"]:
            model = YOLO(file_name)
            model.info()
            model(ASSETS)
        ```

## Adapting from SAM to MobileSAM

MobileSAM retains the same pipeline as the original [SAM](sam.md), including pre-processing, post-processing, and all interfaces. This means you can transition from SAM to MobileSAM with minimal changes to your workflow.

The key difference is the image encoder: MobileSAM replaces the original ViT-H encoder (632M parameters) with a much smaller Tiny-ViT encoder (5M parameters). On a single GPU, MobileSAM processes an image in about 12ms (8ms for the encoder, 4ms for the mask decoder).

### ViT-Based Image Encoder Comparison

| Image Encoder | Original SAM | MobileSAM |
| ------------- | ------------ | --------- |
| Parameters    | 611M         | 5M        |
| Speed         | 452ms        | 8ms       |

### Prompt-Guided Mask Decoder

| Mask Decoder | Original SAM | MobileSAM |
| ------------ | ------------ | --------- |
| Parameters   | 3.876M       | 3.876M    |
| Speed        | 4ms          | 4ms       |

### Whole Pipeline Comparison

| Whole Pipeline (Enc+Dec) | Original SAM | MobileSAM |
| ------------------------ | ------------ | --------- |
| Parameters               | 615M         | 9.66M     |
| Speed                    | 456ms        | 12ms      |

The performance of MobileSAM and the original SAM is illustrated below using both point and box prompts.

![Image with Point as Prompt](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mask-box.avif)

![Image with Box as Prompt](https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/mask-box.avif)

MobileSAM is approximately 7 times smaller and 5 times faster than FastSAM. For further details, visit the [MobileSAM project page](https://github.com/ChaoningZhang/MobileSAM).

## Testing MobileSAM in Ultralytics

Just like the original [SAM](sam.md), Ultralytics provides a simple interface for testing MobileSAM, supporting both Point and Box prompts.

### Model Download

Download the MobileSAM pretrained weights from [Ultralytics assets](https://github.com/ultralytics/assets/releases/download/v8.4.0/mobile_sam.pt).

### Point Prompt

!!! example

    === "Python"

        ```python
        from ultralytics import SAM

        # Load the model
        model = SAM("mobile_sam.pt")

        # Predict a segment based on a single point prompt
        model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

        # Predict multiple segments based on multiple points prompt
        model.predict("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

        # Predict a segment based on multiple points prompt per object
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # Predict a segment using both positive and negative prompts.
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

### Box Prompt

!!! example

    === "Python"

        ```python
        from ultralytics import SAM

        # Load the model
        model = SAM("mobile_sam.pt")

        # Predict a segment based on a single point prompt
        model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

        # Predict multiple segments based on multiple points prompt
        model.predict("ultralytics/assets/zidane.jpg", points=[[400, 370], [900, 370]], labels=[1, 1])

        # Predict a segment based on multiple points prompt per object
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 1]])

        # Predict a segment using both positive and negative prompts.
        model.predict("ultralytics/assets/zidane.jpg", points=[[[400, 370], [900, 370]]], labels=[[1, 0]])
        ```

Both `MobileSAM` and `SAM` share the same API. For more usage details, see the [SAM documentation](sam.md).

### Automatically Build Segmentation Datasets Using a Detection Model

To automatically annotate your dataset with the Ultralytics framework, use the `auto_annotate` function as shown below:

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="mobile_sam.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

## Citations and Acknowledgments

If MobileSAM is helpful in your research or development, please consider citing the following paper:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
        ```

Read the full [MobileSAM paper on arXiv](https://arxiv.org/pdf/2306.14289).

## FAQ

### What Is MobileSAM and How Does It Differ from the Original SAM Model?

MobileSAM is a lightweight, fast [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) model optimized for mobile and edge applications. It maintains the same pipeline as the original SAM but replaces the large ViT-H encoder (632M parameters) with a compact Tiny-ViT encoder (5M parameters). This results in MobileSAM being about 5 times smaller and 7 times faster than the original SAM, operating at roughly 12ms per image versus SAM's 456ms. Explore more about MobileSAM's implementation on the [MobileSAM GitHub repository](https://github.com/ChaoningZhang/MobileSAM).

### How Can I Test MobileSAM Using Ultralytics?

Testing MobileSAM in Ultralytics is straightforward. You can use Point and Box prompts to predict segments. For example, using a Point prompt:

```python
from ultralytics import SAM

# Load the model
model = SAM("mobile_sam.pt")

# Predict a segment based on a point prompt
model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
```

For more details, see the [Testing MobileSAM in Ultralytics](#testing-mobilesam-in-ultralytics) section.

### Why Should I Use MobileSAM for My Mobile Application?

MobileSAM is ideal for mobile and edge applications due to its lightweight design and rapid inference speed. Compared to the original SAM, MobileSAM is about 5 times smaller and 7 times faster, making it suitable for real-time segmentation on devices with limited computational resources. Its efficiency enables mobile devices to perform [real-time image segmentation](https://www.ultralytics.com/glossary/real-time-inference) without significant latency. Additionally, MobileSAM supports [Inference mode](../modes/predict.md) optimized for mobile performance.

### How Was MobileSAM Trained, and Is the Training Code Available?

MobileSAM was trained on a single GPU with a 100k image dataset (1% of the original images) in under a day. While the training code will be released in the future, you can currently access pretrained weights and implementation details from the [MobileSAM GitHub repository](https://github.com/ChaoningZhang/MobileSAM).

### What Are the Primary Use Cases for MobileSAM?

MobileSAM is designed for fast, efficient image segmentation in mobile and edge environments. Primary use cases include:

- **Real-time [object detection and segmentation](https://www.ultralytics.com/glossary/object-detection)** for mobile apps
- **Low-latency image processing** on devices with limited compute
- **Integration in AI-powered mobile applications** for augmented reality (AR), analytics, and more

For more details on use cases and performance, see [Adapting from SAM to MobileSAM](#adapting-from-sam-to-mobilesam) and the [Ultralytics blog on MobileSAM applications](https://www.ultralytics.com/blog/applications-of-meta-ai-segment-anything-model-2-sam-2).
