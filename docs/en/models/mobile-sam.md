---
comments: true
description: Discover MobileSAM, a lightweight and fast image segmentation model for mobile applications. Compare its performance with the original SAM and explore its various modes.
keywords: MobileSAM, image segmentation, lightweight model, fast segmentation, mobile applications, SAM, ViT encoder, Tiny-ViT, Ultralytics
---

![MobileSAM Logo](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/logo2.png)

# Mobile Segment Anything (MobileSAM)

The MobileSAM paper is now available on [arXiv](https://arxiv.org/pdf/2306.14289).

A demonstration of MobileSAM running on a CPU can be accessed at this [demo link](https://huggingface.co/spaces/dhkim2810/MobileSAM). The performance on a Mac i5 CPU takes approximately 3 seconds. On the Hugging Face demo, the interface and lower-performance CPUs contribute to a slower response, but it continues to function effectively.

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

MobileSAM is implemented in various projects including [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling), and [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D).

MobileSAM is trained on a single GPU with a 100k dataset (1% of the original images) in less than a day. The code for this training will be made available in the future.

## Available Models, Supported Tasks, and Operating Modes

This table presents the available models with their specific pre-trained weights, the tasks they support, and their compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), indicated by ✅ emojis for supported modes and ❌ emojis for unsupported modes.

| Model Type | Pre-trained Weights                                                                           | Tasks Supported                              | Inference | Validation | Training | Export |
| ---------- | --------------------------------------------------------------------------------------------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| MobileSAM  | [mobile_sam.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt) | [Instance Segmentation](../tasks/segment.md) | ✅        | ❌         | ❌       | ❌     |

## Adapting from SAM to MobileSAM

Since MobileSAM retains the same pipeline as the original SAM, we have incorporated the original's pre-processing, post-processing, and all other interfaces. Consequently, those currently using the original SAM can transition to MobileSAM with minimal effort.

MobileSAM performs comparably to the original SAM and retains the same pipeline except for a change in the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a smaller Tiny-ViT (5M). On a single GPU, MobileSAM operates at about 12ms per image: 8ms on the image encoder and 4ms on the mask decoder.

The following table provides a comparison of ViT-based image encoders:

| Image Encoder | Original SAM | MobileSAM |
| ------------- | ------------ | --------- |
| Parameters    | 611M         | 5M        |
| Speed         | 452ms        | 8ms       |

Both the original SAM and MobileSAM utilize the same prompt-guided mask decoder:

| Mask Decoder | Original SAM | MobileSAM |
| ------------ | ------------ | --------- |
| Parameters   | 3.876M       | 3.876M    |
| Speed        | 4ms          | 4ms       |

Here is the comparison of the whole pipeline:

| Whole Pipeline (Enc+Dec) | Original SAM | MobileSAM |
| ------------------------ | ------------ | --------- |
| Parameters               | 615M         | 9.66M     |
| Speed                    | 456ms        | 12ms      |

The performance of MobileSAM and the original SAM are demonstrated using both a point and a box as prompts.

![Image with Point as Prompt](https://github.com/ultralytics/docs/releases/download/0/mask-box.avif)

![Image with Box as Prompt](https://github.com/ultralytics/docs/releases/download/0/mask-box.avif)

With its superior performance, MobileSAM is approximately 5 times smaller and 7 times faster than the current FastSAM. More details are available at the [MobileSAM project page](https://github.com/ChaoningZhang/MobileSAM).

## Testing MobileSAM in Ultralytics

Just like the original SAM, we offer a straightforward testing method in Ultralytics, including modes for both Point and Box prompts.

### Model Download

You can download the model [here](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt).

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

We have implemented `MobileSAM` and `SAM` using the same API. For more usage information, please see the [SAM page](sam.md).

### Automatically Build Segmentation Datasets Leveraging a Detection Model

To automatically annotate your dataset using the Ultralytics framework, utilize the `auto_annotate` function as demonstrated below:

!!! example

    === "Python"

        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolo11x.pt", sam_model="mobile_sam.pt")
        ```

{% include "macros/sam-auto-annotate.md" %}

## Citations and Acknowledgements

If you find MobileSAM useful in your research or development work, please consider citing our paper:

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

## FAQ

### What is MobileSAM and how does it differ from the original SAM model?

MobileSAM is a lightweight, fast [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) model designed for mobile applications. It retains the same pipeline as the original SAM but replaces the heavyweight ViT-H encoder (632M parameters) with a smaller Tiny-ViT encoder (5M parameters). This change results in MobileSAM being approximately 5 times smaller and 7 times faster than the original SAM. For instance, MobileSAM operates at about 12ms per image, compared to the original SAM's 456ms. You can learn more about the MobileSAM implementation in various projects [here](https://github.com/ChaoningZhang/MobileSAM).

### How can I test MobileSAM using Ultralytics?

Testing MobileSAM in Ultralytics can be accomplished through straightforward methods. You can use Point and Box prompts to predict segments. Here's an example using a Point prompt:

```python
from ultralytics import SAM

# Load the model
model = SAM("mobile_sam.pt")

# Predict a segment based on a point prompt
model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
```

You can also refer to the [Testing MobileSAM](#testing-mobilesam-in-ultralytics) section for more details.

### Why should I use MobileSAM for my mobile application?

MobileSAM is ideal for mobile applications due to its lightweight architecture and fast inference speed. Compared to the original SAM, MobileSAM is approximately 5 times smaller and 7 times faster, making it suitable for environments where computational resources are limited. This efficiency ensures that mobile devices can perform real-time image segmentation without significant latency. Additionally, MobileSAM's models, such as [Inference](../modes/predict.md), are optimized for mobile performance.

### How was MobileSAM trained, and is the training code available?

MobileSAM was trained on a single GPU with a 100k dataset, which is 1% of the original images, in less than a day. While the training code will be made available in the future, you can currently explore other aspects of MobileSAM in the [MobileSAM GitHub repository](https://github.com/ultralytics/assets/releases/download/v8.2.0/mobile_sam.pt). This repository includes pre-trained weights and implementation details for various applications.

### What are the primary use cases for MobileSAM?

MobileSAM is designed for fast and efficient image segmentation in mobile environments. Primary use cases include:

- **Real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and segmentation** for mobile applications.
- **Low-latency image processing** in devices with limited computational resources.
- **Integration in AI-driven mobile apps** for tasks such as augmented reality (AR) and real-time analytics.

For more detailed use cases and performance comparisons, see the section on [Adapting from SAM to MobileSAM](#adapting-from-sam-to-mobilesam).
