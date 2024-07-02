---
comments: true
description: Discover MobileSAM, a lightweight and fast image segmentation model for mobile applications. Compare its performance with the original SAM and explore its various modes.
keywords: MobileSAM, image segmentation, lightweight model, fast segmentation, mobile applications, SAM, ViT encoder, Tiny-ViT, Ultralytics
---

![MobileSAM Logo](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Mobile Segment Anything (MobileSAM)

The MobileSAM paper is now available on [arXiv](https://arxiv.org/pdf/2306.14289.pdf).

A demonstration of MobileSAM running on a CPU can be accessed at this [demo link](https://huggingface.co/spaces/dhkim2810/MobileSAM). The performance on a Mac i5 CPU takes approximately 3 seconds. On the Hugging Face demo, the interface and lower-performance CPUs contribute to a slower response, but it continues to function effectively.

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

![Image with Point as Prompt](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![Image with Box as Prompt](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

With its superior performance, MobileSAM is approximately 5 times smaller and 7 times faster than the current FastSAM. More details are available at the [MobileSAM project page](https://github.com/ChaoningZhang/MobileSAM).

## Testing MobileSAM in Ultralytics

Just like the original SAM, we offer a straightforward testing method in Ultralytics, including modes for both Point and Box prompts.

### Model Download

You can download the model [here](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt).

### Point Prompt

!!! Example

    === "Python"

        ```python
        from ultralytics import SAM

        # Load the model
        model = SAM("mobile_sam.pt")

        # Predict a segment based on a point prompt
        model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
        ```

### Box Prompt

!!! Example

    === "Python"

        ```python
        from ultralytics import SAM

        # Load the model
        model = SAM("mobile_sam.pt")

        # Predict a segment based on a box prompt
        model.predict("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])
        ```

We have implemented `MobileSAM` and `SAM` using the same API. For more usage information, please see the [SAM page](sam.md).

## Citations and Acknowledgements

If you find MobileSAM useful in your research or development work, please consider citing our paper:

!!! Quote ""

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

### How do I use MobileSAM for image segmentation on a mobile application?

MobileSAM is specifically designed for lightweight and fast image segmentation on mobile applications. To get started, you can download the model weights [here](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) and use the following Python code snippet for inference:

```python
from ultralytics import SAM

# Load the model
model = SAM("mobile_sam.pt")

# Predict a segment based on a point prompt
model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
```

For more detailed usage and various prompts, refer to the [SAM page](sam.md).

### What are the performance benefits of using MobileSAM over the original SAM?

MobileSAM offers significant improvements in both size and speed over the original SAM. Here is a detailed comparison:

- **Image Encoder**: MobileSAM uses a smaller Tiny-ViT (5M parameters) instead of the original heavyweight ViT-H (611M parameters), resulting in an 8ms encoding time versus 452ms with SAM.
- **Overall Pipeline**: MobileSAM's entire pipeline, including image encoding and mask decoding, operates at 12ms per image compared to SAM's 456ms, making it approximately 7 times faster.
    In summary, MobileSAM is about 5 times smaller and 7 times faster than the original SAM, making it ideal for mobile applications.

### Why should developers adopt MobileSAM for mobile applications?

Developers should consider using MobileSAM for mobile applications due to its lightweight and fast performance, making it highly efficient for real-time image segmentation tasks.

- **Efficiency**: MobileSAM's Tiny-ViT encoder allows for rapid processing, achieving segmentation results in just 12ms.
- **Size**: The model size is significantly reduced, making it easier to deploy and run on mobile devices.
    These advancements facilitate real-time applications, such as augmented reality, mobile games, and other interactive experiences.

Learn more about the MobileSAM's performance on its [project page](https://github.com/ChaoningZhang/MobileSAM).

### How easy is it to transition from the original SAM to MobileSAM?

Transitioning from the original SAM to MobileSAM is straightforward as MobileSAM retains the same pipeline, including pre-processing, post-processing, and interfaces. Only the image encoder has been changed to the more efficient Tiny-ViT. Users currently using SAM can switch to MobileSAM with minimal code modifications, benefiting from improved performance without the need for significant reconfiguration.

### What tasks are supported by the MobileSAM model?

The MobileSAM model supports instance segmentation tasks. Currently, it is optimized for [Inference](../modes/predict.md) mode. Additional tasks like validation, training, and export are not supported at this time, as indicated in the mode compatibility table:
| Model Type | Tasks Supported | Inference | Validation | Training | Export |
| ---------- | -------------------------------------------- | --------- | ---------- | -------- | ------ |
| MobileSAM | [Instance Segmentation](../tasks/segment.md) | ✅ | ❌ | ❌ | ❌ |

For more information about supported tasks and operational modes, check the [tasks page](../tasks/segment.md) and the mode details like [Inference](../modes/predict.md), [Validation](../modes/val.md), and [Export](../modes/export.md).
