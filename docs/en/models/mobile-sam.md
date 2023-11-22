---
comments: true
description: Learn more about MobileSAM, its implementation, comparison with the original SAM, and how to download and test it in the Ultralytics framework. Improve your mobile applications today.
keywords: MobileSAM, Ultralytics, SAM, mobile applications, Arxiv, GPU, API, image encoder, mask decoder, model download, testing method
---

![MobileSAM Logo](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# Mobile Segment Anything (MobileSAM)

The MobileSAM paper is now available on [arXiv](https://arxiv.org/pdf/2306.14289.pdf).

A demonstration of MobileSAM running on a CPU can be accessed at this [demo link](https://huggingface.co/spaces/dhkim2810/MobileSAM). The performance on a Mac i5 CPU takes approximately 3 seconds. On the Hugging Face demo, the interface and lower-performance CPUs contribute to a slower response, but it continues to function effectively.

MobileSAM is implemented in various projects including [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything), [AnyLabeling](https://github.com/vietanhdev/anylabeling), and [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D).

MobileSAM is trained on a single GPU with a 100k dataset (1% of the original images) in less than a day. The code for this training will be made available in the future.

## Available Models, Supported Tasks, and Operating Modes

This table presents the available models with their specific pre-trained weights, the tasks they support, and their compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), indicated by ✅ emojis for supported modes and ❌ emojis for unsupported modes.

| Model Type | Pre-trained Weights | Tasks Supported                              | Inference | Validation | Training | Export |
|------------|---------------------|----------------------------------------------|-----------|------------|----------|--------|
| MobileSAM  | `mobile_sam.pt`     | [Instance Segmentation](../tasks/segment.md) | ✅         | ❌          | ❌        | ✅      |

## Adapting from SAM to MobileSAM

Since MobileSAM retains the same pipeline as the original SAM, we have incorporated the original's pre-processing, post-processing, and all other interfaces. Consequently, those currently using the original SAM can transition to MobileSAM with minimal effort.

MobileSAM performs comparably to the original SAM and retains the same pipeline except for a change in the image encoder. Specifically, we replace the original heavyweight ViT-H encoder (632M) with a smaller Tiny-ViT (5M). On a single GPU, MobileSAM operates at about 12ms per image: 8ms on the image encoder and 4ms on the mask decoder.

The following table provides a comparison of ViT-based image encoders:

| Image Encoder | Original SAM | MobileSAM |
|---------------|--------------|-----------|
| Parameters    | 611M         | 5M        |
| Speed         | 452ms        | 8ms       |

Both the original SAM and MobileSAM utilize the same prompt-guided mask decoder:

| Mask Decoder | Original SAM | MobileSAM |
|--------------|--------------|-----------|
| Parameters   | 3.876M       | 3.876M    |
| Speed        | 4ms          | 4ms       |

Here is the comparison of the whole pipeline:

| Whole Pipeline (Enc+Dec) | Original SAM | MobileSAM |
|--------------------------|--------------|-----------|
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
        model = SAM('mobile_sam.pt')

        # Predict a segment based on a point prompt
        model.predict('ultralytics/assets/zidane.jpg', points=[900, 370], labels=[1])
        ```

### Box Prompt

!!! Example

    === "Python"
        ```python
        from ultralytics import SAM

        # Load the model
        model = SAM('mobile_sam.pt')

        # Predict a segment based on a box prompt
        model.predict('ultralytics/assets/zidane.jpg', bboxes=[439, 437, 524, 709])
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
