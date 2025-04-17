---
comments: true
description: Learn how to integrate Microsoft Florence-2 with the Ultralytics Python package for zero-shot object detection, segmentation, captioning, and OCR tasks using unified prompts.
keywords: Florence-2, Microsoft Florence, Ultralytics, object detection, segmentation, image captioning, OCR, vision-language models, unified prompts, computer vision, FLD-5B, auto annotation
---

# Microsoft Florence-2 Integration with Ultralytics Python Package

<a href="https://colab.research.google.com/github/ultralytics/notebooks/blob/main/notebooks/how-to-use-florence-2-for-object-detection-image-captioning-ocr-and-segmentation.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="How to use florence-2 for object detection, image-captioning, ocr and segmentation"></a>

[Florence-2](https://arxiv.org/pdf/2311.06242) is a large vision foundation model developed by [Microsoft](https://www.microsoft.com/) that uses a unified, prompt-based representation to tackle a wide range of computer vision and vision-language tasks. Unlike traditional models that require task-specific architectures, Florence-2 accepts simple text prompts as instructions and produces results in text form for tasks such as image captioning, object detection, optical character recognition (OCR), visual grounding, and segmentation. This unified approach enables a single model to handle image-level, region-level, and pixel-level tasks with a consistent interface.

![Florence-2-architecture](https://github.com/ultralytics/docs/releases/download/0/florence-2-architecture.png)

## Key Features

- **Unified Prompt-Based Representation:** Florence-2 interprets text prompts to perform various vision tasks within a single model. A special token in the prompt activates the desired task, and the model outputs the result in text format (e.g. a caption, a list of objects, coordinates). This prompt-driven design means the same model can handle captioning, detection, segmentation, OCR, and more by simply changing the prompt, much like how language models follow instructions.

- **Multi-Task Learning at Scale:** The model was trained on FLD-5B, a huge dataset of _5.4B visual annotations on 126M images_ gathered through automated labeling . These annotations cover a broad spectrum of vision tasks (from object labels and bounding boxes to dense captions and segmentation masks), all converted into a uniform text format. This large-scale multitask training enables Florence-2 to learn fine-grained visual concepts, spatial relationships, and language descriptions simultaneously, achieving a more _general_ and _comprehensive_ understanding of images.

- **Zero-Shot and Fine-Tuning Capabilities:** Thanks to its broad training, Florence-2 demonstrates state-of-the-art performance on many vision benchmarks without needing specialized fine-tuning. It achieves strong _zero-shot results_ on tasks like image captioning and visual grounding, often outperforming much larger models.

## Auto Annotation with Florence-2

One of the standout capabilities of the Florence-2 model is its support for [object detection](https://www.ultralytics.com/glossary/object-detection). Although Florence-2 doesn't natively offer auto annotation via predefined prompts, this functionality is enabled through integration with the Ultralytics Python package.

With this integration, Florence-2 can be used to generate object detection-style annotations using the `<OD>` prompt for general object detection, you can also specify the exact classes you wish to annotate by using `classes` argument.

!!! example "Florence-2 Auto Annotation using Ultralytics Python Package"

    === "All objects"

        ```python
        from ultralytics.data.annotator import AutoAnnotator

        annotator = AutoAnnotator(model="florence-2-base", device="cpu")  # "florence-2-large", "florence-2-base-ft"

        annotator.annotate(
            data="path/to/image/directory or image",  # Input image(s)
            save=True,  # Save .txt annotations
            output_dir="output",  # Output folder
            save_visuals=True,  # Save annotated images
        )
        ```

    === "Specific objects"

        ```python
        from ultralytics.data.annotator import AutoAnnotator

        annotator = AutoAnnotator(model="florence-2-base", device="cpu")  # "florence-2-large", "florence-2-base-ft"

        annotator.annotate(
            data="path/to/image/directory or image",  # Input image(s)
            save=True,  # Save .txt annotations
            output_dir="output",  # Output folder
            save_visuals=True,  # Save annotated images
            classes=["person"],  # Optional: only annotate specific classes
        )
        ```

## Supported Tasks and Model Variants

### Supported Vision Tasks

Florence-2 supports a variety of vision and vision-language tasks through the use of special prompt tokens. By providing the appropriate prompt, users can instruct the model to perform the desired task. The key tasks supported include:

![Florence-2-tasks](https://github.com/ultralytics/docs/releases/download/0/florence-2-tasks.jpg)

1- **Object detection:** Identify objects in an image along with their bounding boxes. Using the `<OD>` prompt, Florence-2 will output a set of object coordinates and labels in a structured format. For example, the model can return a dictionary containing lists of bounding box coordinates and their corresponding class names as shown below:

!!! Results format "Object Detection"

    ```json
    {
        "<OD>": {
            "bboxes": [[x1, y1, x2, y2], ...],
            "labels": ["label1", "label2", ...]
        }
    }
    ```

2- **Image Captioning:** Generate a natural-language description of the entire image. With the `<CAPTION>` prompt, Florence-2 will analyze the image and produce a descriptive caption in English. The model can provide different levels of detail based on the prompt (for example, using `<DETAILED_CAPTION>` yields a more detailed description). This capability is useful for automatic image captioning or summarizing the content of images. It's output format:

!!! Results format "Image Captioning"

    ```json
    {
        "<CAPTION>": 'description of image.'
    }

    ```

3- **Optical Character Recognition (OCR):** Read text that appears within an image. Using the `<OCR>` prompt, Florence-2 will output the text found in the image (for instance, reading signs, documents, or scene text). It can also provide the location of the text: the `<OCR_WITH_REGION>` prompt yields both the recognized text strings and their bounding quadrilaterals (coordinates of corners of the text region) This makes Florence-2 capable of end-to-end scene text recognition, useful for digitizing documents or assisting in understanding text in natural scenes. Output format:

!!! Results format "Optical Character Recognition (OCR)"

    ```json
    {
        "<OCR_WITH_REGION>": {
            "quad_boxes": [[x1, y1, x2, y2], ...],
            "labels": ["label1", "label2", ...]
        }
    }
    ```

4- **Instance Segmentation:** Recognize object instances and delineate their shapes with pixel-wise masks. Florence-2 can produce segmentation results by outputting polygon coordinates for object masks . In practice, a segmentation task can be activated via a prompt `REFERRING_EXPRESSION_SEGMENTATION`, and the model's text output encodes the mask as a series of vertices in the image.

!!! Results format "Instance Segmentation"

    ```json
    {
        "<REFERRING_EXPRESSION_SEGMENTATION>": {
            {'Polygons': [[[polygon]], ...],
            "labels": ["label1", "label2", ...]
        }
    }
    ```

**Note:** Florence-2 also supports other related tasks such as visual grounding (linking phrases to image regions) and dense region captioning (describing multiple regions in an image), by using the corresponding prompts. All outputs are returned in a consistent JSON-like text format. However, the primary tasks highlighted above cover the core object-level, pixel-level, image-level, and text-reading capabilities.

### Model Variants and Checkpoints

Florence-2 is released in multiple pre-trained model variants, as well as fine-tuned checkpoints, to accommodate different use cases and resource constraints. The table below summarizes the available variants:

| **Model Variant**   | **Parameters** | **Description**                                                                                                                                                                                                                       |
| ------------------- | -------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Florence-2-base     |          0.23B | Pretrained base model (232 million params) trained on FLD-5B.                                                                                                                                                                         |
| Florence-2-large    |          0.77B | Pretrained large model (771 million params) with the same FLD-5B training. This model has a larger capacity and typically yields higher accuracy.                                                                                     |
| Florence-2-base-ft  |          0.23B | Fine-tuned base model on a collection of downstream vision tasks . This checkpoint (generalist model) is the base model further trained on various human-annotated datasets for improved performance on detection, segmentation, etc. |
| Florence-2-large-ft |          0.77B | Fine-tuned large model on downstream tasks . This is the most capable variant, combining the large architecture with fine-tuning for strong out-of-the-box results on many benchmarks.                                                |

Each variant is hosted on Hugging Face platform and available in PyTorch format. The **base** vs **large** models mainly differ in size as reported in the [paper](https://arxiv.org/pdf/2311.06242v1), which impacts speed and memory usage.

The fine-tuned (_-ft_) versions have learned from additional supervised data and are recommended if you need better accuracy on standard vision tasks without doing your own fine-tuning. All versions support the full set of tasks through prompting. Users can choose a model variant depending on the desired trade-off between computational cost and accuracy.

## License and Citation

The Florence-2 model is released under the MIT License. The FLD-5B dataset used for pre-training may have its own usage guidelines, but the model weights themselves are MIT licensed.

If you use Florence-2 in your research or project, please cite the original paper “Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks” (Bin Xiao et al., 2023). The authors provide the following BibTeX entry for reference:

!!! quote ""

    === "BibTeX"

        ```bibtex
            @article{xiao2023florence,
              title={Florence-2: Advancing a unified representation for a variety of vision tasks},
              author={Xiao, Bin and Wu, Haiping and Xu, Weijian and Dai, Xiyang and Hu,
                            Houdong and Lu, Yumao and Zeng, Michael and Liu, Ce and Yuan, Lu},
              journal={arXiv preprint arXiv:2311.06242},
              year={2023}
        }
        ```

We would like to acknowledge **Microsoft** for developing and maintaining the Florence-2 model as a powerful and versatile vision foundation model for the research community. For more details about Florence-2 and its capabilities, please refer to the official [Florence-2 paper on arXiv](https://arxiv.org/abs/2311.06242).
