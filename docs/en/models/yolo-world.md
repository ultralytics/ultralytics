---
comments: true
description: Explore the YOLO-World Model for efficient, real-time open-vocabulary object detection using Ultralytics YOLOv8 advancements. Achieve top performance with minimal computation.
keywords: YOLO-World, Ultralytics, open-vocabulary detection, YOLOv8, real-time object detection, machine learning, computer vision, AI, deep learning, model training
---

# YOLO-World Model

The YOLO-World Model introduces an advanced, real-time [Ultralytics](https://www.ultralytics.com/) [YOLOv8](yolov8.md)-based approach for Open-Vocabulary Detection tasks. This innovation enables the detection of any object within an image based on descriptive texts. By significantly lowering computational demands while preserving competitive performance, YOLO-World emerges as a versatile tool for numerous vision-based applications.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/cfTKj96TjSE"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> YOLO World training workflow on custom dataset
</p>

![YOLO-World Model architecture overview](https://github.com/ultralytics/docs/releases/download/0/yolo-world-model-architecture-overview.avif)

## Overview

YOLO-World tackles the challenges faced by traditional Open-Vocabulary detection models, which often rely on cumbersome [Transformer](https://www.ultralytics.com/glossary/transformer) models requiring extensive computational resources. These models' dependence on pre-defined object categories also restricts their utility in dynamic scenarios. YOLO-World revitalizes the YOLOv8 framework with open-vocabulary detection capabilities, employing vision-[language modeling](https://www.ultralytics.com/glossary/language-modeling) and pre-training on expansive datasets to excel at identifying a broad array of objects in zero-shot scenarios with unmatched efficiency.

## Key Features

1. **Real-time Solution:** Harnessing the computational speed of CNNs, YOLO-World delivers a swift open-vocabulary detection solution, catering to industries in need of immediate results.

2. **Efficiency and Performance:** YOLO-World slashes computational and resource requirements without sacrificing performance, offering a robust alternative to models like SAM but at a fraction of the computational cost, enabling real-time applications.

3. **Inference with Offline Vocabulary:** YOLO-World introduces a "prompt-then-detect" strategy, employing an offline vocabulary to enhance efficiency further. This approach enables the use of custom prompts computed apriori, including captions or categories, to be encoded and stored as offline vocabulary embeddings, streamlining the detection process.

4. **Powered by YOLOv8:** Built upon [Ultralytics YOLOv8](yolov8.md), YOLO-World leverages the latest advancements in real-time object detection to facilitate open-vocabulary detection with unparalleled accuracy and speed.

5. **Benchmark Excellence:** YOLO-World outperforms existing open-vocabulary detectors, including MDETR and GLIP series, in terms of speed and efficiency on standard benchmarks, showcasing YOLOv8's superior capability on a single NVIDIA V100 GPU.

6. **Versatile Applications:** YOLO-World's innovative approach unlocks new possibilities for a multitude of vision tasks, delivering speed improvements by orders of magnitude over existing methods.

## Available Models, Supported Tasks, and Operating Modes

This section details the models available with their specific pre-trained weights, the tasks they support, and their compatibility with various operating modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), denoted by ✅ for supported modes and ❌ for unsupported modes.

!!! note

    All the YOLOv8-World weights have been directly migrated from the official [YOLO-World](https://github.com/AILab-CVC/YOLO-World) repository, highlighting their excellent contributions.

| Model Type      | Pre-trained Weights                                                                                     | Tasks Supported                        | Inference | Validation | Training | Export |
| --------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLOv8s-world   | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8s-worldv2 | [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOv8m-world   | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8m-worldv2 | [yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOv8l-world   | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8l-worldv2 | [yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOv8x-world   | [yolov8x-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8x-worldv2 | [yolov8x-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

## Zero-shot Transfer on COCO Dataset

| Model Type      | mAP  | mAP50 | mAP75 |
| --------------- | ---- | ----- | ----- |
| yolov8s-world   | 37.4 | 52.0  | 40.6  |
| yolov8s-worldv2 | 37.7 | 52.2  | 41.0  |
| yolov8m-world   | 42.0 | 57.0  | 45.6  |
| yolov8m-worldv2 | 43.0 | 58.4  | 46.8  |
| yolov8l-world   | 45.7 | 61.3  | 49.8  |
| yolov8l-worldv2 | 45.8 | 61.3  | 49.8  |
| yolov8x-world   | 47.0 | 63.0  | 51.2  |
| yolov8x-worldv2 | 47.1 | 62.8  | 51.4  |

## Usage Examples

The YOLO-World models are easy to integrate into your Python applications. Ultralytics provides user-friendly Python API and CLI commands to streamline development.

### Train Usage

!!! tip

    We strongly recommend to use `yolov8-worldv2` model for custom training, because it supports deterministic training and also easy to export other formats i.e onnx/tensorrt.

[Object detection](https://www.ultralytics.com/glossary/object-detection) is straightforward with the `train` method, as illustrated below:

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLOWorld()` class to create a model instance in python:

        ```python
        from ultralytics import YOLOWorld

        # Load a pretrained YOLOv8s-worldv2 model
        model = YOLOWorld("yolov8s-worldv2.pt")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLOv8n model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Load a pretrained YOLOv8s-worldv2 model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov8s-worldv2.yaml data=coco8.yaml epochs=100 imgsz=640
        ```

### Predict Usage

Object detection is straightforward with the `predict` method, as illustrated below:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOWorld

        # Initialize a YOLO-World model
        model = YOLOWorld("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

        # Execute inference with the YOLOv8s-world model on the specified image
        results = model.predict("path/to/image.jpg")

        # Show results
        results[0].show()
        ```

    === "CLI"

        ```bash
        # Perform object detection using a YOLO-World model
        yolo predict model=yolov8s-world.pt source=path/to/image.jpg imgsz=640
        ```

This snippet demonstrates the simplicity of loading a pre-trained model and running a prediction on an image.

### Val Usage

Model validation on a dataset is streamlined as follows:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a YOLO-World model
        model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

        # Conduct model validation on the COCO8 example dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate a YOLO-World model on the COCO8 dataset with a specified image size
        yolo val model=yolov8s-world.pt data=coco8.yaml imgsz=640
        ```

### Track Usage

Object tracking with YOLO-World model on a video/images is streamlined as follows:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a YOLO-World model
        model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

        # Track with a YOLO-World model on a video
        results = model.track(source="path/to/video.mp4")
        ```

    === "CLI"

        ```bash
        # Track with a YOLO-World model on the video with a specified image size
        yolo track model=yolov8s-world.pt imgsz=640 source="path/to/video/file.mp4"
        ```

!!! note

    The YOLO-World models provided by Ultralytics come pre-configured with [COCO dataset](../datasets/detect/coco.md) categories as part of their offline vocabulary, enhancing efficiency for immediate application. This integration allows the YOLOv8-World models to directly recognize and predict the 80 standard categories defined in the COCO dataset without requiring additional setup or customization.

### Set prompts

![YOLO-World prompt class names overview](https://github.com/ultralytics/docs/releases/download/0/yolo-world-prompt-class-names-overview.avif)

The YOLO-World framework allows for the dynamic specification of classes through custom prompts, empowering users to tailor the model to their specific needs **without retraining**. This feature is particularly useful for adapting the model to new domains or specific tasks that were not originally part of the [training data](https://www.ultralytics.com/glossary/training-data). By setting custom prompts, users can essentially guide the model's focus towards objects of interest, enhancing the relevance and accuracy of the detection results.

For instance, if your application only requires detecting 'person' and 'bus' objects, you can specify these classes directly:

!!! example

    === "Custom Inference Prompts"

        ```python
        from ultralytics import YOLO

        # Initialize a YOLO-World model
        model = YOLO("yolov8s-world.pt")  # or choose yolov8m/l-world.pt

        # Define custom classes
        model.set_classes(["person", "bus"])

        # Execute prediction for specified categories on an image
        results = model.predict("path/to/image.jpg")

        # Show results
        results[0].show()
        ```

You can also save a model after setting custom classes. By doing this you create a version of the YOLO-World model that is specialized for your specific use case. This process embeds your custom class definitions directly into the model file, making the model ready to use with your specified classes without further adjustments. Follow these steps to save and load your custom YOLOv8 model:

!!! example

    === "Persisting Models with Custom Vocabulary"

        First load a YOLO-World model, set custom classes for it and save it:

        ```python
        from ultralytics import YOLO

        # Initialize a YOLO-World model
        model = YOLO("yolov8s-world.pt")  # or select yolov8m/l-world.pt

        # Define custom classes
        model.set_classes(["person", "bus"])

        # Save the model with the defined offline vocabulary
        model.save("custom_yolov8s.pt")
        ```

        After saving, the custom_yolov8s.pt model behaves like any other pre-trained YOLOv8 model but with a key difference: it is now optimized to detect only the classes you have defined. This customization can significantly improve detection performance and efficiency for your specific application scenarios.

        ```python
        from ultralytics import YOLO

        # Load your custom model
        model = YOLO("custom_yolov8s.pt")

        # Run inference to detect your custom classes
        results = model.predict("path/to/image.jpg")

        # Show results
        results[0].show()
        ```

### Benefits of Saving with Custom Vocabulary

- **Efficiency**: Streamlines the detection process by focusing on relevant objects, reducing computational overhead and speeding up inference.
- **Flexibility**: Allows for easy adaptation of the model to new or niche detection tasks without the need for extensive retraining or data collection.
- **Simplicity**: Simplifies deployment by eliminating the need to repeatedly specify custom classes at runtime, making the model directly usable with its embedded vocabulary.
- **Performance**: Enhances detection [accuracy](https://www.ultralytics.com/glossary/accuracy) for specified classes by focusing the model's attention and resources on recognizing the defined objects.

This approach provides a powerful means of customizing state-of-the-art object detection models for specific tasks, making advanced AI more accessible and applicable to a broader range of practical applications.

## Reproduce official results from scratch(Experimental)

### Prepare datasets

- Train data

| Dataset                                                           | Type      | Samples | Boxes | Annotation Files                                                                                                                           |
| ----------------------------------------------------------------- | --------- | ------- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| [Objects365v1](https://opendatalab.com/OpenDataLab/Objects365_v1) | Detection | 609k    | 9621k | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1)                                                                 |
| [GQA](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)  | Grounding | 621k    | 3681k | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_mixed_train_no_coco.json)         |
| [Flickr30k](https://shannon.cs.illinois.edu/DenotationGraph/)     | Grounding | 149k    | 641k  | [final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/blob/main/mdetr_annotations/final_flickr_separateGT_train.json) |

- Val data

| Dataset                                                                                                 | Type      | Annotation Files                                                                                       |
| ------------------------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| [LVIS minival](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) | Detection | [minival.txt](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) |

### Launch training from scratch

!!! note

    `WorldTrainerFromScratch` is highly customized to allow training yolo-world models on both detection datasets and grounding datasets simultaneously. More details please checkout [ultralytics.model.yolo.world.train_world.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py).

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOWorld
        from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="../datasets/flickr30k/images",
                        json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                    ),
                    dict(
                        img_path="../datasets/GQA/images",
                        json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )
        model = YOLOWorld("yolov8s-worldv2.yaml")
        model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)
        ```

## Citations and Acknowledgements

We extend our gratitude to the [Tencent AILab Computer Vision Center](https://www.tencent.com/) for their pioneering work in real-time open-vocabulary object detection with YOLO-World:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @article{cheng2024yolow,
        title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
        author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
        journal={arXiv preprint arXiv:2401.17270},
        year={2024}
        }
        ```

For further reading, the original YOLO-World paper is available on [arXiv](https://arxiv.org/pdf/2401.17270v2). The project's source code and additional resources can be accessed via their [GitHub repository](https://github.com/AILab-CVC/YOLO-World). We appreciate their commitment to advancing the field and sharing their valuable insights with the community.

## FAQ

### What is the YOLO-World model and how does it work?

The YOLO-World model is an advanced, real-time object detection approach based on the [Ultralytics YOLOv8](yolov8.md) framework. It excels in Open-Vocabulary Detection tasks by identifying objects within an image based on descriptive texts. Using vision-language modeling and pre-training on large datasets, YOLO-World achieves high efficiency and performance with significantly reduced computational demands, making it ideal for real-time applications across various industries.

### How does YOLO-World handle inference with custom prompts?

YOLO-World supports a "prompt-then-detect" strategy, which utilizes an offline vocabulary to enhance efficiency. Custom prompts like captions or specific object categories are pre-encoded and stored as offline vocabulary [embeddings](https://www.ultralytics.com/glossary/embeddings). This approach streamlines the detection process without the need for retraining. You can dynamically set these prompts within the model to tailor it to specific detection tasks, as shown below:

```python
from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("yolov8s-world.pt")

# Define custom classes
model.set_classes(["person", "bus"])

# Execute prediction on an image
results = model.predict("path/to/image.jpg")

# Show results
results[0].show()
```

### Why should I choose YOLO-World over traditional Open-Vocabulary detection models?

YOLO-World provides several advantages over traditional Open-Vocabulary detection models:

- **Real-Time Performance:** It leverages the computational speed of CNNs to offer quick, efficient detection.
- **Efficiency and Low Resource Requirement:** YOLO-World maintains high performance while significantly reducing computational and resource demands.
- **Customizable Prompts:** The model supports dynamic prompt setting, allowing users to specify custom detection classes without retraining.
- **Benchmark Excellence:** It outperforms other open-vocabulary detectors like MDETR and GLIP in both speed and efficiency on standard benchmarks.

### How do I train a YOLO-World model on my dataset?

Training a YOLO-World model on your dataset is straightforward through the provided Python API or CLI commands. Here's how to start training using Python:

```python
from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8s-worldv2.pt")

# Train the model on the COCO8 dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

Or using CLI:

```bash
yolo train model=yolov8s-worldv2.yaml data=coco8.yaml epochs=100 imgsz=640
```

### What are the available pre-trained YOLO-World models and their supported tasks?

Ultralytics offers multiple pre-trained YOLO-World models supporting various tasks and operating modes:

| Model Type      | Pre-trained Weights                                                                                     | Tasks Supported                        | Inference | Validation | Training | Export |
| --------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------- | --------- | ---------- | -------- | ------ |
| YOLOv8s-world   | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8s-worldv2 | [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOv8m-world   | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8m-worldv2 | [yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOv8l-world   | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8l-worldv2 | [yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOv8x-world   | [yolov8x-world.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-world.pt)     | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ❌     |
| YOLOv8x-worldv2 | [yolov8x-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

### How do I reproduce the official results of YOLO-World from scratch?

To reproduce the official results from scratch, you need to prepare the datasets and launch the training using the provided code. The training procedure involves creating a data dictionary and running the `train` method with a custom trainer:

```python
from ultralytics import YOLOWorld
from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

data = {
    "train": {
        "yolo_data": ["Objects365.yaml"],
        "grounding_data": [
            {
                "img_path": "../datasets/flickr30k/images",
                "json_file": "../datasets/flickr30k/final_flickr_separateGT_train.json",
            },
            {
                "img_path": "../datasets/GQA/images",
                "json_file": "../datasets/GQA/final_mixed_train_no_coco.json",
            },
        ],
    },
    "val": {"yolo_data": ["lvis.yaml"]},
}

model = YOLOWorld("yolov8s-worldv2.yaml")
model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)
```
