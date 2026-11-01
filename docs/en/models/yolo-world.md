---
comments: true
description: YOLO-World is a real-time open-vocabulary object detector built on Ultralytics YOLOv8. Detect objects from text prompts, then train, validate and export it.
keywords: YOLO-World, Ultralytics, open-vocabulary detection, zero-shot detection, YOLOv8, real-time object detection, text prompts, computer vision, model training
---

# YOLO-World Model

The YOLO-World Model is a real-time, [Ultralytics](https://www.ultralytics.com) [YOLOv8](yolov8.md)-based approach to Open-Vocabulary Detection. It detects objects in an image from descriptive text. By lowering computational demands while preserving competitive accuracy, YOLO-World suits a wide range of vision applications.

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

![YOLO-World Model architecture overview](https://cdn.ul.run/i/031930bb2e46a9283632a3c6b6cb671c.avif)

## Overview

YOLO-World addresses the challenges of traditional Open-Vocabulary detection models, which often rely on heavy [Transformer](https://www.ultralytics.com/glossary/transformer) models requiring extensive computational resources. Closed-set detectors, in turn, depend on predefined object categories, which restricts their use in dynamic scenarios. YOLO-World extends the YOLOv8 framework with [open-vocabulary detection](https://www.ultralytics.com/blog/getting-hands-on-with-yolo-world), using vision-[language modeling](https://www.ultralytics.com/glossary/language-modeling) and pre-training on large datasets to identify a broad range of objects in [zero-shot](https://www.ultralytics.com/glossary/zero-shot-learning) scenarios efficiently.

For open-vocabulary work that also needs instance masks, visual prompts, or a prompt-free mode, see [YOLOE](yoloe.md), which keeps the same `set_classes()` API.

## Key Features

1. **Real-time Solution:** Built on fast [CNNs](https://www.ultralytics.com/glossary/convolutional-neural-network-cnn), YOLO-World delivers open-vocabulary detection quickly enough for applications that need immediate results.

2. **Efficiency and Performance:** YOLO-World reduces computational and resource requirements while maintaining accuracy, offering an efficient alternative to heavier transformer-based open-vocabulary detectors and enabling real-time applications.

3. **Inference with Offline Vocabulary:** YOLO-World introduces a "prompt-then-detect" strategy, using an offline vocabulary to further improve efficiency. This approach enables the use of custom prompts computed a priori, including captions or categories, to be encoded and stored as offline vocabulary embeddings, streamlining the detection process.

4. **Powered by YOLOv8:** Built upon Ultralytics YOLOv8, YOLO-World adapts its real-time detection architecture for fast, accurate open-vocabulary detection.

5. **Benchmark Excellence:** YOLO-World outperforms existing open-vocabulary detectors, including MDETR and the GLIP series, in speed and efficiency on standard benchmarks, measured on a single NVIDIA V100 GPU.

6. **Versatile Applications:** YOLO-World's efficiency makes open-vocabulary detection practical for many vision tasks, running much faster than earlier open-vocabulary methods.

## Available Models, Supported Tasks, and Operating Modes

This section details the models available with their specific pretrained weights, the tasks they support, and their compatibility with various operating modes such as [Training](../modes/train.md), [Validation](../modes/val.md), [Inference](../modes/predict.md), and [Export](../modes/export.md), denoted by ✅ for supported modes and ❌ for unsupported modes.

!!! note

    All YOLOv8-World weights are migrated directly from the official [YOLO-World](https://github.com/AILab-CVC/YOLO-World) repository; credit for them goes to its authors.

| Model Type      | Pretrained Weights                                                                                      | Tasks Supported                        | Training | Validation | Inference | Export |
| --------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLOv8s-world   | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8s-worldv2 | [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv8m-world   | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8m-worldv2 | [yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv8l-world   | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8l-worldv2 | [yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv8x-world   | [yolov8x-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8x-worldv2 | [yolov8x-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |

## Zero-shot Transfer on COCO Dataset

!!! tip "Performance"

    === "Detection (COCO)"

        | Model Type      | mAP      | mAP<sub>50</sub>    | mAP<sub>75</sub>    |
        | --------------- | -------- | -------- | -------- |
        | yolov8s-world   | 37.4     | 52.0     | 40.6     |
        | yolov8s-worldv2 | 37.7     | 52.2     | 41.0     |
        | yolov8m-world   | 42.0     | 57.0     | 45.6     |
        | yolov8m-worldv2 | 43.0     | 58.4     | 46.8     |
        | yolov8l-world   | 45.7     | 61.3     | 49.8     |
        | yolov8l-worldv2 | 45.8     | 61.3     | 49.8     |
        | yolov8x-world   | 47.0     | **63.0** | 51.2     |
        | yolov8x-worldv2 | **47.1** | 62.8     | **51.4** |

## Usage Examples

Ultralytics provides a [Python API](../usage/python.md) and [CLI commands](../usage/cli.md) for YOLO-World models.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/sWEm3dIGKU8"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> YOLO-World Model Usage examples with Ultralytics | Open Vocab, Prompt-Free & others
</p>

### Train Usage

!!! tip

    We strongly recommend using `yolov8-worldv2` for custom training because it supports deterministic training and exports more easily to formats such as ONNX and TensorRT.

Train a YOLO-World [object detection](https://www.ultralytics.com/glossary/object-detection) model with the `train` method, shown below on the [COCO8](https://platform.ultralytics.com/ultralytics/datasets/coco8) example dataset:

!!! example

    === "Python"

        [PyTorch](https://www.ultralytics.com/glossary/pytorch) pretrained `*.pt` models as well as configuration `*.yaml` files can be passed to the `YOLOWorld()` class to create a model instance in Python:

        ```python
        from ultralytics import YOLOWorld

        # Load a pretrained YOLOv8s-worldv2 model
        model = YOLOWorld("yolov8s-worldv2.pt")

        # Train the model on the COCO8 example dataset for 100 epochs
        results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

        # Run inference with the YOLO-World model on the 'bus.jpg' image
        results = model("path/to/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Load a pretrained YOLOv8s-worldv2 model and train it on the COCO8 example dataset for 100 epochs
        yolo train model=yolov8s-worldv2.pt data=coco8.yaml epochs=100 imgsz=640
        ```

### Predict Usage

Run object detection with the `predict` method, as shown below:

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

This snippet loads a pretrained model and runs prediction on an image.

### Val Usage

Validate the model on a dataset as follows:

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

Run [object tracking](https://www.ultralytics.com/glossary/object-tracking) with a YOLO-World model on videos or images as follows:

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
        yolo track model=yolov8s-world.pt imgsz=640 source="path/to/video.mp4"
        ```

!!! note

    The YOLO-World models provided by Ultralytics come pre-configured with [COCO dataset](../datasets/detect/coco.md) categories as part of their offline vocabulary, so they are ready to use immediately. This allows the YOLOv8-World models to directly recognize and predict the 80 standard categories defined in the COCO dataset without requiring additional setup or customization.

### Set prompts

![YOLO-World prompt class names overview](https://cdn.ul.run/i/ed01ad040094f45b36920515dfe13245.avif)

The YOLO-World framework allows dynamic specification of classes through custom prompts, letting users tailor the model to their specific needs **without retraining**. This feature is particularly useful for adapting the model to new domains or specific tasks that were not originally part of the [training data](https://www.ultralytics.com/glossary/training-data). By setting custom prompts, users guide the model toward objects of interest, improving the relevance and [accuracy](https://www.ultralytics.com/glossary/accuracy) of the detection results.

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

!!! note "Background Class"

    Some users have found that appending an empty string `""` as a background class can improve detection performance in certain scenarios. This behavior appears to be scenario-dependent and the exact mechanism is not fully understood:

    ```python
    model.set_classes(["person", "bus", ""])
    ```

    The empty string stays in `model.names` as a class of its own, including in any model you save, so its detections come back with an empty label. Pass `classes=[0, 1]` at prediction time to keep only your real classes.

You can also save a model after setting custom classes. By doing this you create a version of the YOLO-World model that is specialized for your specific use case. This process embeds your custom class definitions directly into the model file, making the model ready to use with your specified classes without further adjustments. Follow these steps to save and load your custom YOLO-World model:

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

        After saving, the custom_yolov8s.pt model behaves like any other pretrained YOLOv8 model but with a key difference: it is now optimized to detect only the classes you have defined. This customization can significantly improve detection performance and efficiency for your specific application scenarios.

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
- **Performance**: Enhances detection accuracy for specified classes by focusing the model's attention and resources on recognizing the defined objects.

This approach customizes [object detection](../tasks/detect.md) models for specific tasks without retraining, making open-vocabulary detection practical for a broader range of applications.

## Reproduce official results from scratch (Experimental)

### Prepare datasets

- Train data

| Dataset                                                           | Type                                                        | Samples | Boxes | Annotation Files                                                                                                                              |
| ----------------------------------------------------------------- | ----------------------------------------------------------- | ------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| [Objects365v1](https://opendatalab.com/OpenDataLab/Objects365_v1) | Detection                                                   | 609k    | 9621k | [objects365_train.json](https://opendatalab.com/OpenDataLab/Objects365_v1)                                                                    |
| [GQA](https://cs.stanford.edu/people/dorarad/gqa/about.html)      | [Grounding](https://www.ultralytics.com/glossary/grounding) | 621k    | 3681k | [final_mixed_train_no_coco.json](https://huggingface.co/GLIPModel/GLIP/resolve/main/mdetr_annotations/final_mixed_train_no_coco.json)         |
| [Flickr30k](https://github.com/BryanPlummer/flickr30k_entities)   | Grounding                                                   | 149k    | 641k  | [final_flickr_separateGT_train.json](https://huggingface.co/GLIPModel/GLIP/resolve/main/mdetr_annotations/final_flickr_separateGT_train.json) |

- Val data

| Dataset                                                                                                 | Type      | Annotation Files                                                                                       |
| ------------------------------------------------------------------------------------------------------- | --------- | ------------------------------------------------------------------------------------------------------ |
| [LVIS minival](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) | Detection | [minival.txt](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/lvis.yaml) |

### Launch training from scratch

!!! note

    [`WorldTrainerFromScratch`](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/models/yolo/world/train_world.py) is highly customized to allow training YOLO-World models on both detection datasets and grounding datasets simultaneously.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOWorld
        from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

        # Option 1: Use Python dictionary
        data = {
            "train": {
                "yolo_data": ["Objects365.yaml"],
                "grounding_data": [
                    {
                        "img_path": "flickr30k/images",
                        "json_file": "flickr30k/final_flickr_separateGT_train.json",
                    },
                    {
                        "img_path": "GQA/images",
                        "json_file": "GQA/final_mixed_train_no_coco.json",
                    },
                ],
            },
            "val": {"yolo_data": ["lvis.yaml"]},
        }

        # Option 2: Use YAML file (yolo_world_data.yaml)
        # train:
        #   yolo_data:
        #     - Objects365.yaml
        #   grounding_data:
        #     - img_path: flickr/full_images/
        #       json_file: flickr/annotations/final_flickr_separateGT_train_segm.json
        #     - img_path: mixed_grounding/gqa/images
        #       json_file: mixed_grounding/annotations/final_mixed_train_no_coco_segm.json
        # val:
        #   yolo_data:
        #     - lvis.yaml

        model = YOLOWorld("yolov8s-worldv2.yaml")
        model.train(
            data=data,  # or data="yolo_world_data.yaml" if using YAML file
            batch=128,
            epochs=100,
            trainer=WorldTrainerFromScratch,
        )
        ```

## Citations and Acknowledgments

We extend our gratitude to the [Tencent AILab Computer Vision Center](https://www.tencent.com/) for their pioneering work in real-time open-vocabulary object detection with YOLO-World:

!!! quote "Citation"

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

The YOLO-World model is a real-time object detection approach based on the [Ultralytics YOLOv8](yolov8.md) framework. It performs Open-Vocabulary Detection by identifying objects within an image based on descriptive text. Using vision-language modeling and pre-training on large datasets, YOLO-World achieves high efficiency with reduced computational demands, making it well suited for real-time applications.

### How does YOLO-World handle inference with custom prompts?

YOLO-World supports a "prompt-then-detect" strategy, which uses an offline vocabulary to improve efficiency. Custom prompts like captions or specific object categories are pre-encoded and stored as offline vocabulary [embeddings](https://www.ultralytics.com/glossary/embeddings). This approach streamlines the detection process without the need for retraining. You can dynamically set these prompts within the model to tailor it to specific detection tasks, as shown below:

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

- **Real-Time Performance:** It uses fast CNNs for quick, efficient detection.
- **Efficiency and Low Resource Requirement:** YOLO-World maintains high performance while significantly reducing computational and resource demands.
- **Customizable Prompts:** The model supports dynamic prompt setting, allowing users to specify custom detection classes without retraining.
- **Benchmark Excellence:** It outperforms other open-vocabulary detectors like MDETR and GLIP in both speed and efficiency on standard benchmarks.

### How do I train a YOLO-World model on my dataset?

Train a YOLO-World model on your dataset with the Python API or CLI. Here's how to start training using Python:

```python
from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8s-worldv2.pt")

# Train the model on the COCO8 dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

Or using CLI:

```bash
yolo train model=yolov8s-worldv2.pt data=coco8.yaml epochs=100 imgsz=640
```

### What are the available pretrained YOLO-World models and their supported tasks?

Ultralytics offers multiple pretrained YOLO-World models supporting various tasks and operating modes:

| Model Type      | Pretrained Weights                                                                                      | Tasks Supported                        | Training | Validation | Inference | Export |
| --------------- | ------------------------------------------------------------------------------------------------------- | -------------------------------------- | -------- | ---------- | --------- | ------ |
| YOLOv8s-world   | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8s-worldv2 | [yolov8s-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv8m-world   | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8m-worldv2 | [yolov8m-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv8l-world   | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8l-worldv2 | [yolov8l-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8l-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |
| YOLOv8x-world   | [yolov8x-world.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-world.pt)     | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ❌     |
| YOLOv8x-worldv2 | [yolov8x-worldv2.pt](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8x-worldv2.pt) | [Object Detection](../tasks/detect.md) | ✅       | ✅         | ✅        | ✅     |

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
                "img_path": "flickr30k/images",
                "json_file": "flickr30k/final_flickr_separateGT_train.json",
            },
            {
                "img_path": "GQA/images",
                "json_file": "GQA/final_mixed_train_no_coco.json",
            },
        ],
    },
    "val": {"yolo_data": ["lvis.yaml"]},
}

model = YOLOWorld("yolov8s-worldv2.yaml")
model.train(data=data, batch=128, epochs=100, trainer=WorldTrainerFromScratch)
```
