---
comments: true
description: Explore YOLO-World, a CNN-based solution for real-time open-vocabulary object detection in images. Enhanced user interaction, computational efficiency and adaptable across vision tasks.
keywords: YOLO-World, YOLOv8, machine learning, CNN-based solution, object detection, real-time solution, Ultralytics, vision tasks, image processing, industrial applications, user interaction
---

# YOLO-World Model

The YOLO-World Model is a novel, real-time CNN-based solution for the Open-Vocabulary Detection task. This task is designed to detect any object within an image based on given texts. YOLO-World significantly reduces computational demands while maintaining competitive performance, making it a practical choice for a variety of vision tasks.

![YOLO-World Model architecture overview](https://github.com/Laughing-q/assets/assets/61612323/93c8b3cf-baa7-4c32-a405-3744e4ef0f28)

## Overview

YOLO-World is designed to address the limitations of current Open-Vocabulary detection models, usually with a heavy Transformer model with substantial computational resource requirements. Also their reliance on predefined and trained object categories limits their applicability in open scenarios. The YOLO-World is an innovative approach that enhances YOLO with open-vocabulary detection capabilities through vision-language modeling and pre-training on large-scale datasets, excels in detecting a wide range of objects in a zero-shot manner with high efficiency.

## Key Features

1. **Real-time Solution:** By leveraging the computational efficiency of YOLOv8, YOLO-World provides a real-time solution for the open-vocabulary detection task, making it valuable for industrial applications that require quick results.

2. **Efficiency and Performance:** YOLO-World offers a significant reduction in computational and resource demands without compromising on performance quality. It achieves comparable performance to SAM but with drastically reduced computational resources, enabling real-time application.

3. **Inference with Offline Vocabulary:** At the inference stage, YOLO-World present a prompt-then-detect strategy with an offline vocabulary for further efficiency. Users can define a series of custom prompts, which might include captions or categories. The model then utilize the text encoder to encode these prompts and obtain offline vocabulary embeddings. The offline vocabulary allows for avoiding computation for each input and provides the flexibility to adjust the vocabulary as needed.

4. **Based on YOLOv8:** YOLO-World is based on [YOLOv8](../tasks/detect.md), an real-time object detector offering cutting-edge performance in terms of accuracy and speed. This allows it to effectively produce open-vocabulary detection in an image.

5. **Competitive Results on Benchmarks:** On the object proposal task on LVIS minival, YOLO-World achieves high scores at a significantly faster speed than other open-vocabulary detectors(i.e MDETR, GLIP series, Grounding DINO models) on a single NVIDIA V100, demonstrating its efficiency and capability.

6. **Practical Applications:** The proposed approach provides a new, practical solution for a large number of vision tasks at a really high speed, tens or hundreds of times faster than current methods.

## Available Models, Supported Tasks, and Operating Modes

This table presents the available models with their specific pre-trained weights, the tasks they support, and their compatibility with different operating modes like [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), indicated by ✅ emojis for supported modes and ❌ emojis for unsupported modes.

| Model Type | Pre-trained Weights                                                                         | Tasks Supported                              | Inference | Validation | Training | Export |
|------------|---------------------------------------------------------------------------------------------|----------------------------------------------|-----------|------------|----------|--------|
| yolov8s-world  | [yolov8s-world](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-world.pt) | [Object Detection](../tasks/detect.md) | ✅         | ✅          | ❌        | ❌      |
| yolov8m-world  | [yolov8m-world](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-world.pt) | [Object Detection](../tasks/detect.md) | ✅         | ✅          | ❌        | ❌      |
| yolov8l-world  | [yolov8l-world](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt) | [Object Detection](../tasks/detect.md) | ✅         | ✅          | ❌        | ❌      |

## Usage Examples

The YOLO-World models are easy to integrate into your Python applications. Ultralytics provides user-friendly Python API and CLI commands to streamline development.

### Predict Usage

To perform object detection on an image, use the `predict` method as shown below:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLOWorld

        # Define an inference source
        source = 'path/to/bus.jpg'

        # Create a YOLO-World model
        model = YOLOWorld('yolov8s-world.pt')  # or yolov8m/l-world.pt

        # Display model information (optional)
        model.info()

        # Run inference with the YOLOv8s-world model on the 'bus.jpg' image
        results = model('path/to/bus.jpg')

        ```

    === "CLI"

        ```bash
        # Load a YOLO-World model and segment everything with it
        yolo detect predict model=yolov8s-world.pt source=path/to/bus.jpg imgsz=640
        ```

This snippet demonstrates the simplicity of loading a pre-trained model and running a prediction on an image.

### Val Usage

Validation of the model on a dataset can be done as follows:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLOWorld

        # Create a YOLO-World model
        model = YOLOWorld('yolov8s-world.pt')  # or yolov8m/l-world.pt

        # Validate the model on the COCO8 example dataset for 100 epochs
        metrics = model.val(data='coco8.yaml')
        ```

    === "CLI"

        ```bash
        # Load a YOLO-World model and validate it on the COCO8 example dataset at image size 640
        yolo detect val model=yolov8s-world.pt data=coco8.yaml imgsz=640
        ```

!!! Tip "Tip"

    The YOLO-World weights in Ultralytics has already embedded COCO categories as offline vocabulary for further efficiency, which means yolov8-world models can be used directly to predict 80 categories of COCO.

### Set prompts

!!! Example

    === "Inference with new prompts"

        ```python
        from ultralytics import YOLOWorld

        # Create a YOLO-World model
        model = YOLOWorld('yolov8s-world.pt')  # or yolov8m/l-world.pt
        
        # Setup prompts
        prompts = ["person", "bus"]
        model.set_classes(prompts)

        # Predict the categories of person and bus on the 'bus.jpg' image
        results = model('path/to/bus.jpg')
        ```

    === "Save model with offline vocabulary"

        ```python
        model = YOLOWorld('yolov8s-world.pt')  # or yolov8m/l-world.pt
        
        # Setup prompts
        prompts = ["person", "bus"]
        model.set_classes(prompts)

        # Save model with offline vocabulary ["person", "bus"]
        model.save("path/to/model.pt")
        ```


## Citations and Acknowledgements

We would like to acknowledge the YOLO-World authors for their significant contributions in the field of real-time open-vocabulary object detection:

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @article{cheng2024yolow,
        title={YOLO-World: Real-Time Open-Vocabulary Object Detection},
        author={Cheng, Tianheng and Song, Lin and Ge, Yixiao and Liu, Wenyu and Wang, Xinggang and Shan, Ying},
        journal={arXiv preprint arXiv:2401.17270},
        year={2024}
      }
      ```

The original YOLO-World paper can be found on [arXiv](https://arxiv.org/pdf/2401.17270v2.pdf). The authors have made their work publicly available, and the codebase can be accessed on [GitHub](https://github.com/AILab-CVC/YOLO-World). We appreciate their efforts in advancing the field and making their work accessible to the broader community.
