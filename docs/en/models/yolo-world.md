---
comments: true
description: Discover YOLO-World, a YOLOv8-based framework for real-time open-vocabulary object detection in images. It enhances user interaction, boosts computational efficiency, and adapts across various vision tasks.
keywords: YOLO-World, YOLOv8, machine learning, CNN-based framework, object detection, real-time detection, Ultralytics, vision tasks, image processing, industrial applications, user interaction
---

# YOLO-World Model

The YOLO-World Model introduces an advanced, real-time YOLOv8-based approach for Open-Vocabulary Detection tasks. This innovation enables the detection of any object within an image based on descriptive texts. By significantly lowering computational demands while preserving competitive performance, YOLO-World emerges as a versatile tool for numerous vision-based applications.

![YOLO-World Model architecture overview](https://github.com/Laughing-q/assets/assets/61612323/93c8b3cf-baa7-4c32-a405-3744e4ef0f28)

## Overview

YOLO-World tackles the challenges faced by traditional Open-Vocabulary detection models, which often rely on cumbersome Transformer models requiring extensive computational resources. These models' dependence on pre-defined object categories also restricts their utility in dynamic scenarios. YOLO-World revitalizes the YOLO framework with open-vocabulary detection capabilities, employing vision-language modeling and pre-training on expansive datasets to excel at identifying a broad array of objects in zero-shot scenarios with unmatched efficiency.

## Key Features

1. **Real-time Solution:** Harnessing the computational prowess of YOLOv8, YOLO-World delivers a swift open-vocabulary detection solution, catering to industries in need of immediate outcomes.

2. **Efficiency and Performance:** YOLO-World slashes computational and resource requirements without sacrificing performance, offering a robust alternative to models like SAM but at a fraction of the computational cost, enabling real-time applications.

3. **Inference with Offline Vocabulary:** YOLO-World introduces a "prompt-then-detect" strategy, employing an offline vocabulary to enhance efficiency further. This approach enables the use of custom prompts, including captions or categories, to be encoded and stored as offline vocabulary embeddings, streamlining the detection process.

4. **Powered by YOLOv8:** Built upon [YOLOv8](../tasks/detect.md), YOLO-World leverages the latest advancements in real-time object detection to facilitate open-vocabulary detection with unparalleled accuracy and speed.

5. **Benchmark Excellence:** YOLO-World outperforms existing open-vocabulary detectors, including MDETR and GLIP series, in terms of speed and efficiency on standard benchmarks, showcasing its superior capability on a single NVIDIA V100 GPU.

6. **Versatile Applications:** YOLO-World's innovative approach unlocks new possibilities for a multitude of vision tasks, delivering speed improvements by orders of magnitude over existing methods.

## Available Models, Supported Tasks, and Operating Modes

This section details the models available with their specific pre-trained weights, the tasks they support, and their compatibility with various operating modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), denoted by ✅ for supported modes and ❌ for unsupported modes.

!!! Note

    All the YOLOv8-World weights have been directly migrated from the official [YOLO-World](https://github.com/AILab-CVC/YOLO-World) repository, highlighting their excellent contributions.

| Model Type    | Pre-trained Weights                                                                                 | Tasks Supported                        | Inference | Validation | Training | Export |
|---------------|-----------------------------------------------------------------------------------------------------|----------------------------------------|-----------|------------|----------|--------|
| YOLOv8s-world | [yolov8s-world.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s-world.pt) | [Object Detection](../tasks/detect.md) | ✅         | ✅          | ❌        | ❌      |
| YOLOv8m-world | [yolov8m-world.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m-world.pt) | [Object Detection](../tasks/detect.md) | ✅         | ✅          | ❌        | ❌      |
| YOLOv8l-world | [yolov8l-world.pt](https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l-world.pt) | [Object Detection](../tasks/detect.md) | ✅         | ✅          | ❌        | ❌      |

## Zero-shot Transfer on COCO Dataset

| Model Type    | mAP  | mAP50 | mAP75 |
|---------------|------|-------|-------|
| yolov8s-world | 37.4 | 52.0  | 40.6  |
| yolov8m-world | 42.0 | 57.0  | 45.6  |
| yolov8l-world | 45.7 | 61.3  | 49.8  |

## Usage Examples

The YOLO-World models are easy to integrate into your Python applications. Ultralytics provides user-friendly Python API and CLI commands to streamline development.

### Predict Usage

Object detection is straightforward with the `predict` method, as illustrated below:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLOWorld

        # Specify an image for inference
        source = 'path/to/bus.jpg'

        # Initialize a YOLO-World model
        model = YOLOWorld('yolov8s-world.pt')  # or select yolov8m/l-world.pt for different sizes

        # Display model information (optional)
        model.info()

        # Execute inference with the YOLOv8s-world model on the specified image
        results = model.predict(source)
        ```

    === "CLI"

        ```bash
        # Perform object detection using a YOLO-World model
        yolo predict model=yolov8s-world.pt source=path/to/bus.jpg imgsz=640
        ```

This snippet demonstrates the simplicity of loading a pre-trained model and running a prediction on an image.

### Val Usage

Model validation on a dataset is streamlined as follows:

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Create a YOLO-World model
        model = YOLO('yolov8s-world.pt')  # or select yolov8m/l-world.pt for different sizes

        # Conduct model validation on the COCO8 example dataset
        metrics = model.val(data='coco8.yaml')
        ```

    === "CLI"

        ```bash
        # Validate a YOLO-World model on the COCO8 dataset with a specified image size
        yolo val model=yolov8s-world.pt data=coco8.yaml imgsz=640
        ```

!!! Note

    The YOLO-World models provided by Ultralytics come pre-configured with COCO dataset categories as part of their offline vocabulary, enhancing efficiency for immediate application. This integration allows the YOLOv8-World models to directly recognize and predict the 80 standard categories defined in the COCO dataset without requiring additional setup or customization.

### Set prompts

!!! Example

    === "Custom Inference Prompts"

        ```python
        from ultralytics import YOLO

        # Initialize a YOLO-World model
        model = YOLO('yolov8s-world.pt')  # or choose yolov8m/l-world.pt
        
        # Define custom prompts
        prompts = ["person", "bus"]
        model.set_classes(prompts)

        # Execute prediction for specified categories on an image
        results = model.predict('path/to/bus.jpg')
        ```

    === "Persisting Models with Custom Vocabulary"

        ```python
        from ultralytics import YOLO

        # Initialize a YOLO-World model
        model = YOLO('yolov8s-world.pt')  # or select yolov8m/l-world.pt
        
        # Configure custom prompts
        prompts = ["person", "bus"]
        model.set_classes(prompts)

        # Save the model with the defined offline vocabulary
        model.save("path/to/model.pt")
        ```

## Citations and Acknowledgements

We extend our gratitude to the [Tencent AILab](https://ai.tencent.com/) for their pioneering work in real-time open-vocabulary object detection with YOLO-World:

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

For further reading, the original YOLO-World paper is available on [arXiv](https://arxiv.org/pdf/2401.17270v2.pdf). The project's source code and additional resources can be accessed via their [GitHub repository](https://github.com/AILab-CVC/YOLO-World). We appreciate their commitment to advancing the field and sharing their valuable insights with the community.
