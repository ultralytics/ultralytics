---
comments: true
description: Explore detailed documentation of YOLO-Human, an advanced human attribute detection model. Learn about its features, configurations, usage with the Ultralytics Python API, and more.
keywords: YOLO-Human, Ultralytics, human attribute estimation, deep learning, YOLOv8, human-centric datasets, Ultralytics Python API, model training, model prediction, augmentation, visualization, metrics
---

# YOLO-Human Model

<img width="1024" src="https://github.com/ultralytics/ultralytics/assets/3855193/c49d8b56-aed6-4303-82b2-790aa24b5515" alt="Object detection examples">

## Overview
The YOLO-Human model by Ultralytics is designed for advanced human attribute estimation, enhancing the existing YOLOv8 capabilities. This model can estimate various human attributes, including age, biological gender, ethnicity, weight, and height. It is capable of detecting individual humans as well as flocks of humans, making it ideal for applications in demographic analysis, security, and personalized customer experiences.

## Key Features
- **Real-time Human Attribute Estimation**: Leverages the computational speed of CNNs to provide fast and accurate human attribute estimation in real-time.
- **Efficiency and Performance**: Optimized for reduced computational and resource requirements without sacrificing performance, enabling deployment in real-time applications.
- **Comprehensive Attribute Estimation**: Capable of estimating multiple human attributes such as age, biological gender, ethnicity, weight, and height, providing detailed demographic analysis.
- **Detection of Group of Humans**: Enhanced to detect both individual humans and groups of humans, expanding its applicability in various scenarios.
- **Powered by YOLOv8**: Built upon the advanced YOLOv8 architecture, ensuring state-of-the-art performance in human attribute detection.
- **Advanced Augmentations**: Includes enhanced augmentations specifically designed for human attribute data, improving model robustness and accuracy.
- **Detailed Visualization**: Provides comprehensive visualization tools to display detected human attributes effectively.
- **Robust Evaluation Metrics**: Implements new metrics tailored for human attribute detection, ensuring thorough model evaluation and validation.

???+ warning "Disclaimer on Privacy and Security"

    When using YOLO-Human models for human detection and attribute estimation, please be mindful of privacy and security. Ensure compliance with relevant data protection laws and implement strong security measures. Use these models ethically, avoiding applications that could cause harm or discrimination. Ultralytics is not responsible for any misuse. Users are responsible for ensuring their use complies with all legal and ethical standards.


## Available Models, Supported Tasks, and Operating Modes

This section details the models available with their specific pre-trained weights, the tasks they support, and their compatibility with various operating modes such as [Inference](../modes/predict.md), [Validation](../modes/val.md), [Training](../modes/train.md), and [Export](../modes/export.md), denoted by ✅ for supported modes and ❌ for unsupported modes. 

!!! note

    It is important to note that these models have been trained on a specially curated, artificially annotated version of the COCO dataset. This custom dataset was meticulously crafted to enhance the models' performance on specific tasks by incorporating additional annotations and adjustments beyond those available in the public COCO dataset. This enhanced version of the dataset is not publicly available (a sample of the dataset is available at [this page](../datasets/human/coco8-human.md). The artificial annotations were designed to provide more comprehensive and nuanced data, enabling the models to achieve higher accuracy and robustness in their predictions.

| Model Type      | Pre-trained Weights                                                                                     | Tasks Supported                                    | Inference | Validation | Training | Export |
|-----------------|---------------------------------------------------------------------------------------------------------|----------------------------------------------------|-----------|------------|----------|--------|
| YOLOv8n-human   | [yolov8n-human.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-human.pt)     | [Human attributes estimation](../tasks/human.md)   | ✅        | ✅         | ✅       | ✅     |
| YOLOv8s-human   | [yolov8s-human.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-human.pt)     | [Human attributes estimation](../tasks/human.md)   | ✅        | ✅         | ✅       | ✅     |
| YOLOv8m-human   | [yolov8m-human.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-human.pt)     | [Human attributes estimation](../tasks/human.md)   | ✅        | ✅         | ✅       | ✅     |
| YOLOv8l-human   | [yolov8l-human.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-human.pt)     | [Human attributes estimation](../tasks/human.md)   | ✅        | ✅         | ✅       | ✅     |
| YOLOv8x-human   | [yolov8x-human.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-human.pt)     | [Human attributes estimation](../tasks/human.md)   | ✅        | ✅         | ✅       | ✅     |


## Usage Examples

### Train Usage

Train YOLOv8n-human on the COCO8-Human dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-human.yaml")  # build a new model from YAML

        # Train the model
        results = model.train(data="coco8-human.yaml", epochs=100, imgsz=640)
        ```
    === "CLI"

        ```bash
        # Build a new model from YAML and start training from scratch
        yolo human train data=coco8-human.yaml model=yolov8n-human.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo human train data=coco8-human.yaml model=yolov8n-human.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo human train data=coco8-human.yaml model=yolov8n-human.yaml pretrained=yolov8n-human.pt epochs=100 imgsz=640
        ```


#### Dataset format

Human Detection and Attributes Estimation dataset format can be found in detail in the [Dataset Guide](../datasets/human/index.md).

### Val Usage

Validate trained YOLOv8n-human model accuracy on the COCO8-human dataset. No argument need to passed as the `model` retains it's training `data="coco8-human"` and arguments as model attributes.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-human.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list contains map50-95 of each category
        metrics.attrs_accuracy # human attributes estimation accuracy
        ```
    === "CLI"

        ```bash
        yolo human val model=yolov8n-human.pt  # val official model
        yolo human val model=path/to/best.pt  # val custom model
        ```

### Predict Usage

Use a trained YOLOv8n-human model to run predictions on images.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-human.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```
    === "CLI"

        ```bash
        yolo human predict model=yolov8n-human.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo human predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Export Usage

Export a YOLOv8n-human model to a different format like ONNX, CoreML, etc.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-human.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained model

        # Export the model
        model.export(format="onnx")
        ```
    === "CLI"

        ```bash
        yolo export model=yolov8n-human.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`. You can predict or validate directly on exported models, i.e. `yolo predict model=yolov8n-human.onnx`. Usage examples are shown for your model after export completes.

| Format                                            | `format` Argument | Model                     | Metadata | Arguments                                                            |
|---------------------------------------------------|-------------------|---------------------------|----------|----------------------------------------------------------------------|
| [PyTorch](https://pytorch.org/)                   | -                 | `yolov8n-human.pt`              | ✅ | -                                                                    |
| [TorchScript](../integrations/torchscript.md)     | `torchscript`     | `yolov8n-human.torchscript`     | ✅ | `imgsz`, `optimize`, `batch`                                         |
| [ONNX](../integrations/onnx.md)                   | `onnx`            | `yolov8n-human.onnx`            | ✅ | `imgsz`, `half`, `dynamic`, `simplify`, `opset`, `batch`             |
| [OpenVINO](../integrations/openvino.md)           | `openvino`        | `yolov8n-human_openvino_model/` | ✅ | `imgsz`, `half`, `int8`, `batch`                                     |
| [TensorRT](../integrations/tensorrt.md)           | `engine`          | `yolov8n-human.engine`          | ✅ | `imgsz`, `half`, `dynamic`, `simplify`, `workspace`, `int8`, `batch` |
| [CoreML](../integrations/coreml.md)               | `coreml`          | `yolov8n-human.mlpackage`       | ✅ | `imgsz`, `half`, `int8`, `nms`, `batch`                              |
| [TF SavedModel](../integrations/tf-savedmodel.md) | `saved_model`     | `yolov8n-human_saved_model/`    | ✅ | `imgsz`, `keras`, `int8`, `batch`                                    |
| [TF GraphDef](../integrations/tf-graphdef.md)     | `pb`              | `yolov8n-human.pb`              | ❌ | `imgsz`, `batch`                                                     |
| [TF Lite](../integrations/tflite.md)              | `tflite`          | `yolov8n-human.tflite`          | ✅ | `imgsz`, `half`, `int8`, `batch`                                     |
| [TF Edge TPU](../integrations/edge-tpu.md)        | `edgetpu`         | `yolov8n-human_edgetpu.tflite`  | ✅ | `imgsz`, `batch`                                                     |
| [TF.js](../integrations/tfjs.md)                  | `tfjs`            | `yolov8n-human_web_model/`      | ✅ | `imgsz`, `half`, `int8`, `batch`                                     |
| [PaddlePaddle](../integrations/paddlepaddle.md)   | `paddle`          | `yolov8n-human_paddle_model/`   | ✅ | `imgsz`, `batch`                                                     |
| [NCNN](../integrations/ncnn.md)                   | `ncnn`            | `yolov8n-human_ncnn_model/`     | ✅ | `imgsz`, `half`, `batch`                                             |

See full `export` details in the [Export](../modes/export.md) page.
