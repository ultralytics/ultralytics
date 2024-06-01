---
comments: true
description: Official documentation for YOLOv8 by Ultralytics. Learn how to train, validate, predict, and export models in various formats. Including detailed performance stats.
keywords: YOLOv8, Ultralytics, human detection, attribute estimation, age estimation, gender estimation, weight estimation, height estimation, ethnicity estimation
---

# Human Detection and Attribute Estimation

<img width="1024" src="https://github.com/ultralytics/ultralytics/assets/3855193/c49d8b56-aed6-4303-82b2-790aa24b5515" alt="Human attributes estimation example">

Human detection and attributes estimation is a task that involves identifying humans in an image or video stream and estimating their attributes, such as age, biological gender, weight, height, and ethnicity.
The output of the detector is a set of bounding boxes that enclose the humans in the image, along with class labels, confidence scores, and estimated attributes for each person. This task is useful for applications in surveillance, retail analytics, and human-computer interaction.

???+ warning "Disclaimer on Privacy and Security"

    When using YOLO-Human models for human detection and attribute estimation, please be mindful of privacy and security. Ensure compliance with relevant data protection laws and implement strong security measures. Use these models ethically, avoiding applications that could cause harm or discrimination. Ultralytics is not responsible for any misuse. Users are responsible for ensuring their use complies with all legal and ethical standards.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models/v8)

YOLOv8 pretrained human models are shown here. Detect, Segment and Pose models are pretrained on the [COCO](../datasets/detect/coco.md) dataset, while Classify models are pretrained on the [ImageNet](../datasets/classify/imagenet.md) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/models) download automatically from the latest Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

!!! note

    It is important to note that these models have been trained on a specially curated, artificially annotated version of the COCO dataset. This custom dataset was meticulously crafted to enhance the models' performance on specific tasks by incorporating additional annotations and adjustments beyond those available in the public COCO dataset. This enhanced version of the dataset is not publicly available (a sample of the dataset is available at [this page](../datasets/human/coco8-human.md). The artificial annotations were designed to provide more comprehensive and nuanced data, enabling the models to achieve higher accuracy and robustness in their predictions.


| Model                                                                                            | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|--------------------------------------------------------------------------------------------------|-----------------------|----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
| [YOLOv8n-human](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-human.pt) | 640                   | Coming soon          | Coming soon                    | Coming soon                         | Coming soon        | Coming soon       |
| [YOLOv8s-human](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s-human.pt) | 640                   | Coming soon          | Coming soon                    | Coming soon                         | Coming soon        | Coming soon       |
| [YOLOv8m-human](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m-human.pt) | 640                   | Coming soon          | Coming soon                    | Coming soon                         | Coming soon        | Coming soon       |
| [YOLOv8l-human](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l-human.pt) | 640                   | Coming soon          | Coming soon                    | Coming soon                         | Coming soon        | Coming soon       |
| [YOLOv8x-human](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x-human.pt) | 640                   | Coming soon          | Coming soon                    | Coming soon                         | Coming soon        | Coming soon       |

## Train

Train YOLOv8n-human on the COCO8-Human dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](../usage/cfg.md) page.

!!! Example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n-human.yaml")  # build a new model from YAML
        model = YOLO("yolov8n-human.pt")  # load a pretrained model (recommended for training)
        model = YOLO("yolov8n-human.yaml").load("yolov8n-human.pt")  # build from YAML and transfer weights

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


### Dataset format

Human Detection and Attributes Estimation dataset format can be found in detail in the [Dataset Guide](../datasets/human/index.md).

## Val

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

## Predict

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

## Export

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
