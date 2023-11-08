---
comments: true
description: 'Learn how to use Ultralytics YOLO through Command Line: train models, run predictions and exports models to different formats easily using terminal commands.'
keywords: Ultralytics, YOLO, CLI, train, validation, prediction, command line interface, YOLO CLI, YOLO terminal, model training, prediction, exporting
---

# Command Line Interface Usage

The YOLO command line interface (CLI) allows for simple single-line commands without the need for a Python environment. CLI requires no customization or Python code. You can simply run all tasks from the terminal with the `yolo` command.

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: CLI & Python Usage and Live Inference
</p>

!!! example

    === "Syntax"

        Ultralytics `yolo` commands use the following syntax:
        ```bash
        yolo TASK MODE ARGS

        Where   TASK (optional) is one of [detect, segment, classify]
                MODE (required) is one of [train, val, predict, export, track]
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
        ```
        See all ARGS in the full [Configuration Guide](./cfg.md) or with `yolo cfg`

    === "Train"

        Train a detection model for 10 epochs with an initial learning_rate of 0.01
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Val a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "Export"

        Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "Special"

        Run special commands to see version, view settings, run checks and more:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

Where:

- `TASK` (optional) is one of `[detect, segment, classify]`. If it is not passed explicitly YOLOv8 will try to guess the `TASK` from the model type.
- `MODE` (required) is one of `[train, val, predict, export, track]`
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults. For a full list of available `ARGS` see the [Configuration](cfg.md) page and `defaults.yaml`
  GitHub [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/default.yaml).

!!! warning "Warning"

    Arguments must be passed as `arg=val` pairs, split by an equals `=` sign and delimited by spaces ` ` between pairs. Do not use `--` argument prefixes or commas `,` between arguments.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## Train

Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](cfg.md) page.

!!! example "Example"

    === "Train"

        Start training YOLOv8n on COCO128 for 100 epochs at image-size 640.
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

    === "Resume"

        Resume an interrupted training.
        ```bash
        yolo detect train resume model=last.pt
        ```

## Val

Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's training `data` and arguments as model attributes.

!!! example "Example"

    === "Official"

        Validate an official YOLOv8n model.
        ```bash
        yolo detect val model=yolov8n.pt
        ```

    === "Custom"

        Validate a custom-trained model.
        ```bash
        yolo detect val model=path/to/best.pt
        ```

## Predict

Use a trained YOLOv8n model to run predictions on images.

!!! example "Example"

    === "Official"

        Predict with an official YOLOv8n model.
        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

    === "Custom"

        Predict with a custom model.
        ```bash
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'
        ```

## Export

Export a YOLOv8n model to a different format like ONNX, CoreML, etc.

!!! example "Example"

    === "Official"

        Export an official YOLOv8n model to ONNX format.
        ```bash
        yolo export model=yolov8n.pt format=onnx
        ```

    === "Custom"

        Export a custom-trained model to ONNX format.
        ```bash
        yolo export model=path/to/best.pt format=onnx
        ```

Available YOLOv8 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`.

| Format                                                             | `format` Argument | Model                     | Metadata | Arguments                                           |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|-----------------------------------------------------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        | -                                                   |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        | `imgsz`, `optimize`                                 |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `opset`     |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅        | `imgsz`, `half`                                     |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        | `imgsz`, `half`, `dynamic`, `simplify`, `workspace` |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlpackage`       | ✅        | `imgsz`, `half`, `int8`, `nms`                      |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        | `imgsz`, `keras`                                    |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        | `imgsz`                                             |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        | `imgsz`, `half`, `int8`                             |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        | `imgsz`                                             |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        | `imgsz`                                             |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        | `imgsz`                                             |
| [ncnn](https://github.com/Tencent/ncnn)                            | `ncnn`            | `yolov8n_ncnn_model/`     | ✅        | `imgsz`, `half`                                     |

## Overriding default arguments

Default arguments can be overridden by simply passing them as arguments in the CLI in `arg=value` pairs.

!!! tip ""

    === "Train"
        Train a detection model for `10 epochs` with `learning_rate` of `0.01`
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"
        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"
        Validate a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

## Overriding default config file

You can override the `default.yaml` config file entirely by passing a new file with the `cfg` arguments, i.e. `cfg=custom.yaml`.

To do this first create a copy of `default.yaml` in your current working dir with the `yolo copy-cfg` command.

This will create `default_copy.yaml`, which you can then pass as `cfg=default_copy.yaml` along with any additional args, like `imgsz=320` in this example:

!!! example ""

    === "CLI"
        ```bash
        yolo copy-cfg
        yolo cfg=default_copy.yaml imgsz=320
        ```
