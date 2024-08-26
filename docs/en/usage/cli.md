---
comments: true
description: Explore the YOLOv8 command line interface (CLI) for easy execution of detection tasks without needing a Python environment.
keywords: YOLOv8 CLI, command line interface, YOLOv8 commands, detection tasks, Ultralytics, model training, model prediction
---

# Command Line Interface Usage

The YOLO command line interface (CLI) allows for simple single-line commands without the need for a Python environment. CLI requires no customization or Python code. You can simply run all tasks from the terminal with the `yolo` command.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=19"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: CLI
</p>

!!! Example

    === "Syntax"

        Ultralytics `yolo` commands use the following syntax:
        ```bash
        yolo TASK MODE ARGS

        Where   TASK (optional) is one of [detect, segment, classify, pose, obb]
                MODE (required) is one of [train, val, predict, export, track, benchmark]
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
        ```
        See all ARGS in the full [Configuration Guide](cfg.md) or with `yolo cfg`

    === "Train"

        Train a detection model for 10 epochs with an initial learning_rate of 0.01
        ```bash
        yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Val a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640
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

- `TASK` (optional) is one of `[detect, segment, classify, pose, obb]`. If it is not passed explicitly YOLOv8 will try to guess the `TASK` from the model type.
- `MODE` (required) is one of `[train, val, predict, export, track, benchmark]`
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults. For a full list of available `ARGS` see the [Configuration](cfg.md) page and `defaults.yaml`

!!! Warning "Warning"

    Arguments must be passed as `arg=val` pairs, split by an equals `=` sign and delimited by spaces ` ` between pairs. Do not use `--` argument prefixes or commas `,` between arguments.

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## Train

Train YOLOv8n on the COCO8 dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](cfg.md) page.

!!! Example "Example"

    === "Train"

        Start training YOLOv8n on COCO8 for 100 epochs at image-size 640.
        ```bash
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

    === "Resume"

        Resume an interrupted training.
        ```bash
        yolo detect train resume model=last.pt
        ```

## Val

Validate trained YOLOv8n model accuracy on the COCO8 dataset. No argument need to passed as the `model` retains its training `data` and arguments as model attributes.

!!! Example "Example"

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

!!! Example "Example"

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

!!! Example "Example"

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

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## Overriding default arguments

Default arguments can be overridden by simply passing them as arguments in the CLI in `arg=value` pairs.

!!! Tip ""

    === "Train"

        Train a detection model for `10 epochs` with `learning_rate` of `0.01`
        ```bash
        yolo detect train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Validate a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo detect val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640
        ```

## Overriding default config file

You can override the `default.yaml` config file entirely by passing a new file with the `cfg` arguments, i.e. `cfg=custom.yaml`.

To do this first create a copy of `default.yaml` in your current working dir with the `yolo copy-cfg` command.

This will create `default_copy.yaml`, which you can then pass as `cfg=default_copy.yaml` along with any additional args, like `imgsz=320` in this example:

!!! Example

    === "CLI"

        ```bash
        yolo copy-cfg
        yolo cfg=default_copy.yaml imgsz=320
        ```

## FAQ

### How do I use the Ultralytics YOLOv8 command line interface (CLI) for model training?

To train a YOLOv8 model using the CLI, you can execute a simple one-line command in the terminal. For example, to train a detection model for 10 epochs with a learning rate of 0.01, you would run:

```bash
yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
```

This command uses the `train` mode with specific arguments. Refer to the full list of available arguments in the [Configuration Guide](cfg.md).

### What tasks can I perform with the Ultralytics YOLOv8 CLI?

The Ultralytics YOLOv8 CLI supports a variety of tasks including detection, segmentation, classification, validation, prediction, export, and tracking. For instance:

- **Train a Model**: Run `yolo train data=<data.yaml> model=<model.pt> epochs=<num>`.
- **Run Predictions**: Use `yolo predict model=<model.pt> source=<data_source> imgsz=<image_size>`.
- **Export a Model**: Execute `yolo export model=<model.pt> format=<export_format>`.

Each task can be customized with various arguments. For detailed syntax and examples, see the respective sections like [Train](#train), [Predict](#predict), and [Export](#export).

### How can I validate the accuracy of a trained YOLOv8 model using the CLI?

To validate a YOLOv8 model's accuracy, use the `val` mode. For example, to validate a pretrained detection model with a batch size of 1 and image size of 640, run:

```bash
yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640
```

This command evaluates the model on the specified dataset and provides performance metrics. For more details, refer to the [Val](#val) section.

### What formats can I export my YOLOv8 models to using the CLI?

YOLOv8 models can be exported to various formats such as ONNX, CoreML, TensorRT, and more. For instance, to export a model to ONNX format, run:

```bash
yolo export model=yolov8n.pt format=onnx
```

For complete details, visit the [Export](../modes/export.md) page.

### How do I customize YOLOv8 CLI commands to override default arguments?

To override default arguments in YOLOv8 CLI commands, pass them as `arg=value` pairs. For example, to train a model with custom arguments, use:

```bash
yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
```

For a full list of available arguments and their descriptions, refer to the [Configuration Guide](cfg.md). Ensure arguments are formatted correctly, as shown in the [Overriding default arguments](#overriding-default-arguments) section.
