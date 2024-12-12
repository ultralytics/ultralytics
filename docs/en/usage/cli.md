---
comments: true
description: Explore the YOLO11 command line interface (CLI) for easy execution of detection tasks without needing a Python environment.
keywords: YOLO11 CLI, command line interface, YOLO11 commands, detection tasks, Ultralytics, model training, model prediction
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
  <strong>Watch:</strong> Mastering Ultralytics YOLO: CLI
</p>

!!! example

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

        Train a detection model for 10 [epochs](https://www.ultralytics.com/glossary/epoch) with an initial learning_rate of 0.01
        ```bash
        yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Val a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
        ```

    === "Export"

        Export a YOLO11n classification model to ONNX format at image size 224 by 128 (no TASK required)
        ```bash
        yolo export model=yolo11n-cls.pt format=onnx imgsz=224,128
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

- `TASK` (optional) is one of `[detect, segment, classify, pose, obb]`. If it is not passed explicitly YOLO11 will try to guess the `TASK` from the model type.
- `MODE` (required) is one of `[train, val, predict, export, track, benchmark]`
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults. For a full list of available `ARGS` see the [Configuration](cfg.md) page and `defaults.yaml`

!!! warning

    Arguments must be passed as `arg=val` pairs, split by an equals `=` sign and delimited by spaces ` ` between pairs. Do not use `--` argument prefixes or commas `,` between arguments.

    - `yolo predict model=yolo11n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolo11n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolo11n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## Train

Train YOLO11n on the COCO8 dataset for 100 epochs at image size 640. For a full list of available arguments see the [Configuration](cfg.md) page.

!!! example

    === "Train"

        Start training YOLO11n on COCO8 for 100 epochs at image-size 640.
        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=100 imgsz=640
        ```

    === "Resume"

        Resume an interrupted training.
        ```bash
        yolo detect train resume model=last.pt
        ```

## Val

Validate trained YOLO11n model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes.

!!! example

    === "Official"

        Validate an official YOLO11n model.
        ```bash
        yolo detect val model=yolo11n.pt
        ```

    === "Custom"

        Validate a custom-trained model.
        ```bash
        yolo detect val model=path/to/best.pt
        ```

## Predict

Use a trained YOLO11n model to run predictions on images.

!!! example

    === "Official"

        Predict with an official YOLO11n model.
        ```bash
        yolo detect predict model=yolo11n.pt source='https://ultralytics.com/images/bus.jpg'
        ```

    === "Custom"

        Predict with a custom model.
        ```bash
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'
        ```

## Export

Export a YOLO11n model to a different format like ONNX, CoreML, etc.

!!! example

    === "Official"

        Export an official YOLO11n model to ONNX format.
        ```bash
        yolo export model=yolo11n.pt format=onnx
        ```

    === "Custom"

        Export a custom-trained model to ONNX format.
        ```bash
        yolo export model=path/to/best.pt format=onnx
        ```

Available YOLO11 export formats are in the table below. You can export to any format using the `format` argument, i.e. `format='onnx'` or `format='engine'`.

{% include "macros/export-table.md" %}

See full `export` details in the [Export](../modes/export.md) page.

## Overriding default arguments

Default arguments can be overridden by simply passing them as arguments in the CLI in `arg=value` pairs.

!!! tip

    === "Train"

        Train a detection model for `10 epochs` with `learning_rate` of `0.01`
        ```bash
        yolo detect train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
        ```

    === "Predict"

        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo segment predict model=yolo11n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "Val"

        Validate a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo detect val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
        ```

## Overriding default config file

You can override the `default.yaml` config file entirely by passing a new file with the `cfg` arguments, i.e. `cfg=custom.yaml`.

To do this first create a copy of `default.yaml` in your current working dir with the `yolo copy-cfg` command.

This will create `default_copy.yaml`, which you can then pass as `cfg=default_copy.yaml` along with any additional args, like `imgsz=320` in this example:

!!! example

    === "CLI"

        ```bash
        yolo copy-cfg
        yolo cfg=default_copy.yaml imgsz=320
        ```

## FAQ

### How do I use the Ultralytics YOLO11 command line interface (CLI) for model training?

To train a YOLO11 model using the CLI, you can execute a simple one-line command in the terminal. For example, to train a detection model for 10 epochs with a [learning rate](https://www.ultralytics.com/glossary/learning-rate) of 0.01, you would run:

```bash
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
```

This command uses the `train` mode with specific arguments. Refer to the full list of available arguments in the [Configuration Guide](cfg.md).

### What tasks can I perform with the Ultralytics YOLO11 CLI?

The Ultralytics YOLO11 CLI supports a variety of tasks including detection, segmentation, classification, validation, prediction, export, and tracking. For instance:

- **Train a Model**: Run `yolo train data=<data.yaml> model=<model.pt> epochs=<num>`.
- **Run Predictions**: Use `yolo predict model=<model.pt> source=<data_source> imgsz=<image_size>`.
- **Export a Model**: Execute `yolo export model=<model.pt> format=<export_format>`.

Each task can be customized with various arguments. For detailed syntax and examples, see the respective sections like [Train](#train), [Predict](#predict), and [Export](#export).

### How can I validate the accuracy of a trained YOLO11 model using the CLI?

To validate a YOLO11 model's accuracy, use the `val` mode. For example, to validate a pretrained detection model with a [batch size](https://www.ultralytics.com/glossary/batch-size) of 1 and image size of 640, run:

```bash
yolo val model=yolo11n.pt data=coco8.yaml batch=1 imgsz=640
```

This command evaluates the model on the specified dataset and provides performance metrics. For more details, refer to the [Val](#val) section.

### What formats can I export my YOLO11 models to using the CLI?

YOLO11 models can be exported to various formats such as ONNX, CoreML, TensorRT, and more. For instance, to export a model to ONNX format, run:

```bash
yolo export model=yolo11n.pt format=onnx
```

For complete details, visit the [Export](../modes/export.md) page.

### How do I customize YOLO11 CLI commands to override default arguments?

To override default arguments in YOLO11 CLI commands, pass them as `arg=value` pairs. For example, to train a model with custom arguments, use:

```bash
yolo train data=coco8.yaml model=yolo11n.pt epochs=10 lr0=0.01
```

For a full list of available arguments and their descriptions, refer to the [Configuration Guide](cfg.md). Ensure arguments are formatted correctly, as shown in the [Overriding default arguments](#overriding-default-arguments) section.
