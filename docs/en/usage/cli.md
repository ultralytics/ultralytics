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
        yolo cfg # (3)
        yolo checks
        yolo copy-cfg
        yolo help
        yolo info model=yolo11n.pt # (1)
        yolo settings # (2)
        yolo version
        ```

        1. Learn more about [`yolo info`](#info) command.
        2. Learn more about [`yolo settings`](../quickstart.md#ultralytics-settings)
        3. Learn more about [`yolo cfg`](#overriding-default-config-file)

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

## Info

View model weights file information directly in terminal. This can be especially helpful when looking for models which ran [train](../modes/train.md) for specific `ultraltyics` versions or that contain specific class labels.

### Arguments

The following table outlines the accepted arguments for the `yolo info` CLI command. Any other arguments provided will be ignored.

| Name    |  Type  | Default | Description                                                                                                |
| ------- | :----: | :-----: | ---------------------------------------------------------------------------------------------------------- |
| model   | `str`  | `None`  | Path or filename to model weights to return info about. Helpful to ensure model-type is in filename.       |
| verbose | `bool` | `False` | Optional argument to show model layer information (only available for PyTorch models), default is `False`. |

See [CLI syntax section in the quickstart](../quickstart.md/#__tabbed_2_1) or in the [Examples](#example-use-of-yolo-info) section for how to correctly pass arguments to CLI commands.

### Example Use of `yolo info`

!!! example

    === "Basic"

        Displays class names, training arguments, and model metadata. Output will vary by model-type and weights framework.

        ```bash
        # Fetch info for PyTorch model (1)
        yolo info model=yolo11n.pt

        ```

        1. It's possible to use _any_ supported [export format](../modes/export.md#export-formats) model weights.

        ??? success "Preview PyTorch Weights Output"

            Output shown for `.pt` model, however depending on model weights framework or model-type, the output could look different.

            ```
            Ultralytics Model Info:
            --------------------------------------------------------------------------------
            Model File: yolo11n.pt
            Date: 2024-09-25T21:10:26.629566
            Version: 8.2.100
            License: AGPL-3.0 License (https://ultralytics.com/license)
            Docs: https://docs.ultralytics.com
            Epoch: -1
            Best_fitness: None
            Updates: None
            Train_args:
                    task: detect
                    mode: train
                    model: yolo11n.yaml
                    data: coco.yaml
                    epochs: 600
                    time: None
                    patience: 100
                    batch: 128
                    imgsz: 640
                    save: True
                    save_period: -1
                    cache: disk
                    device: 0
                    workers: 8
                    project:
                    name: yolov11n
                    exist_ok: False
                    pretrained: True
                    optimizer: auto
                    verbose: True
                    seed: 0
                    deterministic: True
                    single_cls: False
                    rect: False
                    cos_lr: False
                    close_mosaic: 10
                    resume: False
                    amp: True
                    fraction: 1.0
                    profile: False
                    freeze: None
                    multi_scale: False
                    overlap_mask: True
                    mask_ratio: 4
                    dropout: 0.0
                    val: True
                    split: val
                    save_json: False
                    save_hybrid: False
                    conf: None
                    iou: 0.7
                    max_det: 300
                    half: False
                    dnn: False
                    plots: True
                    source: None
                    vid_stride: 1
                    stream_buffer: False
                    visualize: False
                    augment: False
                    agnostic_nms: False
                    classes: None
                    retina_masks: False
                    embed: None
                    show: False
                    save_frames: False
                    save_txt: False
                    save_conf: False
                    save_crop: False
                    show_labels: True
                    show_conf: True
                    show_boxes: True
                    line_width: None
                    format: torchscript
                    keras: False
                    optimize: False
                    int8: False
                    dynamic: False
                    simplify: False
                    opset: None
                    workspace: 4
                    nms: False
                    lr0: 0.01
                    lrf: 0.01
                    momentum: 0.937
                    weight_decay: 0.0005
                    warmup_epochs: 3.0
                    warmup_momentum: 0.8
                    warmup_bias_lr: 0.0
                    box: 7.5
                    cls: 0.5
                    dfl: 1.5
                    pose: 12.0
                    kobj: 1.0
                    label_smoothing: 0.0
                    nbs: 64
                    hsv_h: 0.015
                    hsv_s: 0.7
                    hsv_v: 0.4
                    degrees: 0.0
                    translate: 0.1
                    scale: 0.5
                    shear: 0.0
                    perspective: 0.0
                    flipud: 0.0
                    fliplr: 0.5
                    bgr: 0.0
                    mosaic: 0.0
                    mixup: 0.0
                    copy_paste: 0.0
                    copy_paste_mode: flip
                    auto_augment: randaugment
                    erasing: 0.4
                    crop_fraction: 1.0
                    cfg: None
                    tracker: botsort.yaml
            Train_metrics:
                    metrics/precision(B): 0.6563741360313602
                    metrics/recall(B): 0.5024662662239141
                    metrics/mAP50(B): 0.5514423440292427
                    metrics/mAP50-95(B): 0.3939947730997559
                    val/box_loss: 1.09948
                    val/cls_loss: 1.13236
                    val/dfl_loss: 1.12834
                    fitness: 0.40731
                    train/cls_loss: 1.07058
                    train/box_loss: 1.07074
                    train/dfl_loss: 1.13056
                    lr/pg0: 0.0001165
                    lr/pg1: 0.0001165
                    lr/pg2: 0.0001165
            Names: total=(80)
                    0: person
                    1: bicycle
                    2: car
                    3: motorcycle
                    4: airplane
                    5: bus
                    6: train
                    7: truck
                    8: boat
                    9: traffic light
                    10: fire hydrant
                    11: stop sign
                    12: parking meter
                    13: bench
                    14: bird
                    15: cat
                    16: dog
                    17: horse
                    18: sheep
                    19: cow
                    20: elephant
                    21: bear
                    22: zebra
                    23: giraffe
                    24: backpack
                    25: umbrella
                    26: handbag
                    27: tie
                    28: suitcase
                    29: frisbee
                    30: skis
                    31: snowboard
                    32: sports ball
                    33: kite
                    34: baseball bat
                    35: baseball glove
                    36: skateboard
                    37: surfboard
                    38: tennis racket
                    39: bottle
                    40: wine glass
                    41: cup
                    42: fork
                    43: knife
                    44: spoon
                    45: bowl
                    46: banana
                    47: apple
                    48: sandwich
                    49: orange
                    50: broccoli
                    51: carrot
                    52: hot dog
                    53: pizza
                    54: donut
                    55: cake
                    56: chair
                    57: couch
                    58: potted plant
                    59: bed
                    60: dining table
                    61: toilet
                    62: tv
                    63: laptop
                    64: mouse
                    65: remote
                    66: keyboard
                    67: cell phone
                    68: microwave
                    69: oven
                    70: toaster
                    71: sink
                    72: refrigerator
                    73: book
                    74: clock
                    75: vase
                    76: scissors
                    77: teddy bear
                    78: hair drier
                    79: toothbrush
            --------------------------------------------------------------------------------
            YOLO11n summary: 319 layers, 2,624,080 parameters, 0 gradients, 6.6 GFLOPs
            ```

        ??? success "Preview ONNX Weights Output"

            ```
            Ultralytics Model Info:
            --------------------------------------------------------------------------------
            Model File: yolo11n.onnx
            Date: 2024-10-09T13:24:46.986346
            Description: Ultralytics YOLO11n model trained on coco.yaml
            Author: Ultralytics
            Version: 8.3.9
            Task: detect
            License: AGPL-3.0 License (https://ultralytics.com/license)
            Docs: https://docs.ultralytics.com
            Stride: 32
            Batch: 1
            Imgsz: [640, 640]
            Names:
                    0: person
                    1: bicycle
                    2: car
                    3: motorcycle
                    4: airplane
                    5: bus
                    6: train
                    7: truck
                    8: boat
                    9: traffic light
                    10: fire hydrant
                    11: stop sign
                    12: parking meter
                    13: bench
                    14: bird
                    15: cat
                    16: dog
                    17: horse
                    18: sheep
                    19: cow
                    20: elephant
                    21: bear
                    22: zebra
                    23: giraffe
                    24: backpack
                    25: umbrella
                    26: handbag
                    27: tie
                    28: suitcase
                    29: frisbee
                    30: skis
                    31: snowboard
                    32: sports ball
                    33: kite
                    34: baseball bat
                    35: baseball glove
                    36: skateboard
                    37: surfboard
                    38: tennis racket
                    39: bottle
                    40: wine glass
                    41: cup
                    42: fork
                    43: knife
                    44: spoon
                    45: bowl
                    46: banana
                    47: apple
                    48: sandwich
                    49: orange
                    50: broccoli
                    51: carrot
                    52: hot dog
                    53: pizza
                    54: donut
                    55: cake
                    56: chair
                    57: couch
                    58: potted plant
                    59: bed
                    60: dining table
                    61: toilet
                    62: tv
                    63: laptop
                    64: mouse
                    65: remote
                    66: keyboard
                    67: cell phone
                    68: microwave
                    69: oven
                    70: toaster
                    71: sink
                    72: refrigerator
                    73: book
                    74: clock
                    75: vase
                    76: scissors
                    77: teddy bear
                    78: hair drier
                    79: toothbrush
            --------------------------------------------------------------------------------
            ```

    === "Show Model Layers"

        Same information from [Basic](cli.md#__tabbed_6_1) with the addition of model layer information.

        !!! warning ""

            Only valid for native PyTorch `.pt` models. If using exported model weights, nothing additional will be shown.

        ```bash
        yolo info model=yolov11n.pt verbose
        ```

        ??? success "Preview of Additional Output"

            ```
            layer                                   name  gradient   parameters                shape         mu      sigma
              0                      model.0.conv.weight     False          432        [16, 3, 3, 3]   -0.00165       0.14 torch.float32
              1                        model.0.bn.weight     False           16                 [16]       3.07        1.9 torch.float32
              2                          model.0.bn.bias     False           16                 [16]      0.805       4.26 torch.float32
              3                      model.1.conv.weight     False         4608       [32, 16, 3, 3]   0.000659     0.0564 torch.float32
              4                        model.1.bn.weight     False           32                 [32]       3.77      0.892 torch.float32
              5                          model.1.bn.bias     False           32                 [32]       1.38       1.52 torch.float32
              6                  model.2.cv1.conv.weight     False         1024       [32, 32, 1, 1]   -0.00347     0.0821 torch.float32
              7                    model.2.cv1.bn.weight     False           32                 [32]       2.95       1.02 torch.float32
              8                      model.2.cv1.bn.bias     False           32                 [32]      0.888       1.48 torch.float32
              9                  model.2.cv2.conv.weight     False         3072       [64, 48, 1, 1]   -0.00604      0.061 torch.float32
             10                    model.2.cv2.bn.weight     False           64                 [64]       1.93        0.8 torch.float32
             11                      model.2.cv2.bn.bias     False           64                 [64]    -0.0274       1.18 torch.float32
             12              model.2.m.0.cv1.conv.weight     False         1152        [8, 16, 3, 3]    -0.0041     0.0431 torch.float32
             13                model.2.m.0.cv1.bn.weight     False            8                  [8]      0.647      0.516 torch.float32
             14                  model.2.m.0.cv1.bn.bias     False            8                  [8]      0.318      0.993 torch.float32
             15              model.2.m.0.cv2.conv.weight     False         1152        [16, 8, 3, 3]    0.00143     0.0598 torch.float32
             16                model.2.m.0.cv2.bn.weight     False           16                 [16]       3.15      0.455 torch.float32
             17                  model.2.m.0.cv2.bn.bias     False           16                 [16]      0.971       2.88 torch.float32
             18                      model.3.conv.weight     False        36864       [64, 64, 3, 3]   -0.00137     0.0247 torch.float32
             19                        model.3.bn.weight     False           64                 [64]      0.967      0.257 torch.float32
             20                          model.3.bn.bias     False           64                 [64]      0.377       1.02 torch.float32
             21                  model.4.cv1.conv.weight     False         4096       [64, 64, 1, 1]   -0.00366     0.0509 torch.float32
             22                    model.4.cv1.bn.weight     False           64                 [64]      0.696      0.281 torch.float32
             23                      model.4.cv1.bn.bias     False           64                 [64]      0.435      0.694 torch.float32
             24                  model.4.cv2.conv.weight     False        12288      [128, 96, 1, 1]   -0.00353     0.0384 torch.float32
             25                    model.4.cv2.bn.weight     False          128                [128]      0.759      0.206 torch.float32
             26                      model.4.cv2.bn.bias     False          128                [128]     -0.834      0.664 torch.float32
             27              model.4.m.0.cv1.conv.weight     False         4608       [16, 32, 3, 3]    -0.0017     0.0368 torch.float32
             28                model.4.m.0.cv1.bn.weight     False           16                 [16]      0.531      0.165 torch.float32
             29                  model.4.m.0.cv1.bn.bias     False           16                 [16]      0.494      0.698 torch.float32
             30              model.4.m.0.cv2.conv.weight     False         4608       [32, 16, 3, 3]   0.000478     0.0378 torch.float32
             31                model.4.m.0.cv2.bn.weight     False           32                 [32]      0.781      0.173 torch.float32
             32                  model.4.m.0.cv2.bn.bias     False           32                 [32]       1.13          1 torch.float32
             33                      model.5.conv.weight     False       147456     [128, 128, 3, 3]  -0.000495     0.0149 torch.float32
             34                        model.5.bn.weight     False          128                [128]      0.712      0.215 torch.float32
             35                          model.5.bn.bias     False          128                [128]      -0.16      0.791 torch.float32
             36                  model.6.cv1.conv.weight     False        16384     [128, 128, 1, 1]   -0.00264     0.0306 torch.float32
             37                    model.6.cv1.bn.weight     False          128                [128]      0.855      0.289 torch.float32
             38                      model.6.cv1.bn.bias     False          128                [128]     -0.147      0.703 torch.float32
             39                  model.6.cv2.conv.weight     False        24576     [128, 192, 1, 1]   -0.00231      0.028 torch.float32
             40                    model.6.cv2.bn.weight     False          128                [128]      0.851      0.211 torch.float32
             41                      model.6.cv2.bn.bias     False          128                [128]     -0.507      0.751 torch.float32
             42              model.6.m.0.cv1.conv.weight     False         2048       [32, 64, 1, 1]   -0.00407     0.0351 torch.float32
             43                model.6.m.0.cv1.bn.weight     False           32                 [32]      0.528      0.156 torch.float32
             44                  model.6.m.0.cv1.bn.bias     False           32                 [32]      0.344      0.558 torch.float32
             45              model.6.m.0.cv2.conv.weight     False         2048       [32, 64, 1, 1]   -0.00208     0.0222 torch.float32
             46                model.6.m.0.cv2.bn.weight     False           32                 [32]        1.3      0.284 torch.float32
             47                  model.6.m.0.cv2.bn.bias     False           32                 [32]     -0.334       0.34 torch.float32
             48              model.6.m.0.cv3.conv.weight     False         4096       [64, 64, 1, 1]   -0.00252     0.0321 torch.float32
             49                model.6.m.0.cv3.bn.weight     False           64                 [64]       1.04      0.227 torch.float32
             50                  model.6.m.0.cv3.bn.bias     False           64                 [64]     -0.382      0.613 torch.float32
             51          model.6.m.0.m.0.cv1.conv.weight     False         9216       [32, 32, 3, 3]   -0.00165      0.023 torch.float32
             52            model.6.m.0.m.0.cv1.bn.weight     False           32                 [32]      0.907      0.135 torch.float32
             53              model.6.m.0.m.0.cv1.bn.bias     False           32                 [32]     -0.611      0.498 torch.float32
             54          model.6.m.0.m.0.cv2.conv.weight     False         9216       [32, 32, 3, 3]   -0.00105      0.022 torch.float32
             55            model.6.m.0.m.0.cv2.bn.weight     False           32                 [32]      0.812      0.187 torch.float32
             56              model.6.m.0.m.0.cv2.bn.bias     False           32                 [32]       0.24      0.676 torch.float32
             57          model.6.m.0.m.1.cv1.conv.weight     False         9216       [32, 32, 3, 3]   -0.00197     0.0227 torch.float32
             58            model.6.m.0.m.1.cv1.bn.weight     False           32                 [32]      0.798       0.14 torch.float32
             59              model.6.m.0.m.1.cv1.bn.bias     False           32                 [32]     -0.578      0.565 torch.float32
             60          model.6.m.0.m.1.cv2.conv.weight     False         9216       [32, 32, 3, 3]  -0.000268      0.022 torch.float32
             61            model.6.m.0.m.1.cv2.bn.weight     False           32                 [32]       1.06      0.215 torch.float32
             62              model.6.m.0.m.1.cv2.bn.bias     False           32                 [32]       1.17      0.767 torch.float32
             63                      model.7.conv.weight     False       294912     [256, 128, 3, 3]  -0.000359     0.0108 torch.float32
             64                        model.7.bn.weight     False          256                [256]      0.879      0.222 torch.float32
             65                          model.7.bn.bias     False          256                [256]     -0.541      0.597 torch.float32
             66                  model.8.cv1.conv.weight     False        65536     [256, 256, 1, 1]   -0.00186     0.0186 torch.float32
             67                    model.8.cv1.bn.weight     False          256                [256]       1.11      0.177 torch.float32
             68                      model.8.cv1.bn.bias     False          256                [256]     -0.682      0.573 torch.float32
             69                  model.8.cv2.conv.weight     False        98304     [256, 384, 1, 1]   -0.00161     0.0168 torch.float32
             70                    model.8.cv2.bn.weight     False          256                [256]        1.2       0.26 torch.float32
             71                      model.8.cv2.bn.bias     False          256                [256]     -0.479      0.633 torch.float32
             72              model.8.m.0.cv1.conv.weight     False         8192      [64, 128, 1, 1]   -0.00251     0.0231 torch.float32
             73                model.8.m.0.cv1.bn.weight     False           64                 [64]      0.666      0.233 torch.float32
             74                  model.8.m.0.cv1.bn.bias     False           64                 [64]     -0.106      0.576 torch.float32
             75              model.8.m.0.cv2.conv.weight     False         8192      [64, 128, 1, 1]    -0.0014     0.0145 torch.float32
             76                model.8.m.0.cv2.bn.weight     False           64                 [64]       1.41      0.125 torch.float32
             77                  model.8.m.0.cv2.bn.bias     False           64                 [64]     -0.264      0.154 torch.float32
             78              model.8.m.0.cv3.conv.weight     False        16384     [128, 128, 1, 1]    -0.0016     0.0206 torch.float32
             79                model.8.m.0.cv3.bn.weight     False          128                [128]       1.28      0.206 torch.float32
             80                  model.8.m.0.cv3.bn.bias     False          128                [128]     -0.696      0.474 torch.float32
             81          model.8.m.0.m.0.cv1.conv.weight     False        36864       [64, 64, 3, 3]   -0.00131      0.014 torch.float32
             82            model.8.m.0.m.0.cv1.bn.weight     False           64                 [64]       1.35      0.184 torch.float32
             83              model.8.m.0.m.0.cv1.bn.bias     False           64                 [64]     -0.922      0.582 torch.float32
             84          model.8.m.0.m.0.cv2.conv.weight     False        36864       [64, 64, 3, 3]  -0.000846     0.0135 torch.float32
             85            model.8.m.0.m.0.cv2.bn.weight     False           64                 [64]       1.15       0.34 torch.float32
             86              model.8.m.0.m.0.cv2.bn.bias     False           64                 [64]    -0.0465      0.716 torch.float32
             87          model.8.m.0.m.1.cv1.conv.weight     False        36864       [64, 64, 3, 3]   -0.00116     0.0134 torch.float32
             88            model.8.m.0.m.1.cv1.bn.weight     False           64                 [64]       1.29      0.221 torch.float32
             89              model.8.m.0.m.1.cv1.bn.bias     False           64                 [64]     -0.553      0.675 torch.float32
             90          model.8.m.0.m.1.cv2.conv.weight     False        36864       [64, 64, 3, 3]  -0.000543     0.0132 torch.float32
             91            model.8.m.0.m.1.cv2.bn.weight     False           64                 [64]       1.64      0.462 torch.float32
             92              model.8.m.0.m.1.cv2.bn.bias     False           64                 [64]      0.456      0.757 torch.float32
             93                  model.9.cv1.conv.weight     False        32768     [128, 256, 1, 1]   -0.00327     0.0232 torch.float32
             94                    model.9.cv1.bn.weight     False          128                [128]      0.804      0.102 torch.float32
             95                      model.9.cv1.bn.bias     False          128                [128]       1.99       0.43 torch.float32
             96                  model.9.cv2.conv.weight     False       131072     [256, 512, 1, 1]   0.000171     0.0151 torch.float32
             97                    model.9.cv2.bn.weight     False          256                [256]      0.899      0.272 torch.float32
             98                      model.9.cv2.bn.bias     False          256                [256]      -1.18       1.03 torch.float32
             99                 model.10.cv1.conv.weight     False        65536     [256, 256, 1, 1]   -0.00251     0.0197 torch.float32
            100                   model.10.cv1.bn.weight     False          256                [256]       1.47       0.33 torch.float32
            101                     model.10.cv1.bn.bias     False          256                [256]     -0.205      0.649 torch.float32
            102                 model.10.cv2.conv.weight     False        65536     [256, 256, 1, 1]   -0.00118     0.0184 torch.float32
            103                   model.10.cv2.bn.weight     False          256                [256]      0.994      0.232 torch.float32
            104                     model.10.cv2.bn.bias     False          256                [256]      -1.02      0.699 torch.float32
            105        model.10.m.0.attn.qkv.conv.weight     False        32768     [256, 128, 1, 1]   0.000541     0.0224 torch.float32
            106          model.10.m.0.attn.qkv.bn.weight     False          256                [256]       1.18      0.409 torch.float32
            107            model.10.m.0.attn.qkv.bn.bias     False          256                [256]    -0.0338      0.345 torch.float32
            108       model.10.m.0.attn.proj.conv.weight     False        16384     [128, 128, 1, 1]   0.000203     0.0221 torch.float32
            109         model.10.m.0.attn.proj.bn.weight     False          128                [128]      0.751      0.153 torch.float32
            110           model.10.m.0.attn.proj.bn.bias     False          128                [128]  -7.34e-06    8.7e-05 torch.float32
            111         model.10.m.0.attn.pe.conv.weight     False         1152       [128, 1, 3, 3]    -0.0134     0.0415 torch.float32
            112           model.10.m.0.attn.pe.bn.weight     False          128                [128]      0.824      0.327 torch.float32
            113             model.10.m.0.attn.pe.bn.bias     False          128                [128]  -1.11e-05   9.13e-05 torch.float32
            114           model.10.m.0.ffn.0.conv.weight     False        32768     [256, 128, 1, 1]   -0.00129     0.0177 torch.float32
            115             model.10.m.0.ffn.0.bn.weight     False          256                [256]      0.974      0.115 torch.float32
            116               model.10.m.0.ffn.0.bn.bias     False          256                [256]       -1.5      0.204 torch.float32
            117           model.10.m.0.ffn.1.conv.weight     False        32768     [128, 256, 1, 1]   0.000214     0.0154 torch.float32
            118             model.10.m.0.ffn.1.bn.weight     False          128                [128]      0.687      0.125 torch.float32
            119               model.10.m.0.ffn.1.bn.bias     False          128                [128]     -4e-06   5.84e-05 torch.float32
            120                 model.13.cv1.conv.weight     False        49152     [128, 384, 1, 1]   -0.00111     0.0211 torch.float32
            121                   model.13.cv1.bn.weight     False          128                [128]      0.716      0.204 torch.float32
            122                     model.13.cv1.bn.bias     False          128                [128]     -0.281       0.84 torch.float32
            123                 model.13.cv2.conv.weight     False        24576     [128, 192, 1, 1]   -0.00186     0.0238 torch.float32
            124                   model.13.cv2.bn.weight     False          128                [128]      0.627      0.202 torch.float32
            125                     model.13.cv2.bn.bias     False          128                [128]     -0.391      0.631 torch.float32
            126             model.13.m.0.cv1.conv.weight     False        18432       [32, 64, 3, 3]   -0.00123     0.0208 torch.float32
            127               model.13.m.0.cv1.bn.weight     False           32                 [32]      0.692      0.133 torch.float32
            128                 model.13.m.0.cv1.bn.bias     False           32                 [32]     -0.328      0.693 torch.float32
            129             model.13.m.0.cv2.conv.weight     False        18432       [64, 32, 3, 3]   -0.00121     0.0198 torch.float32
            130               model.13.m.0.cv2.bn.weight     False           64                 [64]      0.883      0.202 torch.float32
            131                 model.13.m.0.cv2.bn.bias     False           64                 [64]      0.104       0.71 torch.float32
            132                 model.16.cv1.conv.weight     False        16384      [64, 256, 1, 1]   -0.00171     0.0242 torch.float32
            133                   model.16.cv1.bn.weight     False           64                 [64]       0.45      0.186 torch.float32
            134                     model.16.cv1.bn.bias     False           64                 [64]     0.0858      0.879 torch.float32
            135                 model.16.cv2.conv.weight     False         6144       [64, 96, 1, 1]   0.000416     0.0329 torch.float32
            136                   model.16.cv2.bn.weight     False           64                 [64]      0.543       0.26 torch.float32
            137                     model.16.cv2.bn.bias     False           64                 [64]      0.125      0.839 torch.float32
            138             model.16.m.0.cv1.conv.weight     False         4608       [16, 32, 3, 3]   -0.00168      0.029 torch.float32
            139               model.16.m.0.cv1.bn.weight     False           16                 [16]      0.661      0.142 torch.float32
            140                 model.16.m.0.cv1.bn.bias     False           16                 [16]     0.0959      0.687 torch.float32
            141             model.16.m.0.cv2.conv.weight     False         4608       [32, 16, 3, 3]  -0.000963     0.0312 torch.float32
            142               model.16.m.0.cv2.bn.weight     False           32                 [32]      0.768      0.213 torch.float32
            143                 model.16.m.0.cv2.bn.bias     False           32                 [32]     0.0741      0.762 torch.float32
            144                     model.17.conv.weight     False        36864       [64, 64, 3, 3]  -0.000343     0.0124 torch.float32
            145                       model.17.bn.weight     False           64                 [64]      0.771      0.176 torch.float32
            146                         model.17.bn.bias     False           64                 [64]     -0.622      0.593 torch.float32
            147                 model.19.cv1.conv.weight     False        24576     [128, 192, 1, 1]   -0.00103     0.0189 torch.float32
            148                   model.19.cv1.bn.weight     False          128                [128]      0.672      0.196 torch.float32
            149                     model.19.cv1.bn.bias     False          128                [128]     -0.267      0.603 torch.float32
            150                 model.19.cv2.conv.weight     False        24576     [128, 192, 1, 1]   -0.00052     0.0186 torch.float32
            151                   model.19.cv2.bn.weight     False          128                [128]      0.769      0.282 torch.float32
            152                     model.19.cv2.bn.bias     False          128                [128]     -0.423      0.652 torch.float32
            153             model.19.m.0.cv1.conv.weight     False        18432       [32, 64, 3, 3]   -0.00123     0.0177 torch.float32
            154               model.19.m.0.cv1.bn.weight     False           32                 [32]      0.907      0.228 torch.float32
            155                 model.19.m.0.cv1.bn.bias     False           32                 [32]     -0.252       0.78 torch.float32
            156             model.19.m.0.cv2.conv.weight     False        18432       [64, 32, 3, 3]   -0.00091     0.0175 torch.float32
            157               model.19.m.0.cv2.bn.weight     False           64                 [64]       1.19      0.249 torch.float32
            158                 model.19.m.0.cv2.bn.bias     False           64                 [64]     -0.168      0.578 torch.float32
            159                     model.20.conv.weight     False       147456     [128, 128, 3, 3]  -0.000307    0.00708 torch.float32
            160                       model.20.bn.weight     False          128                [128]      0.939      0.195 torch.float32
            161                         model.20.bn.bias     False          128                [128]     -0.584      0.345 torch.float32
            162                 model.22.cv1.conv.weight     False        98304     [256, 384, 1, 1]  -0.000877     0.0117 torch.float32
            163                   model.22.cv1.bn.weight     False          256                [256]        1.1      0.181 torch.float32
            164                     model.22.cv1.bn.bias     False          256                [256]     -0.656      0.443 torch.float32
            165                 model.22.cv2.conv.weight     False        98304     [256, 384, 1, 1]  -0.000582    0.00943 torch.float32
            166                   model.22.cv2.bn.weight     False          256                [256]       1.06       0.38 torch.float32
            167                     model.22.cv2.bn.bias     False          256                [256]     -0.563      0.425 torch.float32
            168             model.22.m.0.cv1.conv.weight     False         8192      [64, 128, 1, 1]   -0.00211       0.02 torch.float32
            169               model.22.m.0.cv1.bn.weight     False           64                 [64]      0.358      0.148 torch.float32
            170                 model.22.m.0.cv1.bn.bias     False           64                 [64]     -0.186       0.44 torch.float32
            171             model.22.m.0.cv2.conv.weight     False         8192      [64, 128, 1, 1]  -0.000656    0.00748 torch.float32
            172               model.22.m.0.cv2.bn.weight     False           64                 [64]       1.15      0.111 torch.float32
            173                 model.22.m.0.cv2.bn.bias     False           64                 [64]     -0.125      0.111 torch.float32
            174             model.22.m.0.cv3.conv.weight     False        16384     [128, 128, 1, 1]  -0.000809     0.0139 torch.float32
            175               model.22.m.0.cv3.bn.weight     False          128                [128]       1.33      0.368 torch.float32
            176                 model.22.m.0.cv3.bn.bias     False          128                [128]     -0.425      0.557 torch.float32
            177         model.22.m.0.m.0.cv1.conv.weight     False        36864       [64, 64, 3, 3]  -0.000836     0.0125 torch.float32
            178           model.22.m.0.m.0.cv1.bn.weight     False           64                 [64]       1.05      0.207 torch.float32
            179             model.22.m.0.m.0.cv1.bn.bias     False           64                 [64]     -0.781      0.649 torch.float32
            180         model.22.m.0.m.0.cv2.conv.weight     False        36864       [64, 64, 3, 3]  -0.000624      0.012 torch.float32
            181           model.22.m.0.m.0.cv2.bn.weight     False           64                 [64]       0.94       0.24 torch.float32
            182             model.22.m.0.m.0.cv2.bn.bias     False           64                 [64]      -0.25      0.694 torch.float32
            183         model.22.m.0.m.1.cv1.conv.weight     False        36864       [64, 64, 3, 3]  -0.000981     0.0117 torch.float32
            184           model.22.m.0.m.1.cv1.bn.weight     False           64                 [64]      0.926      0.233 torch.float32
            185             model.22.m.0.m.1.cv1.bn.bias     False           64                 [64]     -0.813      0.648 torch.float32
            186         model.22.m.0.m.1.cv2.conv.weight     False        36864       [64, 64, 3, 3]  -2.36e-06     0.0108 torch.float32
            187           model.22.m.0.m.1.cv2.bn.weight     False           64                 [64]        1.7      0.405 torch.float32
            188             model.22.m.0.m.1.cv2.bn.bias     False           64                 [64]      0.175      0.628 torch.float32
            189             model.23.cv2.0.0.conv.weight     False        36864       [64, 64, 3, 3]  -0.000899     0.0128 torch.float32
            190               model.23.cv2.0.0.bn.weight     False           64                 [64]      0.781      0.269 torch.float32
            191                 model.23.cv2.0.0.bn.bias     False           64                 [64]     -0.617      0.595 torch.float32
            192             model.23.cv2.0.1.conv.weight     False        36864       [64, 64, 3, 3]  -0.000355     0.0111 torch.float32
            193               model.23.cv2.0.1.bn.weight     False           64                 [64]       2.51        1.1 torch.float32
            194                 model.23.cv2.0.1.bn.bias     False           64                 [64]      0.856      0.831 torch.float32
            195                  model.23.cv2.0.2.weight     False         4096       [64, 64, 1, 1]   1.74e-07     0.0478 torch.float32
            196                    model.23.cv2.0.2.bias     False           64                 [64]      0.998      0.936 torch.float32
            197             model.23.cv2.1.0.conv.weight     False        73728      [64, 128, 3, 3]  -0.000417    0.00831 torch.float32
            198               model.23.cv2.1.0.bn.weight     False           64                 [64]       1.25      0.478 torch.float32
            199                 model.23.cv2.1.0.bn.bias     False           64                 [64]     -0.448      0.632 torch.float32
            200             model.23.cv2.1.1.conv.weight     False        36864       [64, 64, 3, 3]   -0.00022       0.01 torch.float32
            201               model.23.cv2.1.1.bn.weight     False           64                 [64]       2.67        1.1 torch.float32
            202                 model.23.cv2.1.1.bn.bias     False           64                 [64]      0.707      0.634 torch.float32
            203                  model.23.cv2.1.2.weight     False         4096       [64, 64, 1, 1]   1.17e-06     0.0506 torch.float32
            204                    model.23.cv2.1.2.bias     False           64                 [64]      0.998       1.05 torch.float32
            205             model.23.cv2.2.0.conv.weight     False       147456      [64, 256, 3, 3]  -0.000194    0.00543 torch.float32
            206               model.23.cv2.2.0.bn.weight     False           64                 [64]       1.44      0.342 torch.float32
            207                 model.23.cv2.2.0.bn.bias     False           64                 [64]     -0.353       0.62 torch.float32
            208             model.23.cv2.2.1.conv.weight     False        36864       [64, 64, 3, 3]  -0.000113    0.00956 torch.float32
            209               model.23.cv2.2.1.bn.weight     False           64                 [64]       3.07      0.958 torch.float32
            210                 model.23.cv2.2.1.bn.bias     False           64                 [64]      0.785      0.629 torch.float32
            211                  model.23.cv2.2.2.weight     False         4096       [64, 64, 1, 1]   3.79e-06     0.0563 torch.float32
            212                    model.23.cv2.2.2.bias     False           64                 [64]      0.998       1.02 torch.float32
            213           model.23.cv3.0.0.0.conv.weight     False          576        [64, 1, 3, 3]    0.00324     0.0279 torch.float32
            214             model.23.cv3.0.0.0.bn.weight     False           64                 [64]      0.552       0.33 torch.float32
            215               model.23.cv3.0.0.0.bn.bias     False           64                 [64]       1.13        1.1 torch.float32
            216           model.23.cv3.0.0.1.conv.weight     False         5120       [80, 64, 1, 1]    0.00167     0.0228 torch.float32
            217             model.23.cv3.0.0.1.bn.weight     False           80                 [80]      0.647      0.226 torch.float32
            218               model.23.cv3.0.0.1.bn.bias     False           80                 [80]     -0.188      0.805 torch.float32
            219           model.23.cv3.0.1.0.conv.weight     False          720        [80, 1, 3, 3]    0.00502     0.0317 torch.float32
            220             model.23.cv3.0.1.0.bn.weight     False           80                 [80]      0.634      0.294 torch.float32
            221               model.23.cv3.0.1.0.bn.bias     False           80                 [80]      0.976      0.678 torch.float32
            222           model.23.cv3.0.1.1.conv.weight     False         6400       [80, 80, 1, 1]   -0.00167     0.0238 torch.float32
            223             model.23.cv3.0.1.1.bn.weight     False           80                 [80]       2.74      0.478 torch.float32
            224               model.23.cv3.0.1.1.bn.bias     False           80                 [80]       1.22       1.94 torch.float32
            225                  model.23.cv3.0.2.weight     False         6400       [80, 80, 1, 1]   -0.00875     0.0481 torch.float32
            226                    model.23.cv3.0.2.bias     False           80                 [80]      -11.6      0.942 torch.float32
            227           model.23.cv3.1.0.0.conv.weight     False         1152       [128, 1, 3, 3]    0.00635     0.0214 torch.float32
            228             model.23.cv3.1.0.0.bn.weight     False          128                [128]      0.798      0.253 torch.float32
            229               model.23.cv3.1.0.0.bn.bias     False          128                [128]       0.54      0.764 torch.float32
            230           model.23.cv3.1.0.1.conv.weight     False        10240      [80, 128, 1, 1]   1.31e-05     0.0172 torch.float32
            231             model.23.cv3.1.0.1.bn.weight     False           80                 [80]       0.83      0.343 torch.float32
            232               model.23.cv3.1.0.1.bn.bias     False           80                 [80]    -0.0698      0.729 torch.float32
            233           model.23.cv3.1.1.0.conv.weight     False          720        [80, 1, 3, 3]    0.00383      0.028 torch.float32
            234             model.23.cv3.1.1.0.bn.weight     False           80                 [80]       0.74      0.378 torch.float32
            235               model.23.cv3.1.1.0.bn.bias     False           80                 [80]      0.731      0.748 torch.float32
            236           model.23.cv3.1.1.1.conv.weight     False         6400       [80, 80, 1, 1]  -0.000902     0.0225 torch.float32
            237             model.23.cv3.1.1.1.bn.weight     False           80                 [80]        2.9       0.75 torch.float32
            238               model.23.cv3.1.1.1.bn.bias     False           80                 [80]       1.27       1.68 torch.float32
            239                  model.23.cv3.1.2.weight     False         6400       [80, 80, 1, 1]    -0.0103     0.0488 torch.float32
            240                    model.23.cv3.1.2.bias     False           80                 [80]      -10.4      0.725 torch.float32
            241           model.23.cv3.2.0.0.conv.weight     False         2304       [256, 1, 3, 3]    0.00395     0.0176 torch.float32
            242             model.23.cv3.2.0.0.bn.weight     False          256                [256]       1.01      0.259 torch.float32
            243               model.23.cv3.2.0.0.bn.bias     False          256                [256]       0.15      0.372 torch.float32
            244           model.23.cv3.2.0.1.conv.weight     False        20480      [80, 256, 1, 1]  -0.000449     0.0123 torch.float32
            245             model.23.cv3.2.0.1.bn.weight     False           80                 [80]       1.08      0.303 torch.float32
            246               model.23.cv3.2.0.1.bn.bias     False           80                 [80]     -0.023      0.695 torch.float32
            247           model.23.cv3.2.1.0.conv.weight     False          720        [80, 1, 3, 3]     0.0051     0.0224 torch.float32
            248             model.23.cv3.2.1.0.bn.weight     False           80                 [80]      0.838      0.391 torch.float32
            249               model.23.cv3.2.1.0.bn.bias     False           80                 [80]      0.683      0.798 torch.float32
            250           model.23.cv3.2.1.1.conv.weight     False         6400       [80, 80, 1, 1]   -0.00218       0.02 torch.float32
            251             model.23.cv3.2.1.1.bn.weight     False           80                 [80]       3.16      0.736 torch.float32
            252               model.23.cv3.2.1.1.bn.bias     False           80                 [80]       1.54       1.46 torch.float32
            253                  model.23.cv3.2.2.weight     False         6400       [80, 80, 1, 1]     -0.011     0.0476 torch.float32
            254                    model.23.cv3.2.2.bias     False           80                 [80]      -9.08       0.75 torch.float32
            255                 model.23.dfl.conv.weight     False           16        [1, 16, 1, 1]        7.5       4.76 torch.float32
            YOLO11n summary: 319 layers, 2,624,080 parameters, 0 gradients, 6.6 GFLOPs
            ```

    === "Custom Model"

        Unless the custom model file is in your current working directory, try using the full path to the model weights.

        ```bash
        yolo info model=/path/to/yolo_model.pt
        ```

        ??? question "Model Filenames"
            Having the model type in the filename will ensure the correct `class` is used. Examples:

            ```diff
            # Rename to include model-type
            - `best.pt`
            + `best_yolo.pt`  # include "yolo" in filename

            - `best.pt`
            + `best_rtdetr.pt`  # include "rtdetr" in filename
            ```

### Applications for `yolo info`

What's the advantage of using the `yolo info` CLI command? With the CLI command, it's much easier to write a script or use built-in terminal commands to search for model weights that have specific qualities. While testing or experimenting with model training, you might have models trained on a variety of `ultralytics` versions, with different class names, or using a variety of [training arguments](../modes/train.md#train-settings). You can use `yolo info` to help simplify finding these files by searching the command output and selecting files with matching criteria.

!!! example "Find model trained using specific `ultralytics` version"

    These examples show how to use the `yolo info` CLI command to find model weights files that have been trained using a specific version of `ultraltyics`. These examples are meant to be basic and easy to customize. One could easily modify an example to look for any specific characteristic in the output instead.

    !!! warning

        These example may need modification depending on your operating system or terminal and are **not** guaranteed to work.

    === "Bash Echo Matches"

        ```bash
        # Tested with GNU bash, version 5.1.16[1]-release on Ubuntu 22.04.3 LTS

        ver="8.2.100" # (1)
        filenames=("yolov8n.pt" "yolo11n.pt") # (2)

        # Echo filenames from array that match condition
        for filename in "${filenames[@]}"; \
            do output=$(yolo info model="$filename"); \
            [[ "$output" == *"$ver"* ]] && echo "$filename"; done

        # Echo all *.onnx files found in directory with matching string
        for file in ./*.onnx; do [[ -f "$file" ]] && \
            output=$(yolo info model="$file") && \
            [[ "$output" == *"$ver"* ]] && echo "$file"; done
        ```

        1. Set version to find
        2. List filenames for weights to check

    === "Bash Matches to Variable"

        ```bash
        # Tested with GNU bash, version 5.1.16[1]-release on Ubuntu 22.04.3 LTS

        ver="8.2.100" # (1)
        filenames=("yolov8n.pt" "yolo11n.pt") # (2)
        models=() # (3)

        # Save filenames that match into "models" variable instead of echo; then echo "models"
        for filename in "${filenames[@]}"; do
            output=$(yolo info model="$filename")
            if [[ "$output" == *"$ver"* ]]; then
                models+=("$filename") # (4)
            fi
        done
        echo "Matching models: ${models[@]}"

        # Save filenames that match into "models" variable instead of echo; then echo "models"
        models=()
        for file in ./*.pt; do
            if [[ -f "$file" ]]; then
                output=$(yolo info model="$file")
                if [[ "$output" == *"$ver"* ]]; then
                    models+=("$file") # (5)
                fi
            fi
        done
        echo "Matching models: ${models[@]}"
        ```

        1. Set version to find.
        2. List filenames for weights to check.
        3. Variable to add filenames with matching conditions into.
        4. Append filenames when version number found in output of `yolo info` command.
        5. Append filenames when version number found in output of `yolo info` command.

    === "PowerShell"

        ```powershell
        # Tested with PowerShell 7.4.5 on Windows 10

        $ver = "8.2.100" # (1)
        $files = @("yolov8n.pt", "yolo11n.pt") # (2)
        $models = @() # (3)

        # Example with array of filenames (returns filename string)
        $files | ForEach-Object {$out=yolo info model=$_;if ($out | Select-String $ver){$models += $_}}

        # Example with files found in directory (returns `FileInfo` object)
        $model_objs = @()
        $files = Get-ChildItem *.onnx
        $files | ForEach-Object {$out=yolo info model=$_;if ($out | Select-String $ver){$model_objs += $_}}
        ```

        1. Set version to find.
        2. List filenames for weights to check.
        3. Variable `array` to add filenames with matching conditions into.

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
