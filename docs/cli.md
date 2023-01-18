If you want to train, validate or run inference on models and don't need to make any modifications to the code, using
YOLO command line interface is the easiest way to get started.

!!! tip "Syntax"

    ```bash
    yolo task=detect    mode=train    model=yolov8n.yaml      args...
              classify       predict        yolov8n-cls.yaml  args...
              segment        val            yolov8n-seg.yaml  args...
                             export         yolov8n.pt        format=onnx  args...
    ```

The default arguments can be overridden directly by passing custom `arg=val` covered in the next section. You can run
any supported task by setting `task` and `mode` in CLI.
=== "Training"

    |                  | task       | example                                                           |
    |------------------|------------|-------------------------------------------------------------------|
    | Detection        | `detect`   | `yolo detect train data=coco128.yaml model=yolov8n.pt`            |
    | Segmentation     | `segment`  | `yolo segment train data=coco128-seg.yaml model=yolov8n-seg.pt`   |
    | Classification   | `classify` | `yolo classify train data=mnist160 model=yolov8n-cls.pt`          |

=== "Prediction"

    |                  | task       | example                                                           |
    |------------------|------------|-------------------------------------------------------------------|
    | Detection        | `detect`   | `yolo detect predict data=coco128.yaml model=yolov8n.pt`          |
    | Segmentation     | `segment`  | `yolo segment predict data=coco128-seg.yaml model=yolov8n-seg.pt` |
    | Classification   | `classify` | `yolo classify predict data=mnist160 model=yolov8n-cls.pt`        |


=== "Validation"

    |                  | task       | example                                                         |
    |------------------|------------|-----------------------------------------------------------------|
    | Detection        | `detect`   | `yolo detect val data=coco128.yaml model=yolov8n.pt`            |
    | Segmentation     | `segment`  | `yolo segment val data=coco128-seg.yaml model=yolov8n-seg.pt`   |
    | Classification   | `classify` | `yolo classify val data=mnist160 model=yolov8n-cls.pt`          |


!!! note ""

    <b>Note:</b> The arguments don't require `'--'` prefix. These are reserved for special commands covered later

---

## Overriding default config arguments

Default arguments can be overriden by simply passing them as arguments in the CLI.

!!! tip ""

    === "Example 1"
        Train a detection model for `10 epochs` with `learning_rate` of `0.01`
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Example 2"
        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo segment predict model=yolov8n-seg.pt source=https://youtu.be/Zgi9g1ksQHc imgsz=320
        ```

    === "Example 3"
        Validate a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

---

## Overriding default config file

You can override config file entirely by passing a new file. You can create a copy of default config file in your
current working dir as follows:

```bash
yolo copy-config
```

You can then use `cfg=default_copy.yaml` command to pass the new config file along with any addition args:

```bash
yolo cfg=default_copy.yaml args...
```

??? example

    === "Command"
        ```bash
        yolo copy-config
        yolo cfg=default_copy.yaml args...
        ```
