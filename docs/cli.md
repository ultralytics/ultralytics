The YOLO Command Line Interface (CLI) is the easiest way to get started training, validating, predicting and exporting
YOLOv8 models.

The `yolo` command is used for all actions:

!!! example ""

    === "CLI"
    
        ```bash
        yolo TASK MODE ARGS
        ```

Where:

- `TASK` (optional) is one of `[detect, segment, classify]`. If it is not passed explicitly YOLOv8 will try to guess
  the `TASK` from the model type.
- `MODE` (required) is one of `[train, val, predict, export]`
- `ARGS` (optional) are any number of custom `arg=value` pairs like `imgsz=320` that override defaults.
  For a full list of available `ARGS` see the [Configuration](cfg.md) page and `defaults.yaml`
  GitHub [source](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/yolo/cfg/default.yaml).

!!! note ""

    <b>Note:</b> Arguments MUST be passed as `arg=val` with an equals sign and a space between `arg=val` pairs

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

## Train

Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640. For a full list of available arguments see
the [Configuration](cfg.md) page.

!!! example ""

    === "CLI"
    
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
        
        # Train the model
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```

## Val

Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's
training `data` and arguments as model attributes.

!!! example ""

    === "CLI"
    
        ```bash
        yolo detect val model=yolov8n.pt  # val official model
        yolo detect val model=path/to/best.pt  # val custom model
        ```

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Validate the model
        results = model.val()  # no arguments needed, dataset and settings remembered
        ```

## Predict

Use a trained YOLOv8n model to run predictions on images.

!!! example ""

    === "CLI"
    
        ```bash
        yolo detect predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"  # predict with official model
        yolo detect predict model=path/to/best.pt source="https://ultralytics.com/images/bus.jpg"  # predict with custom model
        ```

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```

## Export

Export a YOLOv8n model to a different format like ONNX, CoreML, etc.

!!! example ""

    === "CLI"
    
        ```bash
        yolo export model=yolov8n.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained
        
        # Export the model
        model.export(format="onnx")
        ```

    Available YOLOv8 export formats include:
    
    | Format                                                                     | `format=`          | Model                     |
    |----------------------------------------------------------------------------|--------------------|---------------------------|
    | [PyTorch](https://pytorch.org/)                                            | -                  | `yolov8n.pt`              |
    | [TorchScript](https://pytorch.org/docs/stable/jit.html)                    | `torchscript`      | `yolov8n.torchscript`     |
    | [ONNX](https://onnx.ai/)                                                   | `onnx`             | `yolov8n.onnx`            |
    | [OpenVINO](https://docs.openvino.ai/latest/index.html)                     | `openvino`         | `yolov8n_openvino_model/` |
    | [TensorRT](https://developer.nvidia.com/tensorrt)                          | `engine`           | `yolov8n.engine`          |
    | [CoreML](https://github.com/apple/coremltools)                             | `coreml`           | `yolov8n.mlmodel`         |
    | [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`      | `yolov8n_saved_model/`    |
    | [TensorFlow GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`               | `yolov8n.pb`              |
    | [TensorFlow Lite](https://www.tensorflow.org/lite)                         | `tflite`           | `yolov8n.tflite`          |
    | [TensorFlow Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`          | `yolov8n_edgetpu.tflite`  |
    | [TensorFlow.js](https://www.tensorflow.org/js)                             | `tfjs`             | `yolov8n_web_model/`      |
    | [PaddlePaddle](https://github.com/PaddlePaddle)                            | `paddle`           | `yolov8n_paddle_model/`   |

---

## Overriding default arguments

Default arguments can be overriden by simply passing them as arguments in the CLI in `arg=value` pairs.

!!! tip ""

    === "Example 1"
        Train a detection model for `10 epochs` with `learning_rate` of `0.01`
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "Example 2"
        Predict a YouTube video using a pretrained segmentation model at image size 320:
        ```bash
        yolo segment predict model=yolov8n-seg.pt source='https://youtu.be/Zgi9g1ksQHc' imgsz=320
        ```

    === "Example 3"
        Validate a pretrained detection model at batch-size 1 and image size 640:
        ```bash
        yolo detect val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

---

## Overriding default config file

You can override the `default.yaml` config file entirely by passing a new file with the `cfg` arguments,
i.e. `cfg=custom.yaml`.

To do this first create a copy of `default.yaml` in your current working dir with the `yolo copy-config` command.

This will create `default_copy.yaml`, which you can then pass as `cfg=default_copy.yaml` along with any additional args,
like `imgsz=320` in this example:

!!! example ""

    === "CLI"
        ```bash
        yolo copy-config
        yolo cfg=default_copy.yaml imgsz=320
        ```