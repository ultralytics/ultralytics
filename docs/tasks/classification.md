Image classification is the simplest of the three tasks and involves classifying an entire image into one of a set of
predefined classes.

<img width="1024" src="https://user-images.githubusercontent.com/26833433/212094133-6bb8c21c-3d47-41df-a512-81c5931054ae.png">

The output of an image classifier is a single class label and a confidence score. Image
classification is useful when you need to know only what class an image belongs to and don't need to know where objects
of that class are located or what their exact shape is.

!!! tip "Tip"

    YOLOv8 _classification_ models use the `-cls` suffix, i.e. `yolov8n-cls.pt` and are pretrained on ImageNet.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/v8/cls){.md-button .md-button--primary}

## Train

Train YOLOv8n-cls on the MNIST160 dataset for 100 epochs at image size 64. For a full list of available arguments
see the [Configuration](../cfg.md) page.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-cls.yaml")  # build a new model from scratch
        model = YOLO("yolov8n-cls.pt")  # load a pretrained model (recommended for training)
        
        # Train the model
        results = model.train(data="mnist160", epochs=100, imgsz=64)
        ```
    === "CLI"
    
        ```bash
        yolo classify train data=mnist160 model=yolov8n-cls.pt epochs=100 imgsz=64
        ```

## Val

Validate trained YOLOv8n-cls model accuracy on the MNIST160 dataset. No argument need to passed as the `model` retains
it's training `data` and arguments as model attributes.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Validate the model
        results = model.val()  # no arguments needed, dataset and settings remembered
        ```
    === "CLI"
    
        ```bash
        yolo classify val model=yolov8n-cls.pt  # val official model
        yolo classify val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-cls model to run predictions on images.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```
    === "CLI"
    
        ```bash
        yolo classify predict model=yolov8n-cls.pt source="https://ultralytics.com/images/bus.jpg"  # predict with official model
        yolo classify predict model=path/to/best.pt source="https://ultralytics.com/images/bus.jpg"  # predict with custom model
        ```

## Export

Export a YOLOv8n-cls model to a different format like ONNX, CoreML, etc.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-cls.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained
        
        # Export the model
        model.export(format="onnx")
        ```
    === "CLI"
    
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

    Available YOLOv8-cls export formats include:

    | Format                                                                     | `format=`     | Model                         |
    |----------------------------------------------------------------------------|---------------|-------------------------------|
    | [PyTorch](https://pytorch.org/)                                            | -             | `yolov8n-cls.pt`              |
    | [TorchScript](https://pytorch.org/docs/stable/jit.html)                    | `torchscript` | `yolov8n-cls.torchscript`     |
    | [ONNX](https://onnx.ai/)                                                   | `onnx`        | `yolov8n-cls.onnx`            |
    | [OpenVINO](https://docs.openvino.ai/latest/index.html)                     | `openvino`    | `yolov8n-cls_openvino_model/` |
    | [TensorRT](https://developer.nvidia.com/tensorrt)                          | `engine`      | `yolov8n-cls.engine`          |
    | [CoreML](https://github.com/apple/coremltools)                             | `coreml`      | `yolov8n-cls.mlmodel`         |
    | [TensorFlow SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model` | `yolov8n-cls_saved_model/`    |
    | [TensorFlow GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`          | `yolov8n-cls.pb`              |
    | [TensorFlow Lite](https://www.tensorflow.org/lite)                         | `tflite`      | `yolov8n-cls.tflite`          |
    | [TensorFlow Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`     | `yolov8n-cls_edgetpu.tflite`  |
    | [TensorFlow.js](https://www.tensorflow.org/js)                             | `tfjs`        | `yolov8n-cls_web_model/`      |
    | [PaddlePaddle](https://github.com/PaddlePaddle)                            | `paddle`      | `yolov8n-cls_paddle_model/`   |
    
