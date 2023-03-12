Object detection is a task that involves identifying the location and class of objects in an image or video stream.

<img width="1024" src="https://user-images.githubusercontent.com/26833433/212094133-6bb8c21c-3d47-41df-a512-81c5931054ae.png">

The output of an object detector is a set of bounding boxes that enclose the objects in the image, along with class
labels
and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a
scene, but don't need to know exactly where the object is or its exact shape.

!!! tip "Tip"

    YOLOv8 _detection_ models have no suffix and are the default YOLOv8 models, i.e. `yolov8n.pt` and are pretrained on COCO.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/v8){ .md-button .md-button--primary}

## Train

Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640. For a full list of available arguments see
the [Configuration](../usage/cfg.md) page.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
        
        # Train the model
        model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

## Val

Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's
training `data` and arguments as model attributes.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map    # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps   # a list contains map50-95 of each category
        ```
    === "CLI"
    
        ```bash
        yolo detect val model=yolov8n.pt  # val official model
        yolo detect val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n model to run predictions on images.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```
    === "CLI"
    
        ```bash
        yolo detect predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"  # predict with official model
        yolo detect predict model=path/to/best.pt source="https://ultralytics.com/images/bus.jpg"  # predict with custom model
        ```

Read more details of `predict` in our [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLOv8n model to a different format like ONNX, CoreML, etc.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained
        
        # Export the model
        model.export(format="onnx")
        ```
    === "CLI"
    
        ```bash
        yolo export model=yolov8n.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8 export formats are in the table below. You can predict or validate directly on exported models,
i.e. `yolo predict model=yolov8n.onnx`.

| Format                                                             | `format` Argument | Model                     | Metadata |
|--------------------------------------------------------------------|-------------------|---------------------------|----------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n.pt`              | ✅        |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n.torchscript`     | ✅        |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n.onnx`            | ✅        |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n_openvino_model/` | ✅        |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n.engine`          | ✅        |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n.mlmodel`         | ✅        |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n_saved_model/`    | ✅        |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n.pb`              | ❌        |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n.tflite`          | ✅        |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n_edgetpu.tflite`  | ✅        |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n_web_model/`      | ✅        |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n_paddle_model/`   | ✅        |
