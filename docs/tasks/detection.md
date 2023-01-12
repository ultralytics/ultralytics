Object detection is a task that involves identifying the location and class of objects in an image or video stream.

<img width="1024" src="https://user-images.githubusercontent.com/26833433/212053258-b6948968-4797-4a2b-9247-bfdda77521de.png">

The output of an object detector is a set of bounding boxes that enclose the objects in the image, along with class
labels
and confidence scores for each box. Object detection is a good choice when you need to identify objects of interest in a
scene, but don't need to know exactly where the object is or its exact shape.

!!! tip "Tip"

    YOLOv8 _detection_ models have no suffix and are the default YOLOv8 models, i.e. `yolov8n.pt` and are pretrained on COCO.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/v8){ .md-button .md-button--primary}

## Usage examples

!!! example "1. Train"
Train YOLOv8n on the COCO128 dataset for 100 epochs at image size 640.

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.yaml")  # build a new model from scratch
        model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)
        
        # Train the model
        results = model.train(data="coco128.yaml", epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        yolo task=detect mode=train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640
        ```

!!! example "2. Val"
Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's
training `data` and arguments as model attributes.

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Validate the model
        results = model.val()  # no arguments needed, dataset and settings remembered
        ```
    === "CLI"
    
        ```bash
        yolo task=detect mode=val model=yolov8n.pt  # val official model
        yolo task=detect mode=val model=path/to/best.pt  # val custom model
        ```

!!! example "3. Predict"
Use a trained YOLOv8n model to run predictions on images.

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
        yolo task=detect mode=predict model=yolov8n.pt source="https://ultralytics.com/images/bus.jpg"  # predict with official model
        yolo task=detect mode=predict model=path/to/best.pt source="https://ultralytics.com/images/bus.jpg"  # predict with custom model
        ```

!!! example "4. Export"
Export a YOLOv8n model to a different format like ONNX, CoreML, etc.

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
        yolo mode=export model=yolov8n.pt format=onnx  # export official model
        yolo mode=export model=path/to/best.pt format=onnx  # export custom trained model
        ```

    Available YOLOv8 export formats include:

    | Format                | `format=argument` | Model                   |
    |-----------------------|-------------------|-------------------------|
    | PyTorch               | -                 | yolov8n.pt              |
    | TorchScript           | `torchscript`     | yolov8n.torchscript     |
    | ONNX                  | `onnx`            | yolov8n.onnx            |
    | OpenVINO              | `openvino`        | yolov8n_openvino_model/ |
    | TensorRT              | `engine`          | yolov8n.engine          |
    | CoreML                | `coreml`          | yolov8n.mlmodel         |
    | TensorFlow SavedModel | `saved_model`     | yolov8n_saved_model/    |
    | TensorFlow GraphDef   | `pb`              | yolov8n.pb              |
    | TensorFlow Lite       | `tflite`          | yolov8n.tflite          |
    | TensorFlow Edge TPU   | `edgetpu`         | yolov8n_edgetpu.tflite  |
    | TensorFlow.js         | `tfjs`            | yolov8n_web_model/      |
    | PaddlePaddle          | `paddle`          | yolov8n_paddle_model/   |
