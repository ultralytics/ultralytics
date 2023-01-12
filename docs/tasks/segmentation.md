Instance segmentation goes a step further than object detection and involves identifying individual objects in an image
and segmenting them from the rest of the image.

<img width="1024" src="https://user-images.githubusercontent.com/26833433/212053258-b6948968-4797-4a2b-9247-bfdda77521de.png">

The output of an instance segmentation model is a set of masks or
contours that outline each object in the image, along with class labels and confidence scores for each object. Instance
segmentation is useful when you need to know not only where objects are in an image, but also what their exact shape is.

!!! tip "Tip"

    YOLOv8 _segmentation_ models use the `-seg` suffix, i.e. `yolov8n-seg.pt` and are pretrained on COCO.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/v8/seg){.md-button .md-button--primary}

## Usage examples

!!! example "1. Train"
Train YOLOv8n-seg on the COCO128-seg dataset for 100 epochs at image size 640.

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-seg.yaml")  # build a new model from scratch
        model = YOLO("yolov8n-seg.pt")  # load a pretrained model (recommended for training)
        
        # Train the model
        results = model.train(data="coco128-seg.yaml", epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        yolo task=segment mode=train data=coco128-seg.yaml model=yolov8n-seg.pt epochs=100 imgsz=640
        ```

!!! example "2. Val"
Validate trained YOLOv8n-seg model accuracy on the COCO128-seg dataset. No argument need to passed as the `model`
retains it's training `data` and arguments as model attributes.

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Validate the model
        results = model.val()  # no arguments needed, dataset and settings remembered
        ```
    === "CLI"
    
        ```bash
        yolo task=segment mode=val model=yolov8n-seg.pt  # val official model
        yolo task=segment mode=val model=path/to/best.pt  # val custom model
        ```

!!! example "3. Predict"
Use a trained YOLOv8n-seg model to run predictions on images.

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Predict with the model
        results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
        ```
    === "CLI"
    
        ```bash
        yolo task=segment mode=predict model=yolov8n-seg.pt source="https://ultralytics.com/images/bus.jpg"  # predict with official model
        yolo task=segment mode=predict model=path/to/best.pt source="https://ultralytics.com/images/bus.jpg"  # predict with custom model
        ```

!!! example "4. Export"
Export a YOLOv8n-seg model to a different format like ONNX, CoreML, etc.

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n-seg.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom trained
        
        # Export the model
        model.export(format="onnx")
        ```
    === "CLI"
    
        ```bash
        yolo mode=export model=yolov8n-seg.pt format=onnx  # export official model
        yolo mode=export model=path/to/best.pt format=onnx  # export custom trained model
        ```

    Available YOLOv8-seg export formats include:

    | Format                | `format=argument` | Model                       |
    |-----------------------|-------------------|-----------------------------|
    | PyTorch               | -                 | yolov8n-seg.pt              |
    | TorchScript           | `torchscript`     | yolov8n-seg.torchscript     |
    | ONNX                  | `onnx`            | yolov8n-seg.onnx            |
    | OpenVINO              | `openvino`        | yolov8n-seg_openvino_model/ |
    | TensorRT              | `engine`          | yolov8n-seg.engine          |
    | CoreML                | `coreml`          | yolov8n-seg.mlmodel         |
    | TensorFlow SavedModel | `saved_model`     | yolov8n-seg_saved_model/    |
    | TensorFlow GraphDef   | `pb`              | yolov8n-seg.pb              |
    | TensorFlow Lite       | `tflite`          | yolov8n-seg.tflite          |
    | TensorFlow Edge TPU   | `edgetpu`         | yolov8n-seg_edgetpu.tflite  |
    | TensorFlow.js         | `tfjs`            | yolov8n-seg_web_model/      |
    | PaddlePaddle          | `paddle`          | yolov8n-seg_paddle_model/   |


