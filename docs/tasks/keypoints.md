Key Point Estimation is a task that involves identifying the location of specific points in an image, usually referred
to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive
features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` or 3D `[x, y, visible]`
coordinates.

<img width="1024" src="https://user-images.githubusercontent.com/26833433/212094133-6bb8c21c-3d47-41df-a512-81c5931054ae.png">

The output of a keypoint detector is a set of points that represent the keypoints on the object in the image, usually
along with the confidence scores for each point. Keypoint estimation is a good choice when you need to identify specific
parts of an object in a scene, and their location in relation to each other.

!!! tip "Tip"

    YOLOv8 _keypoints_ models use the `-kpts` suffix, i.e. `yolov8n-kpts.pt`. These models are trained on the COCO dataset and are suitable for a variety of keypoint estimation tasks.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/v8){ .md-button .md-button--primary}

## Train TODO

Train an OpenPose model on a custom dataset of keypoints using the OpenPose framework. For more information on how to
train an OpenPose model on a custom dataset, see the OpenPose Training page.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.yaml')  # build a new model from YAML
        model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # build from YAML and transfer weights
        
        # Train the model
        model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        # Build a new model from YAML and start training from scratch
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

## Val TODO

Validate trained YOLOv8n model accuracy on the COCO128 dataset. No argument need to passed as the `model` retains it's
training `data` and arguments as model attributes.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model
        
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

## Predict TODO

Use a trained YOLOv8n model to run predictions on images.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model
        
        # Predict with the model
        results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
        ```
    === "CLI"
    
        ```bash
        yolo detect predict model=yolov8n.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo detect predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

Read more details of `predict` in our [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export TODO

Export a YOLOv8n model to a different format like ONNX, CoreML, etc.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained
        
        # Export the model
        model.export(format='onnx')
        ```
    === "CLI"
    
        ```bash
        yolo export model=yolov8n.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8-pose export formats are in the table below. You can predict or validate directly on exported models,
i.e. `yolo predict model=yolov8n-pose.onnx`.

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
