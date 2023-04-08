Pose estimation is a task that involves identifying the location of specific points in an image, usually referred
to as keypoints. The keypoints can represent various parts of the object such as joints, landmarks, or other distinctive
features. The locations of the keypoints are usually represented as a set of 2D `[x, y]` or 3D `[x, y, visible]`
coordinates.

<img width="1024" src="https://user-images.githubusercontent.com/26833433/212094133-6bb8c21c-3d47-41df-a512-81c5931054ae.png">

The output of a pose estimation model is a set of points that represent the keypoints on an object in the image, usually
along with the confidence scores for each point. Pose estimation is a good choice when you need to identify specific
parts of an object in a scene, and their location in relation to each other.

!!! tip "Tip"

    YOLOv8 _pose_ models use the `-pose` suffix, i.e. `yolov8n-pose.pt`. These models are trained on the [COCO keypoints](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco-pose.yaml) dataset and are suitable for a variety of pose estimation tasks.

## [Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/v8)

YOLOv8 pretrained Pose models are shown here. Detect, Segment and Pose models are pretrained on
the [COCO](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/coco.yaml) dataset, while Classify
models are pretrained on
the [ImageNet](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/datasets/ImageNet.yaml) dataset.

[Models](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models) download automatically from the latest
Ultralytics [release](https://github.com/ultralytics/assets/releases) on first use.

| Model                                                                                                | size<br><sup>(pixels) | mAP<sup>box<br>50-95 | mAP<sup>pose<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
|------------------------------------------------------------------------------------------------------|-----------------------|----------------------|-----------------------|--------------------------------|-------------------------------------|--------------------|-------------------|
| [YOLOv8n-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt)       | 640                   | -                    | 49.7                  | 131.8                          | 1.18                                | 3.3                | 9.2               |
| [YOLOv8s-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s-pose.pt)       | 640                   | -                    | 59.2                  | 233.2                          | 1.42                                | 11.6               | 30.2              |
| [YOLOv8m-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m-pose.pt)       | 640                   | -                    | 63.6                  | 456.3                          | 2.00                                | 26.4               | 81.0              |
| [YOLOv8l-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l-pose.pt)       | 640                   | -                    | 67.0                  | 784.5                          | 2.59                                | 44.4               | 168.6             |
| [YOLOv8x-pose](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose.pt)       | 640                   | -                    | 68.9                  | 1607.1                         | 3.73                                | 69.4               | 263.2             |
| [YOLOv8x-pose-p6](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x-pose-p6.pt) | 1280                  | -                    | 71.5                  | 4088.7                         | 10.04                               | 99.1               | 1066.4            |

- **mAP<sup>val</sup>** values are for single-model single-scale on [COCO Keypoints val2017](http://cocodataset.org)
  dataset.
  <br>Reproduce by `yolo val pose data=coco-pose.yaml device=0`
- **Speed** averaged over COCO val images using an [Amazon EC2 P4d](https://aws.amazon.com/ec2/instance-types/p4/)
  instance.
  <br>Reproduce by `yolo val pose data=coco8-pose.yaml batch=1 device=0|cpu`

## Train

Train a YOLOv8-pose model on the COCO128-pose dataset.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-pose.yaml')  # build a new model from YAML
        model = YOLO('yolov8n-pose.pt')  # load a pretrained model (recommended for training)
        model = YOLO('yolov8n-pose.yaml').load('yolov8n-pose.pt')  # build from YAML and transfer weights
        
        # Train the model
        model.train(data='coco8-pose.yaml', epochs=100, imgsz=640)
        ```
    === "CLI"
    
        ```bash
        # Build a new model from YAML and start training from scratch
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml epochs=100 imgsz=640

        # Start training from a pretrained *.pt model
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.pt epochs=100 imgsz=640

        # Build a new model from YAML, transfer pretrained weights to it and start training
        yolo pose train data=coco8-pose.yaml model=yolov8n-pose.yaml pretrained=yolov8n-pose.pt epochs=100 imgsz=640
        ```

## Val

Validate trained YOLOv8n-pose model accuracy on the COCO128-pose dataset. No argument need to passed as the `model`
retains it's
training `data` and arguments as model attributes.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load an official model
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
        yolo pose val model=yolov8n-pose.pt  # val official model
        yolo pose val model=path/to/best.pt  # val custom model
        ```

## Predict

Use a trained YOLOv8n-pose model to run predictions on images.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom model
        
        # Predict with the model
        results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
        ```
    === "CLI"
    
        ```bash
        yolo pose predict model=yolov8n-pose.pt source='https://ultralytics.com/images/bus.jpg'  # predict with official model
        yolo pose predict model=path/to/best.pt source='https://ultralytics.com/images/bus.jpg'  # predict with custom model
        ```

See full `predict` mode details in the [Predict](https://docs.ultralytics.com/modes/predict/) page.

## Export

Export a YOLOv8n Pose model to a different format like ONNX, CoreML, etc.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n-pose.pt')  # load an official model
        model = YOLO('path/to/best.pt')  # load a custom trained
        
        # Export the model
        model.export(format='onnx')
        ```
    === "CLI"
    
        ```bash
        yolo export model=yolov8n-pose.pt format=onnx  # export official model
        yolo export model=path/to/best.pt format=onnx  # export custom trained model
        ```

Available YOLOv8-pose export formats are in the table below. You can predict or validate directly on exported models,
i.e. `yolo predict model=yolov8n-pose.onnx`. Usage examples are shown for your model after export completes.

| Format                                                             | `format` Argument | Model                          | Metadata |
|--------------------------------------------------------------------|-------------------|--------------------------------|----------|
| [PyTorch](https://pytorch.org/)                                    | -                 | `yolov8n-pose.pt`              | ✅        |
| [TorchScript](https://pytorch.org/docs/stable/jit.html)            | `torchscript`     | `yolov8n-pose.torchscript`     | ✅        |
| [ONNX](https://onnx.ai/)                                           | `onnx`            | `yolov8n-pose.onnx`            | ✅        |
| [OpenVINO](https://docs.openvino.ai/latest/index.html)             | `openvino`        | `yolov8n-pose_openvino_model/` | ✅        |
| [TensorRT](https://developer.nvidia.com/tensorrt)                  | `engine`          | `yolov8n-pose.engine`          | ✅        |
| [CoreML](https://github.com/apple/coremltools)                     | `coreml`          | `yolov8n-pose.mlmodel`         | ✅        |
| [TF SavedModel](https://www.tensorflow.org/guide/saved_model)      | `saved_model`     | `yolov8n-pose_saved_model/`    | ✅        |
| [TF GraphDef](https://www.tensorflow.org/api_docs/python/tf/Graph) | `pb`              | `yolov8n-pose.pb`              | ❌        |
| [TF Lite](https://www.tensorflow.org/lite)                         | `tflite`          | `yolov8n-pose.tflite`          | ✅        |
| [TF Edge TPU](https://coral.ai/docs/edgetpu/models-intro/)         | `edgetpu`         | `yolov8n-pose_edgetpu.tflite`  | ✅        |
| [TF.js](https://www.tensorflow.org/js)                             | `tfjs`            | `yolov8n-pose_web_model/`      | ✅        |
| [PaddlePaddle](https://github.com/PaddlePaddle)                    | `paddle`          | `yolov8n-pose_paddle_model/`   | ✅        |

See full `export` details in the [Export](https://docs.ultralytics.com/modes/export/) page.
