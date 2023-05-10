---
comments: true
description: Integrate YOLOv8 in Python. Load, use pretrained models, train, and infer images. Export to ONNX. Track objects in videos.
---

# Python Usage

Welcome to the YOLOv8 Python Usage documentation! This guide is designed to help you seamlessly integrate YOLOv8 into
your Python projects for object detection, segmentation, and classification. Here, you'll learn how to load and use
pretrained models, train new models, and perform predictions on images. The easy-to-use Python interface is a valuable
resource for anyone looking to incorporate YOLOv8 into their Python projects, allowing you to quickly implement advanced
object detection capabilities. Let's get started!

For example, users can load a model, train it, evaluate its performance on a validation set, and even export it to ONNX
format with just a few lines of code.

!!! example "Python"

    ```python
    from ultralytics import YOLO
    
    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')
    
    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')
    
    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='coco128.yaml', epochs=3)
    
    # Evaluate the model's performance on the validation set
    results = model.val()
    
    # Perform object detection on an image using the model
    results = model('https://ultralytics.com/images/bus.jpg')
    
    # Export the model to ONNX format
    success = model.export(format='onnx')
    ```

## [Train](../modes/train.md)

Train mode is used for training a YOLOv8 model on a custom dataset. In this mode, the model is trained using the
specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can
accurately predict the classes and locations of objects in an image.

!!! example "Train"

    === "From pretrained(recommended)"
        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.pt') # pass any model type
        model.train(epochs=5)
        ```

    === "From scratch"
        ```python
        from ultralytics import YOLO

        model = YOLO('yolov8n.yaml')
        model.train(data='coco128.yaml', epochs=5)
        ```

    === "Resume"
        ```python
        model = YOLO("last.pt")
        model.train(resume=True)
        ```

[Train Examples](../modes/train.md){ .md-button .md-button--primary}

## [Val](../modes/val.md)

Val mode is used for validating a YOLOv8 model after it has been trained. In this mode, the model is evaluated on a
validation set to measure its accuracy and generalization performance. This mode can be used to tune the hyperparameters
of the model to improve its performance.

!!! example "Val"

    === "Val after training"
        ```python
          from ultralytics import YOLO

          model = YOLO('yolov8n.yaml')
          model.train(data='coco128.yaml', epochs=5)
          model.val()  # It'll automatically evaluate the data you trained.
        ```

    === "Val independently"
        ```python
          from ultralytics import YOLO

          model = YOLO("model.pt")
          # It'll use the data yaml file in model.pt if you don't set data.
          model.val()
          # or you can set the data you want to val
          model.val(data='coco128.yaml')
        ```

[Val Examples](../modes/val.md){ .md-button .md-button--primary}

## [Predict](../modes/predict.md)

Predict mode is used for making predictions using a trained YOLOv8 model on new images or videos. In this mode, the
model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model
predicts the classes and locations of objects in the input images or videos.

!!! example "Predict"

    === "From source"
        ```python
        from ultralytics import YOLO
        from PIL import Image
        import cv2

        model = YOLO("model.pt")
        # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
        results = model.predict(source="0")
        results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

        # from PIL
        im1 = Image.open("bus.jpg")
        results = model.predict(source=im1, save=True)  # save plotted images

        # from ndarray
        im2 = cv2.imread("bus.jpg")
        results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

        # from list of PIL/ndarray
        results = model.predict(source=[im1, im2])
        ```

    === "Results usage"
        ```python
        # results would be a list of Results object including all the predictions by default
        # but be careful as it could occupy a lot memory when there're many images, 
        # especially the task is segmentation.
        # 1. return as a list
        results = model.predict(source="folder")

        # results would be a generator which is more friendly to memory by setting stream=True
        # 2. return as a generator
        results = model.predict(source=0, stream=True)

        for result in results:
            # Detection
            result.boxes.xyxy   # box with xyxy format, (N, 4)
            result.boxes.xywh   # box with xywh format, (N, 4)
            result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
            result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
            result.boxes.conf   # confidence score, (N, 1)
            result.boxes.cls    # cls, (N, 1)

            # Segmentation
            result.masks.data      # masks, (N, H, W)
            result.masks.xy        # x,y segments (pixels), List[segment] * N
            result.masks.xyn       # x,y segments (normalized), List[segment] * N

            # Classification
            result.probs     # cls prob, (num_class, )

        # Each result is composed of torch.Tensor by default, 
        # in which you can easily use following functionality:
        result = result.cuda()
        result = result.cpu()
        result = result.to("cpu")
        result = result.numpy()
        ```

[Predict Examples](../modes/predict.md){ .md-button .md-button--primary}

## [Export](../modes/export.md)

Export mode is used for exporting a YOLOv8 model to a format that can be used for deployment. In this mode, the model is
converted to a format that can be used by other software applications or hardware devices. This mode is useful when
deploying the model to production environments.

!!! example "Export"

    === "Export to ONNX"

        Export an official YOLOv8n model to ONNX with dynamic batch-size and image-size.
        ```python
          from ultralytics import YOLO

          model = YOLO('yolov8n.pt')
          model.export(format='onnx', dynamic=True)
        ```

    === "Export to TensorRT"

        Export an official YOLOv8n model to TensorRT on `device=0` for acceleration on CUDA devices.
        ```python
          from ultralytics import YOLO

          model = YOLO('yolov8n.pt')
          model.export(format='onnx', device=0)
        ```

[Export Examples](../modes/export.md){ .md-button .md-button--primary}

## [Track](../modes/track.md)

Track mode is used for tracking objects in real-time using a YOLOv8 model. In this mode, the model is loaded from a
checkpoint file, and the user can provide a live video stream to perform real-time object tracking. This mode is useful
for applications such as surveillance systems or self-driving cars.

!!! example "Track"

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO('yolov8n.pt')  # load an official detection model
        model = YOLO('yolov8n-seg.pt')  # load an official segmentation model
        model = YOLO('path/to/best.pt')  # load a custom model
        
        # Track with the model
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True) 
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True, tracker="bytetrack.yaml") 
        ```

[Track Examples](../modes/track.md){ .md-button .md-button--primary}

## [Benchmark](../modes/benchmark.md)

Benchmark mode is used to profile the speed and accuracy of various export formats for YOLOv8. The benchmarks provide
information on the size of the exported format, its `mAP50-95` metrics (for object detection and segmentation)
or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various export
formats like ONNX, OpenVINO, TensorRT and others. This information can help users choose the optimal export format for
their specific use case based on their requirements for speed and accuracy.

!!! example "Benchmark"

    === "Python"
    
        Benchmark an official YOLOv8n model across all export formats.
        ```python
        from ultralytics.yolo.utils.benchmarks import benchmark
        
        # Benchmark
        benchmark(model='yolov8n.pt', imgsz=640, half=False, device=0)
        ```

[Benchmark Examples](../modes/benchmark.md){ .md-button .md-button--primary}

## Using Trainers

`YOLO` model class is a high-level wrapper on the Trainer classes. Each YOLO task has its own trainer that inherits
from `BaseTrainer`.

!!! tip "Detection Trainer Example"

        ```python
        from ultralytics.yolo import v8 import DetectionTrainer, DetectionValidator, DetectionPredictor

        # trainer
        trainer = DetectionTrainer(overrides={})
        trainer.train()
        trained_model = trainer.best

        # Validator
        val = DetectionValidator(args=...)
        val(model=trained_model)

        # predictor
        pred = DetectionPredictor(overrides={})
        pred(source=SOURCE, model=trained_model)

        # resume from last weight
        overrides["resume"] = trainer.last
        trainer = detect.DetectionTrainer(overrides=overrides)
        ```

You can easily customize Trainers to support custom tasks or explore R&D ideas.
Learn more about Customizing `Trainers`, `Validators` and `Predictors` to suit your project needs in the Customization
Section.

[Customization tutorials](engine.md){ .md-button .md-button--primary}