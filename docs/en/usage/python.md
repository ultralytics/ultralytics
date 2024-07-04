---
comments: true
description: Learn to integrate YOLOv8 in Python for object detection, segmentation, and classification. Load, train models, and make predictions easily with our comprehensive guide.
keywords: YOLOv8, Python, object detection, segmentation, classification, machine learning, AI, pretrained models, train models, make predictions
---

# Python Usage

Welcome to the YOLOv8 Python Usage documentation! This guide is designed to help you seamlessly integrate YOLOv8 into your Python projects for object detection, segmentation, and classification. Here, you'll learn how to load and use pretrained models, train new models, and perform predictions on images. The easy-to-use Python interface is a valuable resource for anyone looking to incorporate YOLOv8 into their Python projects, allowing you to quickly implement advanced object detection capabilities. Let's get started!

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=58"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: Python
</p>

For example, users can load a model, train it, evaluate its performance on a validation set, and even export it to ONNX format with just a few lines of code.

!!! Example "Python"

    ```python
    from ultralytics import YOLO

    # Create a new YOLO model from scratch
    model = YOLO("yolov8n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolov8n.pt")

    # Train the model using the 'coco8.yaml' dataset for 3 epochs
    results = model.train(data="coco8.yaml", epochs=3)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Perform object detection on an image using the model
    results = model("https://ultralytics.com/images/bus.jpg")

    # Export the model to ONNX format
    success = model.export(format="onnx")
    ```

## [Train](../modes/train.md)

Train mode is used for training a YOLOv8 model on a custom dataset. In this mode, the model is trained using the specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can accurately predict the classes and locations of objects in an image.

!!! Example "Train"

    === "From pretrained(recommended)"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")  # pass any model type
        results = model.train(epochs=5)
        ```

    === "From scratch"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.yaml")
        results = model.train(data="coco8.yaml", epochs=5)
        ```

    === "Resume"

        ```python
        model = YOLO("last.pt")
        results = model.train(resume=True)
        ```

[Train Examples](../modes/train.md){ .md-button }

## [Val](../modes/val.md)

Val mode is used for validating a YOLOv8 model after it has been trained. In this mode, the model is evaluated on a validation set to measure its accuracy and generalization performance. This mode can be used to tune the hyperparameters of the model to improve its performance.

!!! Example "Val"

    === "Val after training"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8 model
        model = YOLO("yolov8n.yaml")

        # Train the model
        model.train(data="coco8.yaml", epochs=5)

        # Validate on training data
        model.val()
        ```

    === "Val on another dataset"

        ```python
        from ultralytics import YOLO

        # Load a YOLOv8 model
        model = YOLO("yolov8n.yaml")

        # Train the model
        model.train(data="coco8.yaml", epochs=5)

        # Validate on separate data
        model.val(data="path/to/separate/data.yaml")
        ```

[Val Examples](../modes/val.md){ .md-button }

## [Predict](../modes/predict.md)

Predict mode is used for making predictions using a trained YOLOv8 model on new images or videos. In this mode, the model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model predicts the classes and locations of objects in the input images or videos.

!!! Example "Predict"

    === "From source"

        ```python
        import cv2
        from PIL import Image

        from ultralytics import YOLO

        model = YOLO("model.pt")
        # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
        results = model.predict(source="0")
        results = model.predict(source="folder", show=True)  # Display preds. Accepts all YOLO predict arguments

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
            result.boxes.xyxy  # box with xyxy format, (N, 4)
            result.boxes.xywh  # box with xywh format, (N, 4)
            result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
            result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
            result.boxes.conf  # confidence score, (N, 1)
            result.boxes.cls  # cls, (N, 1)

            # Segmentation
            result.masks.data  # masks, (N, H, W)
            result.masks.xy  # x,y segments (pixels), List[segment] * N
            result.masks.xyn  # x,y segments (normalized), List[segment] * N

            # Classification
            result.probs  # cls prob, (num_class, )

        # Each result is composed of torch.Tensor by default,
        # in which you can easily use following functionality:
        result = result.cuda()
        result = result.cpu()
        result = result.to("cpu")
        result = result.numpy()
        ```

[Predict Examples](../modes/predict.md){ .md-button }

## [Export](../modes/export.md)

Export mode is used for exporting a YOLOv8 model to a format that can be used for deployment. In this mode, the model is converted to a format that can be used by other software applications or hardware devices. This mode is useful when deploying the model to production environments.

!!! Example "Export"

    === "Export to ONNX"

        Export an official YOLOv8n model to ONNX with dynamic batch-size and image-size.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "Export to TensorRT"

        Export an official YOLOv8n model to TensorRT on `device=0` for acceleration on CUDA devices.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        model.export(format="onnx", device=0)
        ```

[Export Examples](../modes/export.md){ .md-button }

## [Track](../modes/track.md)

Track mode is used for tracking objects in real-time using a YOLOv8 model. In this mode, the model is loaded from a checkpoint file, and the user can provide a live video stream to perform real-time object tracking. This mode is useful for applications such as surveillance systems or self-driving cars.

!!! Example "Track"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolov8n.pt")  # load an official detection model
        model = YOLO("yolov8n-seg.pt")  # load an official segmentation model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Track with the model
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
        ```

[Track Examples](../modes/track.md){ .md-button }

## [Benchmark](../modes/benchmark.md)

Benchmark mode is used to profile the speed and accuracy of various export formats for YOLOv8. The benchmarks provide information on the size of the exported format, its `mAP50-95` metrics (for object detection and segmentation) or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various export formats like ONNX, OpenVINO, TensorRT and others. This information can help users choose the optimal export format for their specific use case based on their requirements for speed and accuracy.

!!! Example "Benchmark"

    === "Python"

        Benchmark an official YOLOv8n model across all export formats.
        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark
        benchmark(model="yolov8n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

[Benchmark Examples](../modes/benchmark.md){ .md-button }

## Explorer

Explorer API can be used to explore datasets with advanced semantic, vector-similarity and SQL search among other features. It also enabled searching for images based on their content using natural language by utilizing the power of LLMs. The Explorer API allows you to write your own dataset exploration notebooks or scripts to get insights into your datasets.

!!! Example "Semantic Search Using Explorer"

    === "Using Images"

        ```python
        from ultralytics import Explorer

        # create an Explorer object
        exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
        exp.create_embeddings_table()

        similar = exp.get_similar(img="https://ultralytics.com/images/bus.jpg", limit=10)
        print(similar.head())

        # Search using multiple indices
        similar = exp.get_similar(
            img=["https://ultralytics.com/images/bus.jpg", "https://ultralytics.com/images/bus.jpg"], limit=10
        )
        print(similar.head())
        ```

    === "Using Dataset Indices"

        ```python
        from ultralytics import Explorer

        # create an Explorer object
        exp = Explorer(data="coco8.yaml", model="yolov8n.pt")
        exp.create_embeddings_table()

        similar = exp.get_similar(idx=1, limit=10)
        print(similar.head())

        # Search using multiple indices
        similar = exp.get_similar(idx=[1, 10], limit=10)
        print(similar.head())
        ```

[Explorer](../datasets/explorer/index.md){ .md-button }

## Using Trainers

`YOLO` model class is a high-level wrapper on the Trainer classes. Each YOLO task has its own trainer that inherits from `BaseTrainer`.

!!! Tip "Detection Trainer Example"

        ```python
        from ultralytics.models.yolo import DetectionPredictor, DetectionTrainer, DetectionValidator

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

You can easily customize Trainers to support custom tasks or explore R&D ideas. Learn more about Customizing `Trainers`, `Validators` and `Predictors` to suit your project needs in the Customization Section.

[Customization tutorials](engine.md){ .md-button }

## FAQ

### How do I integrate Ultralytics YOLOv8 into a Python project for object detection?

To integrate Ultralytics YOLOv8 into your Python project for object detection, follow these steps:

1. **Install Ultralytics**: Ensure you have YOLOv8 installed using `pip install ultralytics`.
2. **Import YOLO Class and Load a Pretrained Model**: This is recommended for training.

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    ```

3. **Perform Object Detection**: Use the model to detect objects in an image.

    ```python
    results = model("https://ultralytics.com/images/bus.jpg")
    ```

For more detailed instructions, refer to our [Installation](../quickstart.md) and [Python Usage](#python-usage) sections.

### What are the benefits of using Ultralytics YOLOv8 for object detection, segmentation, and classification?

Ultralytics YOLOv8 offers several key advantages:

- **High Accuracy**: State-of-the-art performance in object detection, segmentation, and classification tasks.
- **Ease of Use**: Simple Python API that allows quick and easy integration.
- **Pretrained Models**: Access to a variety of pretrained models for faster development.
- **Versatility**: Supports various data inputs including images, directories, URLs, videos, and more.
- **Export Flexibility**: Ability to export models in multiple formats like ONNX, TensorRT, and more, making deployment straightforward and efficient.

Explore detailed use cases and examples in our [Train](../modes/train.md) and [Predict](../modes/predict.md) sections.

### How do I export an Ultralytics YOLOv8 model to ONNX format for deployment?

Exporting an Ultralytics YOLOv8 model to ONNX format is straightforward:

1. **Load the Model**:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    ```

2. **Export to ONNX**:

    ```python
    success = model.export(format="onnx")
    ```

You can specify additional parameters such as dynamic batch size using `model.export(format="onnx", dynamic=True)`. This makes the model compatible with various deployment platforms. For more information, refer to our [Export](../modes/export.md) section.

### What are the different modes available in Ultralytics YOLOv8, and how do they work?

Ultralytics YOLOv8 provides several modes to cater to different stages of the machine learning pipeline:

- **[Train Mode](../modes/train.md)**: For training the model on a custom dataset.
- **[Val Mode](../modes/val.md)**: Used to validate the performance of the trained model.
- **[Predict Mode](../modes/predict.md)**: For making predictions on new data.
- **[Export Mode](../modes/export.md)**: For exporting the model to various formats like ONNX.
- **[Track Mode](../modes/track.md)**: Used for real-time object tracking in video streams.
- **[Benchmark Mode](../modes/benchmark.md)**: To evaluate model performance in terms of speed and accuracy across different formats.

Each mode is designed to maximize the efficiency of various tasks within the model lifecycle. Check the respective mode links for detailed examples and usage instructions.

### How can I fine-tune a pretrained Ultralytics YOLOv8 model on my custom dataset?

Fine-tuning a pretrained Ultralytics YOLOv8 model involves these steps:

1. **Load the Pretrained Model**:

    ```python
    from ultralytics import YOLO

    model = YOLO("yolov8n.pt")
    ```

2. **Train the Model on Custom Dataset**:

    ```python
    results = model.train(data="path/to/custom_dataset.yaml", epochs=5)
    ```

This approach leverages the pretrained weights, allowing you to achieve better performance on your specific dataset with fewer training epochs. For more tips on model fine-tuning, visit our [Train](../modes/train.md) section.
