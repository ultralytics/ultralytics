---
comments: true
description: Learn to integrate YOLO11 in Python for object detection, segmentation, and classification. Load, train models, and make predictions easily with our comprehensive guide.
keywords: YOLO11, Python, object detection, segmentation, classification, machine learning, AI, pretrained models, train models, make predictions
---

# Python Usage

Welcome to the YOLO11 Python Usage documentation! This guide is designed to help you seamlessly integrate YOLO11 into your Python projects for [object detection](https://www.ultralytics.com/glossary/object-detection), segmentation, and classification. Here, you'll learn how to load and use pretrained models, train new models, and perform predictions on images. The easy-to-use Python interface is a valuable resource for anyone looking to incorporate YOLO11 into their Python projects, allowing you to quickly implement advanced object detection capabilities. Let's get started!

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=58"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLO11: Python
</p>

For example, users can load a model, train it, evaluate its performance on a validation set, and even export it to ONNX format with just a few lines of code.

!!! example "Python"

    ```python
    from ultralytics import YOLO

    # Create a new YOLO model from scratch
    model = YOLO("yolo11n.yaml")

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO("yolo11n.pt")

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

Train mode is used for training a YOLO11 model on a custom dataset. In this mode, the model is trained using the specified dataset and hyperparameters. The training process involves optimizing the model's parameters so that it can accurately predict the classes and locations of objects in an image.

!!! example "Train"

    === "From pretrained(recommended)"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")  # pass any model type
        results = model.train(epochs=5)
        ```

    === "From scratch"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.yaml")
        results = model.train(data="coco8.yaml", epochs=5)
        ```

    === "Resume"

        ```python
        model = YOLO("last.pt")
        results = model.train(resume=True)
        ```

[Train Examples](../modes/train.md){ .md-button }

## [Val](../modes/val.md)

Val mode is used for validating a YOLO11 model after it has been trained. In this mode, the model is evaluated on a validation set to measure its [accuracy](https://www.ultralytics.com/glossary/accuracy) and generalization performance. This mode can be used to tune the hyperparameters of the model to improve its performance.

!!! example "Val"

    === "Val after training"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11 model
        model = YOLO("yolo11n.yaml")

        # Train the model
        model.train(data="coco8.yaml", epochs=5)

        # Validate on training data
        model.val()
        ```

    === "Val on another dataset"

        ```python
        from ultralytics import YOLO

        # Load a YOLO11 model
        model = YOLO("yolo11n.yaml")

        # Train the model
        model.train(data="coco8.yaml", epochs=5)

        # Validate on separate data
        model.val(data="path/to/separate/data.yaml")
        ```

[Val Examples](../modes/val.md){ .md-button }

## [Predict](../modes/predict.md)

Predict mode is used for making predictions using a trained YOLO11 model on new images or videos. In this mode, the model is loaded from a checkpoint file, and the user can provide images or videos to perform inference. The model predicts the classes and locations of objects in the input images or videos.

!!! example "Predict"

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

Export mode is used for exporting a YOLO11 model to a format that can be used for deployment. In this mode, the model is converted to a format that can be used by other software applications or hardware devices. This mode is useful when deploying the model to production environments.

!!! example "Export"

    === "Export to ONNX"

        Export an official YOLO11n model to ONNX with dynamic batch-size and image-size.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="onnx", dynamic=True)
        ```

    === "Export to TensorRT"

        Export an official YOLO11n model to TensorRT on `device=0` for acceleration on CUDA devices.
        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11n.pt")
        model.export(format="onnx", device=0)
        ```

[Export Examples](../modes/export.md){ .md-button }

## [Track](../modes/track.md)

Track mode is used for tracking objects in real-time using a YOLO11 model. In this mode, the model is loaded from a checkpoint file, and the user can provide a live video stream to perform real-time object tracking. This mode is useful for applications such as surveillance systems or self-driving cars.

!!! example "Track"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load an official detection model
        model = YOLO("yolo11n-seg.pt")  # load an official segmentation model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Track with the model
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")
        ```

[Track Examples](../modes/track.md){ .md-button }

## [Benchmark](../modes/benchmark.md)

Benchmark mode is used to profile the speed and accuracy of various export formats for YOLO11. The benchmarks provide information on the size of the exported format, its `mAP50-95` metrics (for object detection and segmentation) or `accuracy_top5` metrics (for classification), and the inference time in milliseconds per image across various export formats like ONNX, OpenVINO, TensorRT and others. This information can help users choose the optimal export format for their specific use case based on their requirements for speed and accuracy.

!!! example "Benchmark"

    === "Python"

        Benchmark an official YOLO11n model across all export formats.
        ```python
        from ultralytics.utils.benchmarks import benchmark

        # Benchmark
        benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)
        ```

[Benchmark Examples](../modes/benchmark.md){ .md-button }

## Explorer

Explorer API can be used to explore datasets with advanced semantic, vector-similarity and SQL search among other features. It also enabled searching for images based on their content using natural language by utilizing the power of LLMs. The Explorer API allows you to write your own dataset exploration notebooks or scripts to get insights into your datasets.

!!! example "Semantic Search Using Explorer"

    === "Using Images"

        ```python
        from ultralytics import Explorer

        # create an Explorer object
        exp = Explorer(data="coco8.yaml", model="yolo11n.pt")
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
        exp = Explorer(data="coco8.yaml", model="yolo11n.pt")
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

!!! tip "Detection Trainer Example"

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

### How can I integrate YOLO11 into my Python project for object detection?

Integrating Ultralytics YOLO11 into your Python projects is simple. You can load a pre-trained model or train a new model from scratch. Here's how to get started:

```python
from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("yolo11n.pt")

# Perform object detection on an image
results = model("https://ultralytics.com/images/bus.jpg")

# Visualize the results
for result in results:
    result.show()
```

See more detailed examples in our [Predict Mode](../modes/predict.md) section.

### What are the different modes available in YOLO11?

Ultralytics YOLO11 provides various modes to cater to different [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) workflows. These include:

- **[Train](../modes/train.md)**: Train a model using custom datasets.
- **[Val](../modes/val.md)**: Validate model performance on a validation set.
- **[Predict](../modes/predict.md)**: Make predictions on new images or video streams.
- **[Export](../modes/export.md)**: Export models to various formats like ONNX, TensorRT.
- **[Track](../modes/track.md)**: Real-time object tracking in video streams.
- **[Benchmark](../modes/benchmark.md)**: Benchmark model performance across different configurations.

Each mode is designed to provide comprehensive functionalities for different stages of model development and deployment.

### How do I train a custom YOLO11 model using my dataset?

To train a custom YOLO11 model, you need to specify your dataset and other hyperparameters. Here's a quick example:

```python
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.yaml")

# Train the model with custom dataset
model.train(data="path/to/your/dataset.yaml", epochs=10)
```

For more details on training and hyperlinks to example usage, visit our [Train Mode](../modes/train.md) page.

### How do I export YOLO11 models for deployment?

Exporting YOLO11 models in a format suitable for deployment is straightforward with the `export` function. For example, you can export a model to ONNX format:

```python
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
model.export(format="onnx")
```

For various export options, refer to the [Export Mode](../modes/export.md) documentation.

### Can I validate my YOLO11 model on different datasets?

Yes, validating YOLO11 models on different datasets is possible. After training, you can use the validation mode to evaluate the performance:

```python
from ultralytics import YOLO

# Load a YOLO11 model
model = YOLO("yolo11n.yaml")

# Train the model
model.train(data="coco8.yaml", epochs=5)

# Validate the model on a different dataset
model.val(data="path/to/separate/data.yaml")
```

Check the [Val Mode](../modes/val.md) page for detailed examples and usage.
