---
comments: true
description: Explore common questions and solutions related to Ultralytics YOLO, from hardware requirements to model fine-tuning and real-time detection.
keywords: Ultralytics, YOLO, FAQ, object detection, hardware requirements, fine-tuning, ONNX, TensorFlow, real-time detection, model accuracy
---

# Ultralytics YOLO Frequently Asked Questions (FAQ)

This FAQ section addresses some common questions and issues users might encounter while working with [Ultralytics](https://ultralytics.com) YOLO repositories.

## 1. What is Ultralytics and what does it offer?

Ultralytics is a computer vision AI company that develops and maintains state-of-the-art object detection and image segmentation models, primarily focusing on the YOLO (You Only Look Once) family of models. Ultralytics offers:

- [Open-source implementations of YOLOv5 and YOLOv8](https://docs.ultralytics.com/models/yolov5/)
- [Pre-trained models for various computer vision tasks](https://docs.ultralytics.com/models/)
- [A Python package for easy integration of YOLO models into projects](https://docs.ultralytics.com/usage/python/)
- [Tools for training, testing, and deploying models](https://docs.ultralytics.com/modes/)
- [Extensive documentation and community support](https://docs.ultralytics.com/)

## 2. How do I install the Ultralytics package?

To install the Ultralytics package, you can use pip, the Python package manager. Open a terminal or command prompt and run:

```
pip install ultralytics
```

For the latest development version, you can install directly from the GitHub repository:

```
pip install git+https://github.com/ultralytics/ultralytics.git
```

For more details, refer to the [quickstart guide](https://docs.ultralytics.com/quickstart/).

## 3. What are the system requirements for running Ultralytics models?

Minimum requirements:

- Python 3.7 or later
- PyTorch 1.7 or later
- CUDA-compatible GPU (for GPU acceleration)

Recommended:

- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU with CUDA 11.2+
- 8GB+ RAM
- 50GB+ free disk space (for dataset storage and model training)

For more information, visit [YOLO Common Issues](https://docs.ultralytics.com/guides/yolo-common-issues/).

## 4. How can I train a custom YOLOv8 model on my own dataset?

To train a custom YOLOv8 model:

1. Prepare your dataset in YOLO format (images and corresponding label txt files).
2. Create a YAML file describing your dataset structure and classes.
3. Use the following Python code to start training:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="path/to/your/data.yaml", epochs=100, imgsz=640)
```

For detailed instructions, refer to the [training guide](https://docs.ultralytics.com/modes/train/).

## 5. What pretrained models are available in Ultralytics?

Ultralytics offers a range of pretrained YOLOv8 models for various tasks:

- Object Detection: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- Instance Segmentation: YOLOv8n-seg, YOLOv8s-seg, YOLOv8m-seg, YOLOv8l-seg, YOLOv8x-seg
- Classification: YOLOv8n-cls, YOLOv8s-cls, YOLOv8m-cls, YOLOv8l-cls, YOLOv8x-cls

These models vary in size and complexity, offering different trade-offs between speed and accuracy. Learn more about [pretrained models](https://docs.ultralytics.com/models/yolov8/).

## 6. How do I perform inference using a trained Ultralytics model?

To perform inference with a trained model:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("path/to/your/model.pt")

# Perform inference
results = model("path/to/image.jpg")

# Process results
for r in results:
    print(r.boxes)  # print bbox predictions
    print(r.masks)  # print mask predictions
    print(r.probs)  # print class probabilities
```

For more details, visit the [prediction guide](https://docs.ultralytics.com/modes/predict/).

## 7. Can Ultralytics models be deployed on edge devices or in production environments?

Yes, Ultralytics models can be deployed on various platforms:

- Edge devices: Use TensorRT, ONNX, or OpenVINO for optimized inference on devices like NVIDIA Jetson or Intel Neural Compute Stick.
- Mobile: Convert models to TFLite or Core ML for deployment on Android or iOS devices.
- Cloud: Deploy models using frameworks like TensorFlow Serving or PyTorch Serve.
- Web: Use ONNX.js or TensorFlow.js for in-browser inference.

Ultralytics provides export functions to convert models to various formats for deployment. Learn more about [deployment options](https://docs.ultralytics.com/guides/model-deployment-options/).

## 8. What's the difference between YOLOv5 and YOLOv8?

Key differences include:

- Architecture: YOLOv8 has an improved backbone and head design.
- Performance: YOLOv8 generally offers better accuracy and speed.
- Tasks: YOLOv8 natively supports object detection, instance segmentation, and classification.
- Codebase: YOLOv8 is implemented in a more modular and extensible manner.
- Training: YOLOv8 includes advanced training techniques like multi-dataset training and hyperparameter evolution.

For a detailed comparison, visit [YOLOv5 vs YOLOv8](https://www.ultralytics.com/yolo).

## 9. How can I contribute to the Ultralytics open-source project?

To contribute:

1. Fork the Ultralytics repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure all tests pass.
4. Submit a pull request with a clear description of your changes.
5. Participate in the code review process.

You can also contribute by reporting bugs, suggesting features, or improving documentation. Refer to the [contributing guide](https://docs.ultralytics.com/help/contributing/).

## 10. Where can I find examples and tutorials for using Ultralytics?

You can find examples and tutorials in several places:

- 📚 [Official documentation](https://docs.ultralytics.com/)
- 💻 [GitHub repository](https://github.com/ultralytics/ultralytics)
- ✍️ [Ultralytics blog](https://www.ultralytics.com/blog)
- 💬 [Community forums](https://community.ultralytics.com/)
- 🎥 [YouTube channel](https://youtube.com/ultralytics?sub_confirmation=1)

These resources provide code examples, use cases, and step-by-step guides for various tasks using Ultralytics models.

If you have any more questions or need assistance, don't hesitate to consult the Ultralytics documentation or reach out to the community through [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) or the official [discussion forum](https://github.com/orgs/ultralytics/discussions).
