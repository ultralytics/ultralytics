---
comments: true
description: Explore common questions and solutions related to Ultralytics YOLO, from hardware requirements to model fine-tuning and real-time detection.
keywords: Ultralytics, YOLO, FAQ, object detection, hardware requirements, fine-tuning, ONNX, TensorFlow, real-time detection, model accuracy
---

# Ultralytics YOLO Frequently Asked Questions (FAQ)

This FAQ section addresses common questions and issues users might encounter while working with [Ultralytics](https://www.ultralytics.com/) YOLO repositories.

## FAQ

### What is Ultralytics and what does it offer?

Ultralytics is a [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) AI company specializing in state-of-the-art object detection and [image segmentation](https://www.ultralytics.com/glossary/image-segmentation) models, with a focus on the YOLO (You Only Look Once) family. Their offerings include:

- Open-source implementations of [YOLO26](https://docs.ultralytics.com/models/yolo26/) (latest) and [YOLO11](https://docs.ultralytics.com/models/yolo11/) (previous generation)
- A wide range of [pretrained models](https://docs.ultralytics.com/models/) for various computer vision tasks
- A comprehensive [Python package](https://docs.ultralytics.com/usage/python/) for seamless integration of YOLO models into projects
- Versatile [tools](https://docs.ultralytics.com/modes/) for training, testing, and deploying models
- [Extensive documentation](https://docs.ultralytics.com/) and a supportive community

### How do I install the Ultralytics package?

Installing the Ultralytics package is straightforward using pip:

```bash
pip install ultralytics
```

For the latest development version, install directly from the GitHub repository:

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

Detailed installation instructions can be found in the [quickstart guide](https://docs.ultralytics.com/quickstart/).

### What are the system requirements for running Ultralytics models?

Minimum requirements:

- Python 3.8+
- [PyTorch](https://www.ultralytics.com/glossary/pytorch) 1.8+
- CUDA-compatible GPU (for GPU acceleration)

Recommended setup:

- Python 3.8+
- PyTorch 1.10+
- NVIDIA GPU with CUDA 11.2+
- 8GB+ RAM
- 50GB+ free disk space (for dataset storage and model training)

For troubleshooting common issues, visit the [YOLO Common Issues](https://docs.ultralytics.com/guides/yolo-common-issues/) page.

### How can I train a custom YOLO model on my own dataset?

To train a custom YOLO model:

1. Prepare your dataset in YOLO format (images and corresponding label txt files).
2. Create a YAML file describing your dataset structure and classes.
3. Use the following Python code to start training:

    ```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo26n.yaml")  # build a new model from scratch
    model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(data="path/to/your/data.yaml", epochs=100, imgsz=640)
    ```

For a more in-depth guide, including data preparation and advanced training options, refer to the comprehensive [training guide](https://docs.ultralytics.com/modes/train/).

### What pretrained models are available in Ultralytics?

Ultralytics offers a diverse range of pretrained models for various tasks:

- Object Detection: YOLO26n, YOLO26s, YOLO26m, YOLO26l, YOLO26x
- [Instance Segmentation](https://www.ultralytics.com/glossary/instance-segmentation): YOLO26n-seg, YOLO26s-seg, YOLO26m-seg, YOLO26l-seg, YOLO26x-seg
- Classification: YOLO26n-cls, YOLO26s-cls, YOLO26m-cls, YOLO26l-cls, YOLO26x-cls
- Pose Estimation: YOLO26n-pose, YOLO26s-pose, YOLO26m-pose, YOLO26l-pose, YOLO26x-pose
- Oriented Detection (OBB): YOLO26n-obb, YOLO26s-obb, YOLO26m-obb, YOLO26l-obb, YOLO26x-obb

These models vary in size and complexity, offering different trade-offs between speed and [accuracy](https://www.ultralytics.com/glossary/accuracy). Explore the full range of [pretrained models](https://docs.ultralytics.com/models/) to find the best fit for your project.

### How do I perform inference using a trained Ultralytics model?

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

For advanced inference options, including batch processing and video inference, check out the detailed [prediction guide](https://docs.ultralytics.com/modes/predict/).

### Can Ultralytics models be deployed on edge devices or in production environments?

Absolutely! Ultralytics models are designed for versatile deployment across various platforms:

- Edge devices: Optimize inference on devices like NVIDIA Jetson or Intel Neural Compute Stick using TensorRT, ONNX, or OpenVINO.
- Mobile: Deploy on Android or iOS devices by converting models to TFLite or Core ML.
- Cloud: Leverage frameworks like [TensorFlow](https://www.ultralytics.com/glossary/tensorflow) Serving or PyTorch Serve for scalable cloud deployments.
- Web: Implement in-browser inference using ONNX.js or TensorFlow.js.

Ultralytics provides export functions to convert models to various formats for deployment. Explore the wide range of [deployment options](https://docs.ultralytics.com/guides/model-deployment-options/) to find the best solution for your use case.

### What's the difference between YOLO11 and YOLO26?

Key distinctions include:

- End-to-End NMS-Free Inference: YOLO26 is natively end-to-end, producing predictions directly without non-maximum suppression (NMS), reducing latency and simplifying deployment.
- DFL Removal: YOLO26 removes the Distribution Focal Loss module, simplifying export and improving compatibility with edge and low-power devices.
- MuSGD Optimizer: A hybrid of SGD and Muon (inspired by Moonshot AI's Kimi K2) for more stable training and faster convergence.
- CPU Performance: YOLO26 delivers up to 43% faster CPU inference, making it ideal for devices without GPUs.
- Task-Specific Optimizations: Enhanced segmentation with semantic loss and multi-scale protos, RLE for precision pose estimation, and improved OBB decoding with angle loss.
- Tasks: Both models support [object detection](https://www.ultralytics.com/glossary/object-detection), instance segmentation, classification, pose estimation, and oriented object detection (OBB) in a unified framework.

For an in-depth comparison of features and performance metrics, visit the [YOLO26 documentation page](https://docs.ultralytics.com/models/yolo26/).

### How can I contribute to the Ultralytics open-source project?

Contributing to Ultralytics is a great way to improve the project and expand your skills. Here's how you can get involved:

1. Fork the Ultralytics repository on GitHub.
2. Create a new branch for your feature or bug fix.
3. Make your changes and ensure all tests pass.
4. Submit a pull request with a clear description of your changes.
5. Participate in the code review process.

You can also contribute by reporting bugs, suggesting features, or improving documentation. For detailed guidelines and best practices, refer to the [contributing guide](https://docs.ultralytics.com/help/contributing/).

### How do I install the Ultralytics package in Python?

Installing the Ultralytics package in Python is simple. Use pip by running the following command in your terminal or command prompt:

```bash
pip install ultralytics
```

For the cutting-edge development version, install directly from the GitHub repository:

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

For environment-specific installation instructions and troubleshooting tips, consult the comprehensive [quickstart guide](https://docs.ultralytics.com/quickstart/).

### What are the main features of Ultralytics YOLO?

Ultralytics YOLO boasts a rich set of features for advanced computer vision tasks:

- Real-Time Detection: Efficiently detect and classify objects in real-time scenarios.
- Multi-Task Capabilities: Perform object detection, instance segmentation, classification, and pose estimation with a unified framework.
- Pretrained Models: Access a variety of [pretrained models](https://docs.ultralytics.com/models/) that balance speed and accuracy for different use cases.
- Custom Training: Easily fine-tune models on custom datasets with the flexible [training pipeline](https://docs.ultralytics.com/modes/train/).
- Wide [Deployment Options](https://docs.ultralytics.com/guides/model-deployment-options/): Export models to various formats like TensorRT, ONNX, and CoreML for deployment across different platforms.
- Extensive Documentation: Benefit from comprehensive [documentation](https://docs.ultralytics.com/) and a supportive community for your computer vision workflows.

### How can I improve the performance of my YOLO model?

Enhancing your YOLO model's performance can be achieved through several techniques:

1. [Hyperparameter Tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning): Experiment with different hyperparameters using the [Hyperparameter Tuning Guide](https://docs.ultralytics.com/guides/hyperparameter-tuning/) to optimize model performance.
2. [Data Augmentation](https://www.ultralytics.com/glossary/data-augmentation): Implement techniques like flip, scale, rotate, and color adjustments to enhance your training dataset and improve model generalization.
3. [Transfer Learning](https://www.ultralytics.com/glossary/transfer-learning): Leverage pretrained models and fine-tune them on your specific dataset using the [Train guide](../modes/train.md).
4. Export to Efficient Formats: Convert your model to optimized formats like TensorRT or ONNX for faster inference using the [Export guide](../modes/export.md).
5. Benchmarking: Utilize the [Benchmark Mode](https://docs.ultralytics.com/modes/benchmark/) to measure and improve inference speed and accuracy systematically.

### Can I deploy Ultralytics YOLO models on mobile and edge devices?

Yes, Ultralytics YOLO models are designed for versatile deployment, including mobile and edge devices:

- Mobile: Convert models to TFLite or CoreML for seamless integration into Android or iOS apps. Refer to the [TFLite Integration Guide](https://docs.ultralytics.com/integrations/tflite/) and [CoreML Integration Guide](https://docs.ultralytics.com/integrations/coreml/) for platform-specific instructions.
- Edge Devices: Optimize inference on devices like NVIDIA Jetson or other edge hardware using TensorRT or ONNX. The [Edge TPU Integration Guide](https://docs.ultralytics.com/integrations/edge-tpu/) provides detailed steps for edge deployment.

For a comprehensive overview of deployment strategies across various platforms, consult the [deployment options guide](https://docs.ultralytics.com/guides/model-deployment-options/).

### How can I perform inference using a trained Ultralytics YOLO model?

Performing inference with a trained Ultralytics YOLO model is straightforward:

1. Load the Model:

    ```python
    from ultralytics import YOLO

    model = YOLO("path/to/your/model.pt")
    ```

2. Run Inference:

    ```python
    results = model("path/to/image.jpg")

    for r in results:
        print(r.boxes)  # print bounding box predictions
        print(r.masks)  # print mask predictions
        print(r.probs)  # print class probabilities
    ```

For advanced inference techniques, including batch processing, video inference, and custom preprocessing, refer to the detailed [prediction guide](https://docs.ultralytics.com/modes/predict/).

### Where can I find examples and tutorials for using Ultralytics?

Ultralytics provides a wealth of resources to help you get started and master their tools:

- 📚 [Official documentation](https://docs.ultralytics.com/): Comprehensive guides, API references, and best practices.
- 💻 [GitHub repository](https://github.com/ultralytics/ultralytics): Source code, example scripts, and community contributions.
- ✍️ [Ultralytics blog](https://www.ultralytics.com/blog): In-depth articles, use cases, and technical insights.
- 💬 [Community forums](https://community.ultralytics.com/): Connect with other users, ask questions, and share your experiences.
- 🎥 [YouTube channel](https://www.youtube.com/ultralytics?sub_confirmation=1): Video tutorials, demos, and webinars on various Ultralytics topics.

These resources provide code examples, real-world use cases, and step-by-step guides for various tasks using Ultralytics models.

If you need further assistance, consult the Ultralytics documentation or reach out to the community through [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) or the official [discussion forum](https://github.com/orgs/ultralytics/discussions).
