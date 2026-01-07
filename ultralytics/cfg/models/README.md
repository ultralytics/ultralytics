<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics Model Configurations

Welcome to the [Ultralytics](https://www.ultralytics.com/) models configuration directory. This folder contains a collection of model configuration files (`*.yaml`) that define Ultralytics YOLO model architectures. These configurations are used across common [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks such as [object detection](https://docs.ultralytics.com/tasks/detect/), [image segmentation](https://docs.ultralytics.com/tasks/segment/), pose estimation, oriented bounding boxes (OBB), and image classification.

Configurations are designed to run efficiently on a range of hardware, from standard [CPUs](https://en.wikipedia.org/wiki/Central_processing_unit) to modern [GPUs](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit). Pick a base model that matches your constraints (latency, memory, and accuracy), then customize it as needed.

To get started, choose a `*.yaml` file (see the [YAML format](https://www.ultralytics.com/glossary/yaml)) and use it to [train](https://docs.ultralytics.com/modes/train/) or export your model. For more details, see the Ultralytics [Documentation](https://docs.ultralytics.com/) or open a question on [GitHub Issues](https://github.com/ultralytics/ultralytics/issues).

## 🚀 Usage

Model configuration files (`*.yaml`) can be used directly from the [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli/) via the `yolo` command:

```bash
# Train a YOLO11n detection model using the coco8 dataset for 100 epochs
yolo task=detect mode=train model=yolo11n.yaml data=coco8.yaml epochs=100 imgsz=640
```

The same YAML files can be used from [Python](https://www.python.org/), with the same [configuration arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI:

```python
from ultralytics import YOLO

# Initialize a YOLO11n model from a YAML configuration file
# This creates a model architecture without loading pre-trained weights
model = YOLO("yolo11n.yaml")

# Alternatively, load a pre-trained YOLO11n model directly
# This loads both the architecture and the weights trained on COCO
# model = YOLO("yolo11n.pt")

# Display model information (architecture, layers, parameters, etc.)
model.info()

# Train the model using the COCO8 dataset (a small subset of COCO) for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the trained model on an image
results = model("path/to/image.jpg")
```

## 🏗️ Pre-trained Model Architectures

Ultralytics supports a variety of model architectures. Visit the [Ultralytics Models](https://docs.ultralytics.com/models/) documentation page for details and usage examples, including:

- [YOLO12](https://docs.ultralytics.com/models/yolo12/)
- [YOLO11](https://docs.ultralytics.com/models/yolo11/)
- [YOLOv10](https://docs.ultralytics.com/models/yolov10/)
- [YOLOv9](https://docs.ultralytics.com/models/yolov9/)
- [YOLOv8](https://docs.ultralytics.com/models/yolov8/)
- [YOLOv5](https://docs.ultralytics.com/models/yolov5/)
- [And more...](https://docs.ultralytics.com/models/)

You can easily use any of these models by loading their configuration files (`.yaml`) or their [pre-trained](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) checkpoints (`.pt`).

## 🤝 Contribute New Models

Have you developed a novel YOLO variant, experimented with a unique architecture, or achieved state-of-the-art results through specific tuning? We encourage you to share your innovations with the community by contributing to our Models section! Contributions like new model configurations, architectural improvements, or performance optimizations are highly valuable and help enrich the Ultralytics ecosystem.

Sharing your work here allows others to benefit from your insights and expands the range of available model choices. It's an excellent way to showcase your expertise and make the Ultralytics YOLO platform even more versatile and powerful.

To contribute, review the [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for instructions on submitting a [Pull Request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests).

Thank you for helping improve the Ultralytics model zoo.
