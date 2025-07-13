<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics Model Configurations

Welcome to the [Ultralytics](https://www.ultralytics.com/) Models configuration directory! This directory contains a comprehensive collection of pre-configured model configuration files (`*.yaml`). These files serve as blueprints for creating custom [Ultralytics YOLO](https://docs.ultralytics.com/models/yolo11/) models, meticulously crafted and fine-tuned by the Ultralytics team. Our goal is to provide optimal performance across a diverse range of [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) tasks, including [object detection](https://docs.ultralytics.com/tasks/detect/), [image segmentation](https://docs.ultralytics.com/tasks/segment/), pose estimation, and [object tracking](https://docs.ultralytics.com/modes/track/).

These configurations cater to various scenarios and are engineered for efficiency, running smoothly on different hardware platforms, from standard [CPUs](https://en.wikipedia.org/wiki/Central_processing_unit) to powerful [GPUs](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit). Whether you're an experienced [machine learning](https://en.wikipedia.org/wiki/Machine_learning) practitioner or new to the YOLO ecosystem, this directory offers an excellent starting point for your custom model development journey.

To begin, explore the models within this directory and select one that aligns with your project requirements. You can then use the corresponding `*.yaml` file (learn more about the [YAML format](https://www.ultralytics.com/glossary/yaml)) to [train](https://docs.ultralytics.com/modes/train/) and deploy your custom YOLO model effortlessly. For detailed guidance, refer to the Ultralytics [Documentation](https://docs.ultralytics.com/), and don't hesitate to reach out to the community via [GitHub Issues](https://github.com/ultralytics/ultralytics/issues) if you need support. Start building your custom YOLO model today!

## 🚀 Usage

Model `*.yaml` configuration files can be directly utilized in the [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli/) using the `yolo` command:

```bash
# Train a YOLO11n detection model using the coco8 dataset for 100 epochs
yolo task=detect mode=train model=yolo11n.yaml data=coco8.yaml epochs=100 imgsz=640
```

These files are [Python](https://www.python.org/)-compatible, accepting the same [configuration arguments](https://docs.ultralytics.com/usage/cfg/) as shown in the CLI example:

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

Ultralytics supports a variety of cutting-edge model architectures. Visit the [Ultralytics Models](https://docs.ultralytics.com/models/) documentation page for in-depth information and usage examples for each model, including:

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

To contribute, please review our [Contributing Guide](https://docs.ultralytics.com/help/contributing/) for detailed instructions on submitting a [Pull Request (PR)](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests) 🛠️. We eagerly await your contributions!

Let's collaborate to enhance the capabilities and diversity of the Ultralytics YOLO models 🙏!
