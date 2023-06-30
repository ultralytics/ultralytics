## Models

Welcome to the Ultralytics Models directory! Here you will find a wide variety of pre-configured model configuration
files (`*.yaml`s) that can be used to create custom YOLO models. The models in this directory have been expertly crafted
and fine-tuned by the Ultralytics team to provide the best performance for a wide range of object detection and image
segmentation tasks.

These model configurations cover a wide range of scenarios, from simple object detection to more complex tasks like
instance segmentation and object tracking. They are also designed to run efficiently on a variety of hardware platforms,
from CPUs to GPUs. Whether you are a seasoned machine learning practitioner or just getting started with YOLO, this
directory provides a great starting point for your custom model development needs.

To get started, simply browse through the models in this directory and find one that best suits your needs. Once you've
selected a model, you can use the provided `*.yaml` file to train and deploy your custom YOLO model with ease. See full
details at the Ultralytics [Docs](https://docs.ultralytics.com/models), and if you need help or have any questions, feel free
to reach out to the Ultralytics team for support. So, don't wait, start creating your custom YOLO model now!

### Usage

Model `*.yaml` files may be used directly in the Command Line Interface (CLI) with a `yolo` command:

```bash
yolo task=detect mode=train model=yolov8n.yaml data=coco128.yaml epochs=100
```

They may also be used directly in a Python environment, and accepts the same
[arguments](https://docs.ultralytics.com/usage/cfg/) as in the CLI example above:

```python
from ultralytics import YOLO

model = YOLO("model.yaml")  # build a YOLOv8n model from scratch
# YOLO("model.pt")  use pre-trained model if available
model.info()  # display model information
model.train(data="coco128.yaml", epochs=100)  # train the model
```

## Pre-trained Model Architectures

Ultralytics supports many model architectures. Visit https://docs.ultralytics.com/models to view detailed information
and usage. Any of these models can be used by loading their configs or pretrained checkpoints if available.

## Contributing New Models

If you've developed a new model architecture or have improvements for existing models that you'd like to contribute to the Ultralytics community, please submit your contribution in a new Pull Request. For more details, visit our [Contributing Guide](https://docs.ultralytics.com/help/contributing).
