---
comments: true
description: Learn to customize the Ultralytics YOLO Trainer for specific tasks. Step-by-step instructions with Python examples for maximum model performance.
keywords: Ultralytics, YOLO, Trainer Customization, Python, Machine Learning, AI, Model Training, DetectionTrainer, Custom Models
---

# Advanced Customization

Both the Ultralytics YOLO command-line and Python interfaces are high-level abstractions built upon base engine executors. This guide focuses on the `Trainer` engine, explaining how to customize it for your specific needs.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=104"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLO: Advanced Customization
</p>

## BaseTrainer

The `BaseTrainer` class provides a generic training routine adaptable for various tasks. Customize it by overriding specific functions or operations while adhering to the required formats. For example, integrate your own custom model and dataloader by overriding these functions:

- `get_model(cfg, weights)`: Builds the model to be trained.
- `get_dataloader()`: Builds the dataloader.

For more details and source code, see the [`BaseTrainer` Reference](../reference/engine/trainer.md).

## DetectionTrainer

Here's how to use and customize the Ultralytics YOLO `DetectionTrainer`:

```python
from ultralytics.models.yolo.detect import DetectionTrainer

trainer = DetectionTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # Get the best model
```

### Customizing the DetectionTrainer

To train a custom detection model not directly supported, overload the existing `get_model` functionality:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model given configuration and weight files."""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
```

Further customize the trainer by modifying the [loss function](https://www.ultralytics.com/glossary/loss-function) or adding a [callback](callbacks.md) to upload the model to Google Drive every 10 [epochs](https://www.ultralytics.com/glossary/epoch). Here's an example:

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


class MyCustomModel(DetectionModel):
    def init_criterion(self):
        """Initializes the loss function and adds a callback for uploading the model to Google Drive every 10 epochs."""
        ...


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Returns a customized detection model instance configured with specified config and weights."""
        return MyCustomModel(...)


# Callback to upload model weights
def log_model(trainer):
    """Logs the path of the last model weight used by the trainer."""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = CustomTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callbacks
trainer.train()
```

For more information on callback triggering events and entry points, see the [Callbacks Guide](../usage/callbacks.md).

## Other Engine Components

Customize other components like `Validators` and `Predictors` similarly. For more information, refer to the documentation for [Validators](../reference/engine/validator.md) and [Predictors](../reference/engine/predictor.md).

## Using YOLO with Custom Trainers

The `YOLO` model class provides a high-level wrapper for the Trainer classes. You can leverage this architecture for greater flexibility in your machine learning workflows:

```python
from ultralytics import YOLO
from ultralytics.models.yolo.detect import DetectionTrainer


# Create a custom trainer
class MyCustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Custom code implementation."""
        ...


# Initialize YOLO model
model = YOLO("yolo11n.pt")

# Train with custom trainer
results = model.train(trainer=MyCustomTrainer, data="coco8.yaml", epochs=3)
```

This approach allows you to maintain the simplicity of the YOLO interface while customizing the underlying training process to suit your specific requirements.

## FAQ

### How do I customize the Ultralytics YOLO DetectionTrainer for specific tasks?

Customize the `DetectionTrainer` for specific tasks by overriding its methods to adapt to your custom model and dataloader. Start by inheriting from `DetectionTrainer` and redefine methods like `get_model` to implement custom functionalities. Here's an example:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model given configuration and weight files."""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # Get the best model
```

For further customization, such as changing the [loss function](https://www.ultralytics.com/glossary/loss-function) or adding a [callback](https://www.ultralytics.com/glossary/callback), refer to the [Callbacks Guide](../usage/callbacks.md).

### What are the key components of the BaseTrainer in Ultralytics YOLO?

The `BaseTrainer` serves as the foundation for training routines, customizable for various tasks by overriding its generic methods. Key components include:

- `get_model(cfg, weights)`: Builds the model to be trained.
- `get_dataloader()`: Builds the dataloader.
- `preprocess_batch()`: Handles batch preprocessing before model forward pass.
- `set_model_attributes()`: Sets model attributes based on dataset information.
- `get_validator()`: Returns a validator for model evaluation.

For more details on customization and source code, see the [`BaseTrainer` Reference](../reference/engine/trainer.md).

### How can I add a callback to the Ultralytics YOLO DetectionTrainer?

Add callbacks to monitor and modify the training process in `DetectionTrainer`. Here's how to add a callback to log model weights after every training [epoch](https://www.ultralytics.com/glossary/epoch):

```python
from ultralytics.models.yolo.detect import DetectionTrainer


# Callback to upload model weights
def log_model(trainer):
    """Logs the path of the last model weight used by the trainer."""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = DetectionTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callbacks
trainer.train()
```

For more details on callback events and entry points, refer to the [Callbacks Guide](../usage/callbacks.md).

### Why should I use Ultralytics YOLO for model training?

Ultralytics YOLO provides a high-level abstraction over powerful engine executors, making it ideal for rapid development and customization. Key benefits include:

- **Ease of Use**: Both command-line and Python interfaces simplify complex tasks.
- **Performance**: Optimized for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and various vision AI applications.
- **Customization**: Easily extendable for custom models, [loss functions](https://www.ultralytics.com/glossary/loss-function), and dataloaders.
- **Modularity**: Components can be modified independently without affecting the entire pipeline.
- **Integration**: Seamlessly works with popular frameworks and tools in the ML ecosystem.

Learn more about YOLO's capabilities by exploring the main [Ultralytics YOLO](https://www.ultralytics.com/yolo) page.

### Can I use the Ultralytics YOLO DetectionTrainer for non-standard models?

Yes, the `DetectionTrainer` is highly flexible and customizable for non-standard models. Inherit from `DetectionTrainer` and overload methods to support your specific model's needs. Here's a simple example:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model."""
        ...


trainer = CustomDetectionTrainer(overrides={...})
trainer.train()
```

For comprehensive instructions and examples, review the [`DetectionTrainer` Reference](../reference/models/yolo/detect/train.md).
