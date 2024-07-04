---
comments: true
description: Learn to customize the YOLOv8 Trainer for specific tasks. Step-by-step instructions with Python examples for maximum model performance.
keywords: Ultralytics, YOLOv8, Trainer Customization, Python, Machine Learning, AI, Model Training, DetectionTrainer, Custom Models
---

Both the Ultralytics YOLO command-line and Python interfaces are simply a high-level abstraction on the base engine executors. Let's take a look at the Trainer engine.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/GsXGnb-A4Kc?start=104"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Mastering Ultralytics YOLOv8: Advanced Customization
</p>

## BaseTrainer

BaseTrainer contains the generic boilerplate training routine. It can be customized for any task based over overriding the required functions or operations as long the as correct formats are followed. For example, you can support your own custom model and dataloader by just overriding these functions:

- `get_model(cfg, weights)` - The function that builds the model to be trained
- `get_dataloader()` - The function that builds the dataloader More details and source code can be found in [`BaseTrainer` Reference](../reference/engine/trainer.md)

## DetectionTrainer

Here's how you can use the YOLOv8 `DetectionTrainer` and customize it.

```python
from ultralytics.models.yolo.detect import DetectionTrainer

trainer = DetectionTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # get best model
```

### Customizing the DetectionTrainer

Let's customize the trainer **to train a custom detection model** that is not supported directly. You can do this by simply overloading the existing the `get_model` functionality:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model given configuration and weight files."""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
```

You now realize that you need to customize the trainer further to:

- Customize the `loss function`.
- Add `callback` that uploads model to your Google Drive after every 10 `epochs` Here's how you can do it:

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


# callback to upload model weights
def log_model(trainer):
    """Logs the path of the last model weight used by the trainer."""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = CustomTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.train()
```

To know more about Callback triggering events and entry point, checkout our [Callbacks Guide](callbacks.md)

## Other engine components

There are other components that can be customized similarly like `Validators` and `Predictors`. See Reference section for more information on these.



## FAQ

### How do I customize the YOLOv8 `DetectionTrainer` for specific tasks?

To customize the YOLOv8 `DetectionTrainer` for specific tasks, you can subclass it and override key functions. For instance, you can overload the `get_model` function to load a custom detection model:

```python
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Returns a customized detection model instance configured with specified config and weights."""
        return MyCustomModel(...)
```

You can further extend the customization by adding callbacks or modifying loss functions. [Learn more about `DetectionTrainer` customization](#customizing-the-detectiontrainer).

### What are the benefits of using Ultralytics YOLO for model training?

Ultralytics YOLO offers several benefits for model training, including:

- **High Performance**: Achieves state-of-the-art accuracy and speed.
- **Ease of Use**: Simple interfaces for Python and command-line usage.
- **Flexibility**: Highly customizable trainers like the `DetectionTrainer` allow users to fine-tune models for specific tasks.
- **Comprehensive Documentation**: Detailed guides and references to help with setup and customization.

These features make it an excellent choice for both beginners and advanced users. [Explore more about YOLO features](https://www.ultralytics.com/yolo).

### How can I integrate custom loss functions in the YOLOv8 Trainer?

To integrate custom loss functions in the YOLOv8 Trainer, you can subclass the `DetectionModel` and override its `init_criterion` method. Here's an example:

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

trainer = CustomTrainer(overrides={...})
trainer.train()
```

For more details, see [Customizing the DetectionTrainer](#customizing-the-detectiontrainer).

### What are the key functions to override in `BaseTrainer` for custom models?

In `BaseTrainer`, the key functions to override for custom models include:

- `get_model(cfg, weights)`: Builds the model to be trained.
- `get_dataloader()`: Builds the data loader for training.

These functions allow you to define specific configurations and data processing steps tailored to your custom models.

Example:
```python
from ultralytics.models.yolo.detect import DetectionTrainer

class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Returns a customized detection model instance configured with specified config and weights."""
        ...

    def get_dataloader(self):
        """Returns a customized dataloader."""
        ...
```

Learn more about overriding key functions in the [BaseTrainer Reference](../reference/engine/trainer.md).

### Why should I use Ultralytics YOLOv8 for custom detection models?

Ultralytics YOLOv8 is ideal for custom detection models due to its:

- **High Flexibility**: Easily customizable for various tasks.
- **User-Friendly Interface**: Simplified Python and command-line interfaces.
- **Performance**: Maintains high speed and accuracy.
- **Comprehensive Resources**: Detailed documentation and community support.

This combination of features makes YOLOv8 a powerful tool for developing custom detection models quickly and efficiently. Learn more about how to [train custom detection models](#customizing-the-detectiontrainer).