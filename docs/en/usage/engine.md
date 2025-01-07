---
comments: true
description: Learn to customize the YOLO11 Trainer for specific tasks. Step-by-step instructions with Python examples for maximum model performance.
keywords: Ultralytics, YOLO11, Trainer Customization, Python, Machine Learning, AI, Model Training, DetectionTrainer, Custom Models
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
  <strong>Watch:</strong> Mastering Ultralytics YOLO: Advanced Customization
</p>

## BaseTrainer

BaseTrainer contains the generic boilerplate training routine. It can be customized for any task based over overriding the required functions or operations as long the as correct formats are followed. For example, you can support your own custom model and dataloader by just overriding these functions:

- `get_model(cfg, weights)` - The function that builds the model to be trained
- `get_dataloader()` - The function that builds the dataloader More details and source code can be found in [`BaseTrainer` Reference](../reference/engine/trainer.md)

## DetectionTrainer

Here's how you can use the YOLO11 `DetectionTrainer` and customize it.

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

### How do I customize the Ultralytics YOLO11 DetectionTrainer for specific tasks?

To customize the Ultralytics YOLO11 `DetectionTrainer` for a specific task, you can override its methods to adapt to your custom model and dataloader. Start by inheriting from `DetectionTrainer` and then redefine methods like `get_model` to implement your custom functionalities. Here's an example:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model given configuration and weight files."""
        ...


trainer = CustomTrainer(overrides={...})
trainer.train()
trained_model = trainer.best  # get best model
```

For further customization like changing the `loss function` or adding a `callback`, you can reference our [Callbacks Guide](../usage/callbacks.md).

### What are the key components of the BaseTrainer in Ultralytics YOLO11?

The `BaseTrainer` in Ultralytics YOLO11 serves as the foundation for training routines and can be customized for various tasks by overriding its generic methods. Key components include:

- `get_model(cfg, weights)` to build the model to be trained.
- `get_dataloader()` to build the dataloader.

For more details on the customization and source code, see the [`BaseTrainer` Reference](../reference/engine/trainer.md).

### How can I add a callback to the Ultralytics YOLO11 DetectionTrainer?

You can add callbacks to monitor and modify the training process in Ultralytics YOLO11 `DetectionTrainer`. For instance, here's how you can add a callback to log model weights after every training [epoch](https://www.ultralytics.com/glossary/epoch):

```python
from ultralytics.models.yolo.detect import DetectionTrainer


# callback to upload model weights
def log_model(trainer):
    """Logs the path of the last model weight used by the trainer."""
    last_weight_path = trainer.last
    print(last_weight_path)


trainer = DetectionTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callbacks
trainer.train()
```

For further details on callback events and entry points, refer to our [Callbacks Guide](../usage/callbacks.md).

### Why should I use Ultralytics YOLO11 for model training?

Ultralytics YOLO11 offers a high-level abstraction on powerful engine executors, making it ideal for rapid development and customization. Key benefits include:

- **Ease of Use**: Both command-line and Python interfaces simplify complex tasks.
- **Performance**: Optimized for real-time [object detection](https://www.ultralytics.com/glossary/object-detection) and various vision AI applications.
- **Customization**: Easily extendable for custom models, [loss functions](https://www.ultralytics.com/glossary/loss-function), and dataloaders.

Learn more about YOLO11's capabilities by visiting [Ultralytics YOLO](https://www.ultralytics.com/yolo).

### Can I use the Ultralytics YOLO11 DetectionTrainer for non-standard models?

Yes, Ultralytics YOLO11 `DetectionTrainer` is highly flexible and can be customized for non-standard models. By inheriting from `DetectionTrainer`, you can overload different methods to support your specific model's needs. Here's a simple example:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomDetectionTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        """Loads a custom detection model."""
        ...


trainer = CustomDetectionTrainer(overrides={...})
trainer.train()
```

For more comprehensive instructions and examples, review the [DetectionTrainer](../reference/engine/trainer.md) documentation.
