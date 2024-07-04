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

### How do I customize the `DetectionTrainer` in Ultralytics YOLOv8?

Customizing the `DetectionTrainer` in Ultralytics YOLOv8 involves subclassing and overriding specific methods. For instance, you can extend the existing `get_model` method to load a custom detection model. Here's a sample code snippet:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        return MyCustomModel(cfg, weights)


trainer = CustomTrainer(overrides={...})
trainer.train()
```

In this example, `MyCustomModel` is a user-defined model tailored to specific requirements. For more details on callbacks and other functionalities, refer to the [Callbacks Guide](../usage/callbacks.md).

### What are the key functions to override when customizing the `BaseTrainer` in YOLOv8?

When customizing the `BaseTrainer` in YOLOv8, the essential functions to override are:

- `get_model(cfg, weights)`: Builds the model to be trained.
- `get_dataloader()`: Builds the dataloader.

These functions allow you to integrate custom models and dataloaders into the training loop, ensuring flexibility and adaptability for various machine learning tasks. For a detailed reference, see the [`BaseTrainer` documentation](../reference/engine/trainer.md).

### How can I add a callback to my custom YOLOv8 trainer to perform actions after every training epoch?

To add a callback that performs actions after every training epoch, you can use the `add_callback` method. For instance, if you want to upload the model to Google Drive after every 10 epochs, you can implement it as follows:

```python
from ultralytics.models.yolo.detect import DetectionTrainer


def upload_to_drive(trainer):
    # Custom logic to upload model to Google Drive
    print("Uploading model to Google Drive")


trainer = DetectionTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", upload_to_drive)
trainer.train()
```

This approach helps in automating specific tasks during the training process. For more on callback events, check out our [Callbacks Guide](../usage/callbacks.md).

### Why should I use Ultralytics YOLOv8 for custom model training?

Ultralytics YOLOv8 offers multiple advantages for custom model training, such as high flexibility, state-of-the-art performance, and easy integration with various custom data and model types. The comprehensive API and support for callbacks enable fine-grained control over the training process, ensuring optimal results [learn more](https://docs.ultralytics.com/models/yolov8/).

### What are some practical examples of customizing the YOLOv8 Trainer?

Practical examples of customizing the YOLOv8 Trainer include:

- Overriding the `get_model` method to integrate a custom detection model.
- Customizing the loss function by modifying the `init_criterion`.
- Adding callbacks for tasks like logging or uploading models.

Here is a customization example that alters the `init_criterion`:

```python
from ultralytics.nn.tasks import DetectionModel


class MyCustomModel(DetectionModel):
    def init_criterion(self):
        # Customize loss function initialization
        ...


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        return MyCustomModel(cfg, weights)


trainer = CustomTrainer(overrides={...})
trainer.train()
```

For additional details and examples, visit our [Trainer Documentation](../reference/engine/trainer.md).
