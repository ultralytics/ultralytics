---
comments: true
description: Discover how to customize and extend base Ultralytics YOLO Trainer engines. Support your custom model and dataloader by overriding built-in functions.
keywords: Ultralytics, YOLO, trainer engines, BaseTrainer, DetectionTrainer, customizing trainers, extending trainers, custom model, custom dataloader
---

Both the Ultralytics YOLO command-line and python interfaces are simply a high-level abstraction on the base engine executors. Let's take a look at the Trainer engine.

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
    def get_model(self, cfg, weights): ...


trainer = CustomTrainer(overrides={...})
trainer.train()
```

You now realize that you need to customize the trainer further to:

- Customize the `loss function`.
- Add `callback` that uploads model to your Google Drive after every 10 `epochs`
  Here's how you can do it:

```python
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel


class MyCustomModel(DetectionModel):
    def init_criterion(self): ...


class CustomTrainer(DetectionTrainer):
    def get_model(self, cfg, weights):
        return MyCustomModel(...)


# callback to upload model weights
def log_model(trainer):
    last_weight_path = trainer.last
    ...


trainer = CustomTrainer(overrides={...})
trainer.add_callback("on_train_epoch_end", log_model)  # Adds to existing callback
trainer.train()
```

To know more about Callback triggering events and entry point, checkout our [Callbacks Guide](callbacks.md)

## Other engine components

There are other components that can be customized similarly like `Validators` and `Predictors`
See Reference section for more information on these.
