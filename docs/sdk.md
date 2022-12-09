# Python SDK

We provide 2 pythonic interfaces for YOLO models:

<b> Model Interface </b> - High-level wrapper around the `Trainer` Classes

<b> Trainers </b> - Provides access to Trainer Classes that can be extended for pursuing R&D ideas.

______________________________________________________________________

### Model Interface
This is the easiest, no-nonsense way of simply using yolo models in a python environment. 
It can be imported from the `ultralytics` module.
!!! tip "Model Usage"

    | Name                       |Function                                      | Usage                             |           
    | --------------------------------| -------------------------------------------- | --------------------------------- |       
    |` YOLO()`                     | Initialize a model wrapper. **Optionally**, <br> accepts model `type` which defaults to `'v8'`                      |                         ```model = YOLO()```                                                     |   |
    |`.new(cfg)`                  | Initialize a new model from a given config `cfg`                                                                | ```model.new("cfg.yaml")```                                                                      |   |
    |`.load(model)`               | Load a model from a given model weight `model`                                                                  | ```model.load("model.pt")```                                                                     |
    |`__call__(imgs)`             | Runs inference on given `imgs` tensor                                                                           | ```model(imgs)```                                                                                |
    |`train(data, **kwargs) `     | Trains the model on given `data`. Accepts additional train args can be passed                              | ```model.train(data="", epochs=1, ...)```                                                        |   |
    |
    |`.resume(task)`         | Resumes training of the last given `task` type if **optionally** `model` is not passed, else resumes from the given model | ```model.resume(task="detect") model.resume(task="detect", model="*.pt")``` |   |
    
    !!! note ""
        More functionality coming soon
Model Class reference >

!!! example "Examples"
    === "Training"
        ```python
        from ultralytics import YOLO

        model = YOLO()
        model.new("n.yaml") # pass any model type
        model.train(data="coco128.yaml", epochs=5)
        ```

    === "Training pretrained"
        ```python
        from ultralytics import YOLO

        model = YOLO()
        model.load("n.pt") # pass any model type
        model(...) # inference
        model.train(data="coco128.yaml", epochs=5)
        ```

    === "Resume Training"
        ```python
        from ultralytics import YOLO

        model = YOLO()
        model.resume(task="detect") # resume last detection training
        model.resume(task="detect", model="last.pt") # resume from a given model
        ```

### Working with Trainers
