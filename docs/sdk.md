## Using YOLO models
This is the simplest way of simply using yolo models in a python environment. It can be imported from the `ultralytics` module.

!!! example "Usage"
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

    More functionality coming soon

To know more about using `YOLO` models, refer Model class refernce

[Model reference](#){ .md-button .md-button--primary}

---
### Customizing Tasks with Trainers
`YOLO` model class is a high-level wrapper on the Trainer classes. Each YOLO task has its own trainer that inherits from `BaseTrainer`. 
You can easily cusotmize Trainers to support custom tasks or explore R&D ideas.

!!! tip "Trainer Examples"
    === "DetectionTrainer"
        ```python
        from ultralytics import yolo

        trainer = yolo.DetectionTrainer(data=..., epochs=1) # override default configs
        trainer.train()
        ```

    === "SegmentationTrainer"
        ```python
        from ultralytics import yolo

        trainer = yolo.SegmentationTrainer(data=..., epochs=1) # override default configs
        trainer.train()
        ```
    === "ClassificationTrainer"
        ```python
        from ultralytics import yolo

        trainer = yolo.ClassificationTrainer(data=..., epochs=1) # override default configs
        trainer.train()
        ```

Learn more about Customizing `Trainers`, `Validators` and `Predictors` to suit your project needs in the Customization Section. More details about the base engine classes is available in the reference section.

[Customization tutorials](#){ .md-button .md-button--primary}
