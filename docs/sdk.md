## Using YOLO models
This is the simplest way of simply using yolo models in a python environment. It can be imported from the `ultralytics` module.

!!! example "Usage"
    === "Training"
        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.yaml")
        model(img_tensor) # Or model.forward(). inference.
        model.train(data="coco128.yaml", epochs=5)
        ```

    === "Training pretrained"
        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt") # pass any model type
        model(...) # inference
        model.train(epochs=5)
        ```

    === "Resume Training"
        ```python
        from ultralytics import YOLO

        model = YOLO()
        model.resume(task="detect") # resume last detection training
        model.resume(model="last.pt") # resume from a given model/run
        ```
    
    === "Visualize/save Predictions"
    ```python
    from ultralytics import YOLO

    model = YOLO("model.pt")
    model.predict(source="0") # accepts all formats - img/folder/vid.*(mp4/format). 0 for webcam
    model.predict(source="folder", view_img=True) # Display preds. Accepts all yolo predict arguments

    ```

!!! note "Export and Deployment"

    === "Export, Fuse & info" 
        ```python
        from ultralytics import YOLO

        model = YOLO("model.pt")
        model.fuse()  
        model.info(verbose=True)  # Print model information
        model.export(format=)  # TODO: 

        ```
    === "Deployment"


    More functionality coming soon

To know more about using `YOLO` models, refer Model class Reference

[Model reference](reference/model.md){ .md-button .md-button--primary}

---
### Using Trainers
`YOLO` model class is a high-level wrapper on the Trainer classes. Each YOLO task has its own trainer that inherits from `BaseTrainer`. 
!!! tip "Detection Trainer Example"
        ```python
        from ultralytics.yolo import v8 import DetectionTrainer, DetectionValidator, DetectionPredictor

        # trainer
        trainer = DetectionTrainer(overrides={})
        trainer.train()
        trained_model = trainer.best

        # Validator
        val = DetectionValidator(args=...)
        val(model=trained_model)

        # predictor
        pred = DetectionPredictor(overrides={})
        pred(source=SOURCE, model=trained_model)

        # resume from last weight
        overrides["resume"] = trainer.last
        trainer = detect.DetectionTrainer(overrides=overrides)

        ```
You can easily customize Trainers to support custom tasks or explore R&D ideas.
Learn more about Customizing `Trainers`, `Validators` and `Predictors` to suit your project needs in the Customization Section.

[Customization tutorials](engine.md){ .md-button .md-button--primary}
