This is the simplest way of simply using YOLOv8 models in a Python environment. It can be imported from
the `ultralytics` module.

!!! example "Train"

    === "From pretrained(recommanded)"
        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt") # pass any model type
        model.train(epochs=5)
        ```

    === "From scratch"
        ```python
        from ultralytics import YOLO

        model = YOLO("yolov8n.yaml")
        model.train(data="coco128.yaml", epochs=5)
        ```

    === "Resume"
        ```python
        TODO: Resume feature is under development and should be released soon.
        ```

!!! example "Val"

    === "Val after training"
        ```python
          from ultralytics import YOLO

          model = YOLO("yolov8n.yaml")
          model.train(data="coco128.yaml", epochs=5)
          model.val()  # It'll automatically evaluate the data you trained.
        ```

    === "Val independently"
        ```python
          from ultralytics import YOLO

          model = YOLO("model.pt")
          # It'll use the data yaml file in model.pt if you don't set data.
          model.val()
          # or you can set the data you want to val
          model.val(data="coco128.yaml")
        ```

!!! example "Predict"

    === "From source"
        ```python
        from ultralytics import YOLO

        model = YOLO("model.pt")
        model.predict(source="0") # accepts all formats - img/folder/vid.*(mp4/format). 0 for webcam
        model.predict(source="folder", show=True) # Display preds. Accepts all yolo predict arguments

        ```

    === "From image/ndarray/tensor"
        ```python
        # TODO, still working on it.
        ```


    === "Return outputs"
        ```python
        from ultralytics import YOLO

        model = YOLO("model.pt")
        outputs = model.predict(source="0", return_outputs=True) # treat predict as a Python generator
        for output in outputs:
          # each output here is a dict.
          # for detection
          print(output["det"])  # np.ndarray, (N, 6), xyxy, score, cls
          # for segmentation
          print(output["det"])  # np.ndarray, (N, 6), xyxy, score, cls
          print(output["segment"])  # List[np.ndarray] * N, bounding coordinates of masks
          # for classify
          print(output["prob"]) # np.ndarray, (num_class, ), cls prob

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

`YOLO` model class is a high-level wrapper on the Trainer classes. Each YOLO task has its own trainer that inherits
from `BaseTrainer`.

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
Learn more about Customizing `Trainers`, `Validators` and `Predictors` to suit your project needs in the Customization
Section.

[Customization tutorials](engine.md){ .md-button .md-button--primary}
