The simplest way of simply using YOLOv8 directly in a Python environment.

!!! example "Train"

    === "From pretrained(recommended)"
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
        # TODO: Resume feature is under development and should be released soon.
        model = YOLO("last.pt")
        model.train(resume=True)
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
        from PIL import Image
        import cv2

        model = YOLO("model.pt")
        # accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam
        results = model.predict(source="0")
        results = model.predict(source="folder", show=True) # Display preds. Accepts all YOLO predict arguments

        # from PIL
        im1 = Image.open("bus.jpg")
        results = model.predict(source=im1, save=True)  # save plotted images

        # from ndarray
        im2 = cv2.imread("bus.jpg")
        results = model.predict(source=im2, save=True, save_txt=True)  # save predictions as labels

        # from list of PIL/ndarray
        results = model.predict(source=[im1, im2])
        ```

    === "Results usage"
        ```python
        # results would be a list of Results object including all the predictions by default
        # but be careful as it could occupy a lot memory when there're many images, 
        # especially the task is segmentation.
        # 1. return as a list
        results = model.predict(source="folder")

        # results would be a generator which is more friendly to memory by setting stream=True
        # 2. return as a generator
        results = model.predict(source=0, stream=True)

        for result in results:
            # detection
            result.boxes.xyxy   # box with xyxy format, (N, 4)
            result.boxes.xywh   # box with xywh format, (N, 4)
            result.boxes.xyxyn  # box with xyxy format but normalized, (N, 4)
            result.boxes.xywhn  # box with xywh format but normalized, (N, 4)
            result.boxes.conf   # confidence score, (N, 1)
            result.boxes.cls    # cls, (N, 1)

            # segmentation
            result.masks.masks     # masks, (N, H, W)
            result.masks.segments  # bounding coordinates of masks, List[segment] * N

            # classification
            result.probs     # cls prob, (num_class, )

        # Each result is composed of torch.Tensor by default, 
        # in which you can easily use following functionality:
        result = result.cuda()
        result = result.cpu()
        result = result.to("cpu")
        result = result.numpy()
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

[Model reference](../reference/model.md){ .md-button .md-button--primary}

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
