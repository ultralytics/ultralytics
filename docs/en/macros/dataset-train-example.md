!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("{{ model_name }}")  # load a pretrained model (recommended for training)

        # Train the model
        results = model.train(data="{{ dataset }}", epochs=100, imgsz={{imgsz}})
        ```

    === "CLI"

        ```bash
        # Start training from a pretrained *.pt model
        yolo {{ task }} train data={{ dataset }} model={{ model_name }} epochs=100 imgsz={{imgsz}}
        ```
