## Installation

Install YOLOv8 via the `ultralytics` pip package for the latest stable release or by cloning the [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) repository for the most up-to-date version.

!!! note "pip install (recommended)"
    ```
    pip install ultralytics
    ```
!!! note "git clone"
    ```
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install -e '.[dev]'
    ```
    See contributing section to know more about contributing to the project


## CLI
The command line YOLO interface lets you simply train, validate or infer models on various tasks and versions.
CLI requires no customization or code. You can simply run all tasks from the terminal with the `yolo` command.
!!! note
    === "Syntax"
        ```bash
        yolo task=detect    mode=train    model=yolov8n.yaml      args...
                  classify       predict        yolov8n-cls.yaml  args...
                  segment        val            yolov8n-seg.yaml  args...
                                 export         yolov8n.pt        format=onnx  args...
        ```

    === "Example training"
        ```bash
        yolo task=detect mode=train model=yolov8n.pt data=coco128.yaml device=0
        ```
    === "Example Multi-GPU training"
        ```bash
        yolo task=detect mode=train model=yolov8n.pt data=coco128.yaml device=\'0,1,2,3\'
        ```
[CLI Guide](cli.md){ .md-button .md-button--primary}

## Python API
The Python API allows users to easily use YOLOv8 in their Python projects. It provides functions for loading and running the model, as well as for processing the model's output. The interface is designed to be easy to use, so that users can quickly implement object detection in their projects.

Overall, the Python interface is a useful tool for anyone looking to incorporate object detection, segmentation or classification into their Python projects using YOLOv8.

!!! note
    ```python
    from ultralytics import YOLO

    model = YOLO('yolov8n.yaml')                # build a new model from scratch
    model = YOLO('yolov8n.pt')                  # load a pretrained model (recommended for best training results)
    results = model.train(data='coco128.yaml')  # train the model
    results = model.val()                       # evaluate model performance on the validation set
    results = model.predict(source='bus.jpg')   # predict on an image
    success = model.export(format='onnx')       # export the model to ONNX format
    ```
[API Guide](sdk.md){ .md-button .md-button--primary}
