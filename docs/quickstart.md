## Installation
!!! note "Latest Stable Release"
    ```
    pip install ultralytics
    ```
??? tip "Development and Contributing"
    ```
    git clone https://github.com/ultralytics/ultralytics
    cd ultralytics
    pip install -e '.[dev]'
    ```
    See contributing section to know more about contributing to the project


## CLI
The command line YOLO interface let's you simply train, validate or infer models on various tasks and versions.
CLI requires no customization or code. You can simply run all tasks from the terminal
!!! tip
    === "Syntax"
        ```bash
        yolo task=detect    mode=train  model=s.yaml    epochs=1 ...
                   ...             ...          ...
                 segment          infer       s-cls.pt
                 classify         val         s-seg.pt
        ```

    === "Example training"
        ```bash
        yolo task=detect mode=train model=s.yaml 
        ```
        TODO:  add terminal screen/gif
    === "Example training DDP"
        ```bash
        yolo task=detect mode=train model=s.yaml device=\'0,1,2,3\'
        ```
[CLI Guide](cli.md){ .md-button .md-button--primary}

## Python API
Ultralytics YOLO comes with pythonic Model and Trainer interface. 
!!! tip
    ```python
    import ultralytics
    from ultralytics import YOLO

    model = YOLO("yolov8n-seg.yaml") # automatically detects task type
    model = YOLO("yolov8n.pt") # load checkpoint
    model.train(data="coco128-seg.yaml", epochs=1, lr0=0.01, ...)
    model.train(data="coco128-seg.yaml", epochs=1, lr0=0.01, device="0,1,2,3") # DDP mode
    ```
[API Guide](sdk.md){ .md-button .md-button--primary}
