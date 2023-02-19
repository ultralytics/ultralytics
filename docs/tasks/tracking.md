Object tracking is a task that involves identifying the location and class of objects, then assigning a unique ID to that detection in an image or video stream.

The output of tracker is the same as detection with an added object ID.
## Available Trackers
The following tracking algorithms have been implemented and can be enabled by passing `tracker=tracker_type.yaml`

* Bot-Sort - `botsort.yaml`

* ByteTrack - `bytetrack.yaml`

The default tracker is botsort

## Predict

Use a trained YOLOv8n model to run predictions on images.

!!! example ""

    === "Python"
    
        ```python
        from ultralytics import YOLO
        
        # Load a model
        model = YOLO("yolov8n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model
        
        # Track with the model
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True) 
        results = model.track(source="https://youtu.be/Zgi9g1ksQHc", show=True, tracker="bytetrack.yaml") 
        ```
    === "CLI"
    
        ```bash
        yolo track model=yolov8n.pt source="https://youtu.be/Zgi9g1ksQHc"  # official model
        yolo track predict model=path/to/best.pt source=...  # custom model
        yolo track predict model=path/to/best.pt  tracker="bytetrack.yaml" # bytetrack tracker

        ```