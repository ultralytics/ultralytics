| Argument     | Type                    | Description                                                                                             | Default        |
| ------------ | ----------------------- | ------------------------------------------------------------------------------------------------------- | -------------- |
| `data`       | `str`                   | Path to a folder containing images to be annotated.                                                     |                |
| `det_model`  | `str`, optional         | Pre-trained YOLO detection model. Defaults to 'yolo11x.pt'.                                             | `"yolo11x.pt"` |
| `sam_model`  | `str`, optional         | Pre-trained SAM 2 segmentation model. Defaults to 'sam2_b.pt'.                                          | `"sam_b.pt"`   |
| `device`     | `str`, optional         | Device to run the models on. Defaults to an empty string (CPU or GPU, if available).                    | `""`           |
| `conf`       | `float`, optional       | Confidence threshold for detection model; default is 0.25.                                              | `0.25`         |
| `iou`        | `float`, optional       | IoU threshold for filtering overlapping boxes in detection results; default is 0.45.                    | `0.45`         |
| `imgsz`      | `int`, optional         | Input image resize dimension; default is 640.                                                           | `640`          |
| `max_det`    | `int`, optional         | Limits detections per image to control outputs in dense scenes.                                         | `300`          |
| `classes`    | `list`, optional        | Filters predictions to specified class IDs, returning only relevant detections.                         | `None`         |
| `output_dir` | `str`, `None`, optional | Directory to save the annotated results. Defaults to a 'labels' folder in the same directory as 'data'. | `None`         |
