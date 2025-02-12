| Argument     | Type        | Default        | Description                                                                                    |
| ------------ | ----------- | -------------- | ---------------------------------------------------------------------------------------------- |
| `data`       | `str`       | required       | Path to directory containing target images for annotation or segmentation.                     |
| `det_model`  | `str`       | `"yolo11x.pt"` | YOLO detection model path for initial object detection.                                        |
| `sam_model`  | `str`       | `"sam2_b.pt"`  | SAM2 model path for segmentation (supports t/s/b/l variants and SAM2.1) and mobile_sam models. |
| `device`     | `str`       | `""`           | Computation device (e.g., 'cuda:0', 'cpu', or '' for automatic device detection).              |
| `conf`       | `float`     | `0.25`         | YOLO detection confidence threshold for filtering weak detections.                             |
| `iou`        | `float`     | `0.45`         | IoU threshold for Non-Maximum Suppression to filter overlapping boxes.                         |
| `imgsz`      | `int`       | `640`          | Input size for resizing images (must be multiple of 32).                                       |
| `max_det`    | `int`       | `300`          | Maximum number of detections per image for memory efficiency.                                  |
| `classes`    | `list[int]` | `None`         | List of class indices to detect (e.g., `[0, 1]` for person & bicycle).                         |
| `output_dir` | `str`       | `None`         | Save directory for annotations (defaults to './labels' relative to data path).                 |
