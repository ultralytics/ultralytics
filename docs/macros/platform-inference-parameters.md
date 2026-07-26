| Parameter   | Type   | Default | Range      | Description                                        |
| ----------- | ------ | ------- | ---------- | -------------------------------------------------- |
| `file`      | file   | -       | -          | Image or video file (required unless `source` set) |
| `conf`      | float  | 0.25    | 0.01 – 1.0 | Minimum confidence threshold                       |
| `iou`       | float  | 0.7     | 0.0 – 0.95 | NMS IoU threshold                                  |
| `imgsz`     | int    | 640     | 32 – 1280  | Input image size in pixels                         |
| `normalize` | bool   | false   | -          | Return bounding box coordinates as 0 – 1           |
| `decimals`  | int    | 5       | 0 – 10     | Decimal precision for coordinate values            |
| `source`    | string | -       | -          | Image URL or base64 string (alternative to `file`) |
