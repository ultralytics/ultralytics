| Argument  | Type    | Default          | Description                                                                                                                                                |
| --------- | ------- | ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `source`  | `str`   | `None`           | Specifies the source directory for images or videos. Supports file paths and URLs.                                                                         |
| `persist` | `bool`  | `False`          | Enables persistent tracking of objects between frames, maintaining IDs across video sequences.                                                             |
| `tracker` | `str`   | `'botsort.yaml'` | Specifies the tracking algorithm to use, e.g., `bytetrack.yaml` or `botsort.yaml`.                                                                         |
| `conf`    | `float` | `0.3`            | Sets the confidence threshold for detections; lower values allow more objects to be tracked but may include false positives.                               |
| `iou`     | `float` | `0.5`            | Sets the [Intersection over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) (IoU) threshold for filtering overlapping detections. |
| `classes` | `list`  | `None`           | Filters results by class index. For example, `classes=[0, 2, 3]` only tracks the specified classes.                                                        |
| `verbose` | `bool`  | `True`           | Controls the display of tracking results, providing a visual output of tracked objects.                                                                    |
