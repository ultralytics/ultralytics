# Shared Helpers for the Inference in Python

A small module shared across the [Ultralytics YOLO](https://docs.ultralytics.com) Python inference examples. It depends only on [NumPy](https://numpy.org/) and [OpenCV](https://opencv.org/). A backend only runs the model; everything else is handled here.

An example uses it by adding this folder to the import path:

```python
sys.path.append(str(Path(__file__).resolve().parents[1] / "common"))
import yolo_common as yolo
```

| Function      | Provides                                                                                                     |
| ------------- | ------------------------------------------------------------------------------------------------------------ |
| `parse_args`  | The command-line arguments shared by every example (`--model`, `--source`, `--conf`, `--iou`, `--out`, ...). |
| `preprocess`  | Letterbox (stretch for RT-DETR, center crop for classify), BGR to RGB, normalize, FP32 or FP16 blob.         |
| `postprocess` | NMS plus box, mask, keypoint, OBB, RT-DETR, class, and semantic decoding into a result dict.                 |
| `draw`        | Annotates the image with boxes, masks, skeletons, rotated boxes, class maps, or top-5 classes.               |
| `finish`      | Prints the detections, saves the annotated image to `--out`, and optionally shows it with `--show`.          |

`COCO_NAMES` holds the 80 COCO class names as a fallback for backends that read no model metadata.
