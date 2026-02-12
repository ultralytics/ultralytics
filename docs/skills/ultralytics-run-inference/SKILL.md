---
name: ultralytics-run-inference
description: Run YOLO model inference on images, videos, or streams using Ultralytics. Use when the user needs to make predictions, validate models, or run real-time detection. Covers prediction, validation, and result processing.
license: AGPL-3.0
metadata:
    author: Burhan-Q
    version: "1.0"
    ultralytics-version: ">=8.4.11"
---

# Run YOLO Inference

## When to use this skill

Use this skill when you need to:

- Run predictions on images, videos, or webcam streams
- Validate model performance on a test dataset
- Process and visualize detection results
- Run real-time inference

## Prerequisites

- Python ≥3.8 with PyTorch ≥1.8 installed
- `ultralytics` package installed
    - Cloned repo install or package install
    - `uv pip install ultralytics --upgrade` OR `pip install ultralytics --upgrade`
- Trained YOLO model or pretrained weights

## Inference Methods

### 1. Predict on Images

**Python API:**

```python
from ultralytics import YOLO

# Load model
model = YOLO("yolo26n.pt")  # or path to your custom model

# Predict on single image
results = model("path/to/image.jpg")

# Predict on multiple images
results = model(["image1.jpg", "image2.jpg", "image3.jpg"])

# Predict on directory
results = model("path/to/images/")

# Process results
for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs

    # Visualize
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
```

**CLI:**

```bash
yolo detect predict model=yolo26n.pt source='path/to/image.jpg'
```

### 2. Predict on Videos

**Python API:**

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Predict on video
results = model("path/to/video.mp4", stream=True)

# Process results
for result in results:
    result.show()  # display frame
```

**CLI:**

```bash
yolo detect predict model=yolo26n.pt source='path/to/video.mp4'
```

### 3. Predict on Webcam/Stream

**Python API:**

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Webcam (0 = default camera)
results = model(0, stream=True)

# RTSP stream
results = model("rtsp://example.com/stream", stream=True)

for result in results:
    result.show()
```

**CLI:**

```bash
# Webcam
yolo detect predict model=yolo26n.pt source=0

# RTSP stream
yolo detect predict model=yolo26n.pt source='rtsp://example.com/stream'
```

### 4. Validate Model

**Python API:**

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

# Validate on test dataset
metrics = model.val(data="data.yaml", split="test")

# Access metrics
print(f"mAP50-95: {metrics.box.map}")
print(f"mAP50: {metrics.box.map50}")
print(f"mAP75: {metrics.box.map75}")
```

**CLI:**

```bash
yolo detect val model=yolo26n.pt data=data.yaml
```

## Inference Parameters

| Parameter       | Description                              | Default  |
| --------------- | ---------------------------------------- | -------- |
| `source`        | Image/video/directory/stream path        | Required |
| `conf`          | Confidence threshold (0-1)               | 0.25     |
| `iou`           | NMS IoU threshold (0-1)                  | 0.7      |
| `imgsz`         | Input image size                         | 640      |
| `device`        | GPU device or 'cpu'                      | 0        |
| `max_det`       | Maximum detections per image             | 300      |
| `vid_stride`    | Video frame stride                       | 1        |
| `stream_buffer` | Buffer incoming frames for video streams | False    |
| `visualize`     | Visualize features                       | False    |
| `augment`       | Use test-time augmentation               | False    |
| `agnostic_nms`  | Class-agnostic NMS                       | False    |
| `classes`       | Filter by class (list of IDs)            | None     |
| `retina_masks`  | Use high-res segmentation masks          | False    |
| `embed`         | Return feature vectors                   | None     |

## Processing Results

### Access Detection Data

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg")

for result in results:
    # Bounding boxes
    boxes = result.boxes.xyxy  # box coordinates (x1, y1, x2, y2)
    confidences = result.boxes.conf  # confidence scores
    class_ids = result.boxes.cls  # class IDs

    # Iterate over detections
    for box, conf, cls in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = box
        print(f"Class: {model.names[int(cls)]}, Conf: {conf:.2f}, Box: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
```

### Filter by Class

```python
# Detect only persons (class 0) and cars (class 2)
results = model("image.jpg", classes=[0, 2])
```

### Filter by Confidence

```python
# Only show detections with confidence > 0.5
results = model("image.jpg", conf=0.5)
```

### Save Results

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
results = model("image.jpg", save=True, project="runs/detect", name="exp", exist_ok=False)
```

Results are saved to `runs/detect/exp/`.

### Export Results to JSON

```python
import json

results = model("image.jpg")

detections = []
for result in results:
    for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
        detections.append({"class": model.names[int(cls)], "confidence": float(conf), "bbox": [float(x) for x in box]})

with open("detections.json", "w") as f:
    json.dump(detections, f, indent=2)
```

## Common Issues

**Low Detection Rate:**

- Reduce `conf` threshold
- Use test-time augmentation (`augment=True`)
- Try different `imgsz` values

**Too Many False Positives:**

- Increase `conf` threshold
- Reduce `iou` threshold for stricter NMS

**Slow Inference:**

- Use GPU (`device=0`)
- Reduce `imgsz`
- Use a smaller model (n < s < m < l < x)
- Increase `vid_stride` for videos

## Next Steps

- Export model for deployment: see `ultralytics-export-model` skill
- Train on custom data: see `ultralytics-train-model` skill
- Upload model to [Ultralytics Platform](https://platform.ultralytics.com) for cloud inference deployment or sharing models with the community.

## References

- [Ultralytics Prediction Docs](https://docs.ultralytics.com/modes/predict/)
- [Ultralytics Validation Docs](https://docs.ultralytics.com/modes/val/)
- [Results API](https://docs.ultralytics.com/reference/engine/results/)
