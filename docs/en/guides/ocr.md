---
title: Scene Text OCR with YOLO
comments: true
description: Detect, recognize, order, and track scene text with Ultralytics YOLO detectors and optional PaddleOCR recognition.
keywords: Ultralytics, YOLO, OCR, scene text, text detection, text recognition, PaddleOCR, video OCR
---

# Scene Text OCR with YOLO

The `OCR` solution combines an Ultralytics YOLO text detector with text recognition. It handles axis-aligned boxes,
oriented boxes (OBB), segmentation polygons, recognition crops, reading order, normalized results, and video text/track
association. Detection and recognition remain separate components: use a detector trained for your scene-text domain.

## Quickstart

PaddleOCR is optional and loads only when recognition first runs. Install a PaddlePaddle build appropriate for your
platform and PaddleOCR 3.x; no OCR packages are added to the base Ultralytics dependencies.

```bash
pip install paddlepaddle "paddleocr>=3.0,<4"
yolo solutions ocr model="scene-text-obb.pt" source="path/to/video.mp4" recognizer="paddleocr"
```

```python
import cv2

from ultralytics import solutions

ocr = solutions.OCR(
    model="scene-text-obb.pt",  # custom detect, OBB, or segmentation text detector
    recognizer="paddleocr",
    tracker="bytetrack.yaml",
)
im0 = cv2.imread("path/to/image.jpg")
result = ocr(im0)
cv2.imwrite("ocr.jpg", result.plot_im)

for text in result.texts:
    print(text["text"], text["confidence"], text.get("track_id"))
```

A Platform scene-text checkpoint uses the same interface after the catalog owner supplies its approved artifact and
model card:

```python
ocr = solutions.OCR(model="platform-scene-text.pt", recognizer="paddleocr")
result = ocr(im0)
```

!!! warning "Checkpoint availability"

    The example filename is a placeholder, not a published model identifier. Use the exact artifact name and geometry
    from the Platform model catalog. An unavailable checkpoint is not replaced implicitly with a general COCO model.

## Custom recognizer

Pass a callable to integrate another recognizer without changing the detection, geometry, ordering, or tracking path.
The callable receives one BGR `numpy.ndarray` crop and returns `(text, recognition_confidence)`.

```python
def recognize(crop):
    return "example", 0.95


ocr = solutions.OCR(model="customer-text.pt", recognizer=recognize)
result = ocr(im0)  # feed frames from labels.mp4 through the existing video loop
```

Invalid backend output raises a `TypeError`. Unsupported backend names and missing Paddle dependencies produce
actionable errors. Empty detections and empty recognition lists are normal results, not backend failures.

## Results schema

Each call returns an [`engine.Results`](../reference/engine/results.md) object. `result.texts` is a list in deterministic
reading order; it is empty when no text is detected. Ordinary non-OCR results use `texts=None`.

| Key                    | Type                | Description                                                                 |
| ---------------------- | ------------------- | --------------------------------------------------------------------------- |
| `text`                 | `str`               | Recognized text.                                                            |
| `confidence`           | `float`             | Recognition confidence, kept separate from detection confidence.            |
| `detection_confidence` | `float`             | YOLO detector confidence.                                                   |
| `detection_index`      | `int`               | Source detection index before reading-order sorting.                        |
| `box`                  | `list[list[float]]` | Four original-image corners for box or OBB detections.                      |
| `polygon`              | `list[list[float]]` | Original-image segmentation polygon; replaces `box` for polygon detections. |
| `track_id`             | `int`               | Track ID when tracking supplies one.                                        |

`result.summary()`, `result.to_json()`, and other `DataExportMixin` formats retain `text` and `text_confidence` on the
corresponding detection row. The existing detection `confidence` remains unchanged. Indexing and tensor/device result
conversions preserve text-to-detection alignment.

Entries are grouped into horizontal lines, ordered top-to-bottom and left-to-right, with source detection indices as
stable tie-breakers. Coordinates always refer to the original image, not the detector input or recognition crop.

## Geometry and video behavior

- **Boxes:** clipped axis-aligned recognition crops.
- **OBBs:** affine-rectified recognition crops using the detected corners.
- **Segmentation polygons:** the minimum-area rotated rectangle is rectified for recognition; the original polygon is
  retained in `result.texts`.
- **Tracking:** still images use prediction by default (`tracker=None`), avoiding persistent-state filtering between
  independent images. Set a tracker such as `bytetrack.yaml` for video association; the OCR CLI uses ByteTrack by default.
  Recognition runs for a new track and refreshes every `recognition_interval` frames (default `30`). Valid cached text is
  reused between refreshes and when the same track returns after temporary occlusion.
- **Changed text:** appears at the next scheduled refresh. Retired track IDs and their cached text are removed. A new
  track ID is not assumed to represent previous text.

Adjust the refresh interval for the scene and latency budget:

```python
ocr = solutions.OCR(model="scene-text-obb.pt", recognizer="paddleocr", recognition_interval=10)
```

### OCR arguments

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "recognizer", "recognition_interval", "tracker", "verbose"]) }}

{% include "macros/solutions-track-args.md" %}

## Train a text detector

Train the detector with the standard YOLO detection, segmentation, or OBB workflow; OCR does not add a native training
task or train the external recognizer. For rotated scene text with quadrilateral annotations, use an OBB dataset and
disable augmentations that corrupt text orientation:

```bash
yolo obb train model="yolo26n-obb.pt" data="scene-text-obb.yaml" epochs=100 imgsz=1024 seed=0 flipud=0.0
yolo obb val model="runs/obb/train/weights/best.pt" data="scene-text-obb.yaml" split=test
```

Record the dataset source, license, checksums, split, augmentation settings, seed, and evaluation labels. Detection
metrics alone do not measure recognition quality: evaluate character error rate (CER) and word error rate (WER) only
when transcripts are available, and distinguish ground-truth-crop recognition from end-to-end OCR.

## Dependencies and licenses

| Component       | Required                          | Role                                                         | License                |
| --------------- | --------------------------------- | ------------------------------------------------------------ | ---------------------- |
| Ultralytics     | Yes                               | YOLO detection, geometry, tracking, and result orchestration | AGPL-3.0               |
| PaddlePaddle    | Only for `recognizer="paddleocr"` | Recognition runtime                                          | Apache-2.0             |
| PaddleOCR 3.x   | Only for `recognizer="paddleocr"` | Text-line recognition models and inference adapter           | Apache-2.0             |
| Custom callable | No                                | User-owned recognition integration                           | Defined by the backend |

PaddleOCR may download recognition weights on first use. Review the selected model's model card and redistribution
terms before packaging or deploying those artifacts. Dataset and custom checkpoint licenses are separate.

## Export and deployment

Export the YOLO detector with the normal [Export mode](../modes/export.md). The composed Python OCR solution, including
PaddleOCR or a custom recognizer, is **not** embedded in a YOLO ONNX, TensorRT, or other detector export. Deploy the
exported detector and recognition runtime as separate components, or keep this Python composition layer.
