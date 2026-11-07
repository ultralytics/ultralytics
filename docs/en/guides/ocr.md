---
title: Scene Text OCR with YOLO26
comments: true
description: Detect, recognize, and order scene text with Ultralytics YOLO26 detectors and optional PaddleOCR recognition.
keywords: Ultralytics, YOLO26, OCR, scene text, text detection, text recognition, PaddleOCR
---

# Scene Text OCR using Ultralytics YOLO26

## What is Scene Text OCR?

Optical character recognition (OCR) detects and reads text in images. The [OCR solution](../reference/solutions/ocr.md)
combines an Ultralytics YOLO26 text detector with a text-recognition backend. It supports axis-aligned detection boxes,
oriented boxes (OBB), and segmentation polygons, then returns recognized text in reading order.

Detection and recognition remain separate components, so you can pair a detector trained for your scene-text domain with
PaddleOCR or your own recognition model.

## Why Use Scene Text OCR?

- **Flexible text geometry:** Use detection, OBB, or segmentation models for horizontal, rotated, or irregular text.
- **Interchangeable recognition:** Run PaddleOCR without extra integration code, or pass a callable for another model.
- **Structured output:** Access recognized text, recognition and detection confidences, source geometry, and an annotated
  image from one results object.

## Run Scene Text OCR with YOLO26

PaddleOCR is optional and loads only when recognition first runs. Install a PaddlePaddle build appropriate for your
platform and PaddleOCR 3.x; no OCR packages are added to the base Ultralytics dependencies.

```bash
pip install paddlepaddle "paddleocr>=3.0,<4"
```

Pass an image and a text detector to the `OCR` solution. The result contains the annotated image and recognized text:

!!! example "Scene Text OCR using Ultralytics YOLO26"

    === "CLI"

        ```bash
        yolo solutions ocr model="scene-text-obb.pt" source="path/to/image.jpg" recognizer="paddleocr"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        ocr = solutions.OCR(
            model="scene-text-obb.pt",  # custom detect, OBB, or segmentation text detector
            recognizer="paddleocr",
        )
        im0 = cv2.imread("path/to/image.jpg")
        assert im0 is not None, "Error reading image file"
        result = ocr(im0)
        cv2.imwrite("ocr.jpg", result.plot_im)

        for text in result.texts:
            print(text["text"], text["confidence"])
        ```

### YOLO26-Text Checkpoints

The public YOLO26-Text family provides `n`, `s`, `m`, `l`, and `x` OBB checkpoints trained on COCO-Text and HierText at
640 pixels. Set `ULTRALYTICS_API_KEY` to your own [Ultralytics Platform](https://platform.ultralytics.com/settings) API
key, then load a checkpoint by its Platform URI:

```bash
export ULTRALYTICS_API_KEY="your-api-key"
```

```python
ocr = solutions.OCR(model="ul://ultralytics-yolo/yolo26-text/yolo26n-text", recognizer="paddleocr")
result = ocr(im0)
```

Replace `yolo26n-text` with `yolo26s-text`, `yolo26m-text`, `yolo26l-text`, or `yolo26x-text` for a larger model.

### `OCR` Arguments

Here's a table with the `OCR` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "recognizer", "line_width", "verbose"]) }}

Additionally, the following visualization settings are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "show_conf", "show_labels", "show_boxes"]) }}

## How Scene Text OCR Works

1. **Detect text:** YOLO26 produces axis-aligned boxes, oriented boxes, or segmentation polygons.
2. **Create recognition crops:** Boxes are clipped to the image. OBBs are affine-rectified using their corners. For
   segmentation polygons, the minimum-area rotated rectangle is rectified while the original polygon is retained.
3. **Recognize text:** PaddleOCR processes all crops in batches of 16. Custom recognizers receive one crop at a time.
4. **Order and render results:** Text is grouped into horizontal lines and ordered top-to-bottom and left-to-right. The
   visualization uses a Unicode font for non-ASCII text such as Chinese.

Coordinates always refer to the original image, not the detector input or recognition crop.

## Use a Custom Recognizer

Pass a callable to integrate another recognition model without changing the detection, geometry, or ordering path. The
callable runs once for each nonempty BGR `numpy.ndarray` crop. Return one `(text, confidence)` tuple per call, not a
list of batch results:

| Tuple item   | Accepted type                       | Description                                                           |
| ------------ | ----------------------------------- | --------------------------------------------------------------------- |
| `text`       | `str`                               | Decoded text for this crop. Return `""` when no text is recognized.   |
| `confidence` | `float`, `int`, or `numpy.floating` | Numeric recognition confidence, converted to `float` by the solution. |

```python
def recognize(crop):
    # Run your recognition model on the BGR crop and decode its output here.
    return "example", 0.95


ocr = solutions.OCR(model="customer-text.pt", recognizer=recognize)
result = ocr(im0)
```

Non-string text or nonnumeric confidence raises a `TypeError`. Unsupported backend names and missing Paddle dependencies
produce actionable errors. When no text is detected, `result.texts` is empty and the recognizer is not called.

## OCR Results

Each call returns a [`SolutionResults`](../reference/solutions/solutions.md#ultralytics.solutions.solutions.SolutionResults)
object. Use `result.plot_im` for the annotated image and `result.texts` for records in deterministic reading order:

| Key                    | Type                | Description                                                                 |
| ---------------------- | ------------------- | --------------------------------------------------------------------------- |
| `text`                 | `str`               | Recognized text.                                                            |
| `confidence`           | `float`             | Recognition confidence, kept separate from detection confidence.            |
| `detection_confidence` | `float`             | YOLO detector confidence.                                                   |
| `detection_index`      | `int`               | Source detection index before reading-order sorting.                        |
| `box`                  | `list[list[float]]` | Four original-image corners for box or OBB detections.                      |
| `polygon`              | `list[list[float]]` | Original-image segmentation polygon; replaces `box` for polygon detections. |

## Train a Text Detector

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

## Dependencies and Licenses

| Component       | Required                          | Role                                                    | License                |
| --------------- | --------------------------------- | ------------------------------------------------------- | ---------------------- |
| Ultralytics     | Yes                               | YOLO detection, crop geometry, and result orchestration | AGPL-3.0               |
| PaddlePaddle    | Only for `recognizer="paddleocr"` | Recognition runtime                                     | Apache-2.0             |
| PaddleOCR 3.x   | Only for `recognizer="paddleocr"` | Text-line recognition models and inference adapter      | Apache-2.0             |
| Custom callable | No                                | User-owned recognition integration                      | Defined by the backend |

PaddleOCR may download recognition weights on first use. Review the selected model's model card and redistribution
terms before packaging or deploying those artifacts. Dataset and custom checkpoint licenses are separate.

## FAQ

### Can I use my own OCR model instead of PaddleOCR?

Yes. Pass a callable as `recognizer` and return `(text, confidence)` for each BGR crop. See
[Use a Custom Recognizer](#use-a-custom-recognizer) for the input and output contract.

### Does OCR track text across video frames?

No. OCR treats every input image independently, without track IDs or cached recognition. Process each video frame as a
separate input when frame-by-frame OCR is required.

### Can I export the complete OCR pipeline?

You can export the YOLO detector with the normal [Export mode](../modes/export.md), but PaddleOCR or a custom recognizer
is **not** embedded in the ONNX, TensorRT, or other detector artifact. Deploy the exported detector and recognition
runtime as separate components, or keep this Python composition layer.
