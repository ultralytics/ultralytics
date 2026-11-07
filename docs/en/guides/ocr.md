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

Detection and recognition remain separate components, so you can pair your scene-text detector with PaddleOCR or your
own recognition model.

## Why Use Scene Text OCR?

- **Flexible text geometry:** Use detection, OBB, or segmentation models for horizontal, rotated, or irregular text.
- **Interchangeable recognition:** Run PaddleOCR without extra integration code, or pass a callable for another model.
- **Structured output:** Access recognized text, recognition and detection confidences, source geometry, and an annotated
  image from one results object.

## Run Scene Text OCR with YOLO26

PaddleOCR is optional and loads when `OCR` is created with `recognizer="paddleocr"`, installing PaddleOCR 3.7 or newer
(PP-OCRv6 recognition) and the CPU PaddlePaddle 3.1+ build if they are missing. For GPU recognition, install a matching
`paddlepaddle-gpu` 3.1+ build first; no OCR packages are added to the base Ultralytics dependencies.

Pass an image and a text detector to the `OCR` solution. The result contains the annotated image and recognized text:

!!! example "Scene Text OCR using Ultralytics YOLO26"

    === "CLI"

        ```bash
        yolo solutions ocr model="scene-text-obb.pt" source="path/to/video.mp4" recognizer="paddleocr"
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

The public YOLO26-Text family provides `n`, `s`, `m`, `l`, and `x` OBB checkpoints. Set `ULTRALYTICS_API_KEY` to your
own [Ultralytics Platform](https://platform.ultralytics.com/settings) API key, then load a checkpoint by its Platform
URI:

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
{{ param_table(["show", "show_conf", "show_labels"]) }}

## How Scene Text OCR Works

1. **Detect text:** YOLO26 produces axis-aligned boxes, oriented boxes, or segmentation polygons.
2. **Create recognition crops:** Boxes are clipped to the image. OBBs are affine-rectified along the edge that spans
   furthest left-to-right, so near-square boxes stay upright and tilted text lines lie horizontally in reading
   direction. For segmentation polygons, the minimum-area rotated rectangle is rectified while the original polygon is
   retained. Crops at least 1.5× taller than wide are rotated 90° counter-clockwise, as in PaddleOCR.
3. **Recognize text:** PaddleOCR recognizes all crops in batches of 16. Its text-line orientation classifier flags
   crops that look upside down; those are also read after a 180° turn, and the more confident reading is kept. Custom
   recognizers receive one crop at a time, without orientation correction.
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


ocr = solutions.OCR(model="custom-text.pt", recognizer=recognize)
result = ocr(im0)
```

Unsupported backend names raise a `ValueError`. When no text is detected, `result.texts` is empty and the recognizer
is not called.

## OCR Results

Each call returns a [`SolutionResults`](../reference/solutions/solutions.md#ultralytics.solutions.solutions.SolutionResults)
object. Use `result.plot_im` for the annotated image and `result.texts` for records in deterministic reading order:

| Key                    | Type                | Description                                                                           |
| ---------------------- | ------------------- | ------------------------------------------------------------------------------------- |
| `text`                 | `str`               | Recognized text.                                                                      |
| `confidence`           | `float`             | Recognition confidence, kept separate from detection confidence.                      |
| `detection_confidence` | `float`             | YOLO detector confidence.                                                             |
| `detection_index`      | `int`               | Source detection index before reading-order sorting.                                  |
| `polygon`              | `list[list[float]]` | Original-image polygon: four corners for box and OBB detections, or the mask outline. |

## Dependencies and Licenses

| Component         | Required                          | Role                                                     | License                |
| ----------------- | --------------------------------- | -------------------------------------------------------- | ---------------------- |
| Ultralytics       | Yes                               | YOLO detection, crop geometry, and result orchestration  | AGPL-3.0               |
| PaddlePaddle 3.1+ | Only for `recognizer="paddleocr"` | Orientation and recognition runtime                      | Apache-2.0             |
| PaddleOCR 3.7+    | Only for `recognizer="paddleocr"` | Text-line orientation and recognition models and adapter | Apache-2.0             |
| Custom callable   | No                                | User-owned recognition integration                       | Defined by the backend |

PaddleOCR may download orientation and recognition weights on first use. Review the selected model's model card and
redistribution terms before packaging or deploying those artifacts. Custom checkpoint licenses are separate.

## FAQ

### Can I use my own OCR model instead of PaddleOCR?

Yes. Pass a callable as `recognizer` and return `(text, confidence)` for each BGR crop. See
[Use a Custom Recognizer](#use-a-custom-recognizer) for the input and output contract.

### Does OCR read rotated or upside-down text?

Yes. OBB and segmentation detections are cropped upright along the text line. A box alone cannot tell which way its
text reads, so with `recognizer="paddleocr"` a text-line orientation classifier flags crops that look upside down, and
OCR keeps whichever of the original and 180°-turned readings is more confident. Custom recognizers receive the geometric
crop and handle orientation themselves.

### Does OCR track text across video frames?

No. OCR treats every input image independently, without track IDs or cached recognition. Process each video frame as a
separate input when frame-by-frame OCR is required.

### Can I export the complete OCR pipeline?

You can export the YOLO detector with the normal [Export mode](../modes/export.md), but PaddleOCR or a custom recognizer
is **not** embedded in the ONNX, TensorRT, or other detector artifact. Deploy the exported detector and recognition
runtime as separate components, or keep this Python composition layer.
