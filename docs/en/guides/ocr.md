---
title: Scene Text OCR with YOLO26
comments: true
description: Detect, recognize, and order scene and document text at any rotation with Ultralytics YOLO26-Text detectors and PP-OCRv6 recognition.
keywords: Ultralytics, YOLO26, OCR, scene text, document OCR, text detection, text recognition, PP-OCRv6, ONNX Runtime
---

# Scene Text OCR using Ultralytics YOLO26

## What is Scene Text OCR?

Optical character recognition (OCR) detects and reads text in images. The [OCR solution](../reference/solutions/ocr.md)
combines an Ultralytics YOLO26-Text detector with PaddleOCR's PP-OCRv6 recognizer, run on ONNX Runtime. It reads
photos, documents, and screenshots at any rotation and returns every word with its line in reading order.

Detection and recognition remain separate components, so you can pair any text detector with PP-OCRv6 or your own
recognition model.

## Why Use Scene Text OCR?

- **Any rotation:** Pages turned 90° or upside down read in the right order without preprocessing.
- **Flexible text geometry:** Use detection, OBB, or segmentation models for horizontal, rotated, or irregular text.
- **No extra framework:** PP-OCRv6 runs on ONNX Runtime from the Ultralytics assets release, on CPU or CUDA, with no
  PaddlePaddle install.
- **Structured output:** Access recognized text, recognition and detection confidences, source geometry, line numbers,
  and an annotated image from one results object.

## Run Scene Text OCR with YOLO26

The first run downloads YOLO26x-Text and PP-OCRv6 and installs ONNX Runtime (`onnxruntime-gpu` when CUDA is available).
No OCR packages are added to the base Ultralytics dependencies.

!!! example "Scene Text OCR using Ultralytics YOLO26"

    === "CLI"

        ```bash
        yolo solutions ocr source="path/to/video.mp4"
        ```

    === "Python"

        ```python
        import cv2

        from ultralytics import solutions

        ocr = solutions.OCR()  # YOLO26x-Text at imgsz=1280 with PP-OCRv6
        im0 = cv2.imread("path/to/image.jpg")
        assert im0 is not None, "Error reading image file"
        result = ocr(im0)
        cv2.imwrite("ocr.jpg", result.plot_im)

        for text in result.texts:
            print(text["line"], text["text"], text["confidence"])
        ```

### YOLO26-Text Checkpoints and Settings

[YOLO26-Text on Ultralytics Platform](https://platform.ultralytics.com/ultralytics-yolo/yolo26-text)

The YOLO26-Text family provides `n`, `s`, `m`, `l`, and `x` OBB checkpoints, which download automatically, for example
`solutions.OCR(model="yolo26s-text.pt")`. The defaults come from a benchmark of 80 HierText validation photos with
their transcriptions, PDF pages scored against their text layers, and screenshots:

| Setting          | Default        | Benchmark result                                                                                                 |
| ---------------- | -------------- | ---------------------------------------------------------------------------------------------------------------- |
| `model`          | `yolo26x-text` | Best word F1 on photos and documents; `yolo26l-text` is within 0.5 points.                                       |
| `imgsz`          | `1280`         | 960 loses 6 points of recall on photos; 1920 adds nothing. Smaller images run at their own size, never upscaled. |
| Recognition unit | Word           | Reading words one by one beats reading whole lines by 4 F1 points on photos.                                     |
| `max_det`        | `1000`         | A dense document page holds more than the default 300 words.                                                     |

To train your own OBB text detector, use the [Scene Text OBB dataset](../datasets/obb/scene-text.md), which combines
COCO-Text and HierText word annotations.

### `OCR` Arguments

Here's a table with the `OCR` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "recognizer", "line_width", "verbose"]) }}

Additionally, the following visualization settings are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "show_conf", "show_labels"]) }}

## How Scene Text OCR Works

1. **Detect text:** YOLO26 produces axis-aligned boxes, oriented boxes, or segmentation polygons. A mask is read from
   the minimum-area rectangle around its outline.
2. **Find the page rotation:** The text direction is the mean long-side angle of the detected words, weighted by how
   elongated they are. Words under 2:1, such as single characters, abstain, so a page of only those stays unrotated.
3. **Crop words upright:** The page is rotated so its text runs left to right, and each word is cropped and rectified
   there. Words are grouped into lines: same baseline within half the smaller text height, split at gaps wider than 1.5
   text heights.
4. **Settle 0° or 180°:** A box cannot tell which way its text reads, so OCR reads the 8 widest words of the page both
   ways and keeps the turn that reads more confidently, since upside-down text reads as low-confidence noise.
5. **Recognize words:** PP-OCRv6 reads every word, in width-sorted batches of 8 at 48 px high, padded to at least 320 px
   as PaddleOCR does. Words that read as nothing are dropped from the results.
6. **Order and render results:** Words come back line by line, top to bottom and left to right in the page's own frame.
   The visualization uses a Unicode font for non-ASCII text such as Chinese.

Coordinates always refer to the original image, not the detector input, rotated page, or recognition crop.

## Use a Custom Recognizer

Pass a callable to integrate another recognition model without changing the detection, rotation, geometry, or ordering
path. The callable runs once for each nonempty BGR `numpy.ndarray` word crop, including the crops read both ways to settle
the page's turn. Return one `(text, confidence)` tuple per call, not a
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
| `line`                 | `int`               | Line number in reading order; words of one line share it.                             |
| `polygon`              | `list[list[float]]` | Original-image polygon: four corners for box and OBB detections, or the mask outline. |

## Dependencies and Licenses

| Component       | Required                           | Role                                                         | License                |
| --------------- | ---------------------------------- | ------------------------------------------------------------ | ---------------------- |
| Ultralytics     | Yes                                | YOLO detection, crop geometry, and result orchestration      | AGPL-3.0               |
| ONNX Runtime    | Only for `recognizer="paddleocr"`  | PP-OCRv6 recognition runtime, 1.26+ on Python 3.11+          | MIT                    |
| PP-OCRv6 medium | For `"paddleocr"` and `"tensorrt"` | Recognition weights from the Ultralytics assets release      | Apache-2.0             |
| TensorRT        | Only for `recognizer="tensorrt"`   | FP16 PP-OCRv6 engine on NVIDIA GPUs, instead of ONNX Runtime | NVIDIA proprietary     |
| Custom callable | No                                 | User-owned recognition integration                           | Defined by the backend |

PP-OCRv6 weights download on first use. Review the [model card](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec)
and redistribution terms before packaging or deploying them. Custom checkpoint licenses are separate.

## FAQ

### Can I use my own OCR model instead of PP-OCRv6?

Yes. Pass a callable as `recognizer` and return `(text, confidence)` for each BGR crop. See
[Use a Custom Recognizer](#use-a-custom-recognizer) for the input and output contract.

### Does OCR read rotated or upside-down text?

Yes, at any rotation. OCR finds the page's text direction from the word boxes, crops words upright in that frame, and
reads the widest words both ways to settle 0° or 180°, so pages turned 90° or upside down return the same text in the
same reading order. See [How Scene Text OCR Works](#how-scene-text-ocr-works).

### Does OCR track text across video frames?

No. OCR treats every input image independently, without track IDs or cached recognition. Process each video frame as a
separate input when frame-by-frame OCR is required.

### Can I run PP-OCRv6 recognition with TensorRT?

Yes, on NVIDIA GPUs: `solutions.OCR(recognizer="tensorrt")` reads the same text about 2.5x faster than ONNX Runtime on
dense pages. The first run builds an FP16 TensorRT engine (about 30 seconds) and caches it next to the PP-OCRv6 model
for each TensorRT version and GPU architecture. The default `recognizer="paddleocr"` builds nothing and starts in about
a second.

### Can I export the complete OCR pipeline?

You can export the YOLO detector with the normal [Export mode](../modes/export.md), but the PP-OCRv6 or custom
recognizer is **not** embedded in the ONNX, TensorRT, or other detector artifact. Deploy the exported detector and recognition
runtime as separate components, or keep this Python composition layer.
