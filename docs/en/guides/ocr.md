---
title: YOLO OCR - Scene Text Detection and Recognition with YOLO26
comments: true
description: Run YOLO OCR to detect, read, and order scene and document text at any rotation with Ultralytics YOLO26-Text detectors and PP-OCRv6, or read pre-cropped text lines in batches.
keywords: Ultralytics, YOLO OCR, YOLO26, OCR, scene text, document OCR, text detection, text recognition, TextRecognizer, text line recognition, PP-OCRv6, PP-OCRv5, Korean OCR, Cyrillic OCR, Arabic OCR, ONNX Runtime
---

# Scene Text OCR using Ultralytics YOLO26

## What is Scene Text OCR?

Optical character recognition (OCR) detects and reads text in images. The [OCR solution](../reference/solutions/ocr.md)
combines an Ultralytics YOLO26-Text detector with PaddleOCR's PP-OCRv6 recognizer, run on ONNX Runtime. It reads
photos, documents, and screenshots at any rotation and returns every word with its line in reading order. See
[Supported Languages](#supported-languages) for the scripts PP-OCRv6 can read, and the PaddleOCR language models for
Korean, Cyrillic, Arabic, Devanagari, Thai, and other scripts.

|                                                         Input                                                          |                                                                        OCR result: 113 words in 23 lines                                                                        |
| :--------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| ![Part of a blank IRS Form 8802 with printed field labels](https://cdn.ul.run/i/65193c38435f0505dbbcd7031e102195.avif) | ![The same form with a box and the recognized text drawn over each of the 113 words read by Ultralytics YOLO26 OCR](https://cdn.ul.run/i/39c244f30a97866ac3844bac656ef475.avif) |

`solutions.OCR()` with the default YOLO26x-Text detector and PP-OCRv6 recognizer on part of a blank IRS Form 8802. The
boxes and labels are drawn from `result.texts`.

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

The first run downloads YOLO26x-Text (126 MB) and PP-OCRv6 (77 MB) to the current working directory and installs ONNX
Runtime 1.26 or newer (`onnxruntime-gpu` when CUDA is available). No OCR packages are added to the base Ultralytics
dependencies.

!!! note "Python 3.11 or newer"

    Recognition on ONNX Runtime, the default, requires Python 3.11 or newer. On Python 3.8 to 3.10, pass a
    [custom recognizer](#use-a-custom-recognizer).

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

The CLI saves an annotated `ocr.avi` under the Ultralytics `runs` directory and does not print or save the recognized
text; use Python to work with the [results](#ocr-results).

### YOLO26-Text Checkpoints and Settings

[YOLO26-Text on Ultralytics Platform](https://platform.ultralytics.com/ultralytics-yolo/yolo26-text)

The YOLO26-Text family provides `n`, `s`, `m`, `l`, and `x` OBB checkpoints, which download automatically, for example
`solutions.OCR(model="yolo26s-text.pt")`. The `model` and `imgsz` rows cite the [benchmarks](#benchmarks) below; the
recognition-unit and `max_det` rows come from an earlier test on HierText validation photos, PDF pages scored against
their text layers, and screenshots:

| Setting          | Default        | Benchmark result                                                                                                                                                                                   |
| ---------------- | -------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model`          | `yolo26x-text` | Highest word F1 of the five checkpoints in the [benchmarks](#benchmarks): 77.4 at `imgsz=1280`, with `yolo26l-text` at 76.6.                                                                       |
| `imgsz`          | `1280`         | With `yolo26x-text` in the [benchmarks](#benchmarks), word F1 is 75.1 at 960, 77.4 at 1280, and 78.0 at 1920 on photos of up to 1600 pixels. Smaller images run at their own size, never upscaled. |
| Recognition unit | Word           | Reading words one by one beats reading whole lines by 4 F1 points on photos.                                                                                                                       |
| `max_det`        | `1000`         | A dense document page holds more than the default 300 words.                                                                                                                                       |

To train your own OBB text detector, use the [Scene Text OBB dataset](../datasets/obb/scene-text.md), which combines
COCO-Text and HierText word annotations.

### `OCR` Arguments

Here's a table with the `OCR` arguments:

{% from "macros/solutions-args.md" import param_table %}
{{ param_table(["model", "recognizer", "line_width", "verbose"]) }}

Additionally, the following visualization settings are supported:

{% from "macros/visualization-args.md" import param_table %}
{{ param_table(["show", "show_conf", "show_labels"]) }}

`OCR` also passes the [inference arguments](../modes/predict.md#inference-arguments) `conf`, `iou`, `classes`, `device`,
`imgsz`, `max_det`, and `quantize` to the detector; prediction arguments that are not [solution
arguments](../solutions/index.md#solutions-arguments), such as `agnostic_nms`, raise a `ValueError`. On a CUDA device
the built-in recognizers run on the same GPU as the detector; on other devices they run on the CPU.

## Choose a Model and Recognizer

The [benchmarks](#benchmarks) cover photos with about 100 words each on one GPU and on a 50-photo CPU subset, and their
best settings are candidates to validate on your own images:

| Tested scenario            | Setting                                                                  | Measured                                                                                                      |
| -------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------- |
| Default settings           | Defaults: `yolo26x-text.pt` and `recognizer="paddleocr"` at `imgsz=1280` | 77.4 word F1 in 106.3 ms per image                                                                            |
| TensorRT recognizer        | `recognizer="tensorrt"` on an NVIDIA GPU                                 | 106.3 ms to 57.7 ms per image at the same 77.4 word F1, after a 31-second engine build                        |
| Smallest detector          | `model="yolo26n-text.pt"`                                                | Word F1 from 77.4 to 73.6 and latency from 106.3 ms to 93.6 ms, with a checkpoint of 5.7 MB instead of 126 MB |
| English recognizer on CPU  | `recognizer="en_PP-OCRv5_mobile_rec"`                                    | 1540 ms to 835 ms per image on CPU with YOLO26x-Text; on the GPU run, word F1 from 77.4 to 73.1               |
| Higher detector resolution | `imgsz=1920`                                                             | Word F1 from 77.4 to 78.0 over all 300 photos, 223 of which exceed 1280 pixels                                |

## How Scene Text OCR Works

1. **Detect text:** YOLO26 produces axis-aligned boxes, oriented boxes, or segmentation polygons. A mask is read from
   the minimum-area rectangle around its outline.
2. **Find the page rotation:** The text direction is the mean long-side angle of the detected words, weighted by how
   elongated they are. Words under 2:1, such as single characters, abstain, so a page of only those stays unrotated. On
   a weak sideways vote, boxes shorter than twice the median short side do not vote, so an isolated upright `1` or `I`
   does not turn the page.
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

## Supported Languages

PP-OCRv6 can only output characters in its built-in 18,708-character set. The detector finds words in any script, but
text in a script outside that set comes back as lookalike characters from the set. For example, Cyrillic `ТЕКСТ` reads
as `TEKCT`, often with high confidence. Coverage of each language's full alphabet, upper and lower case:

| Script        | Languages                                                                                                                                                                       | Coverage                                                                           |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| Chinese (CJK) | Simplified and Traditional Chinese                                                                                                                                              | ✅ About 15,600 CJK characters                                                     |
| Japanese      | Japanese                                                                                                                                                                        | ✅ Hiragana, katakana, and kanji                                                   |
| Latin         | English, French, German, Spanish, Portuguese, Italian, Dutch, Swedish, Finnish, Danish, Norwegian, Polish, Czech, Hungarian, Romanian, Turkish, Croatian, Indonesian, and Malay | ✅ Every letter and accented letter except German capital `ẞ` and Spanish `¡`      |
| Latin         | Vietnamese                                                                                                                                                                      | ⚠️ Partial: most toned vowels, such as `ạ`, `ế`, `ở`, and `ữ`, are missing         |
| Greek         | Greek                                                                                                                                                                           | ⚠️ Partial: base letters only; accented letters and final `ς` are missing          |
| Cyrillic      | Russian, Ukrainian, Belarusian, Bulgarian, Serbian, Kazakh, and others                                                                                                          | ❌ Not in PP-OCRv6; use a [language model](#read-other-scripts)                    |
| Other         | Korean (Hangul), Arabic, Hebrew, Hindi and other Devanagari, Thai                                                                                                               | ❌ Not in PP-OCRv6; use a [language model](#read-other-scripts), except for Hebrew |

Digits, ASCII punctuation, and common symbols such as `€`, `£`, `¥`, and `°` are included.

### Read Other Scripts

Pass a PaddleOCR PP-OCRv5 language model name as `recognizer` to read a script PP-OCRv6 does not cover. Each model runs
on ONNX Runtime like PP-OCRv6, downloads from the Ultralytics assets release on first use, and needs no PaddlePaddle
install:

```python
ocr = solutions.OCR(recognizer="korean_PP-OCRv5_mobile_rec")
result = ocr(im0)
```

| `recognizer`                     | Script and example languages                                        |
| -------------------------------- | ------------------------------------------------------------------- |
| `korean_PP-OCRv5_mobile_rec`     | Korean (Hangul) and Latin letters                                   |
| `cyrillic_PP-OCRv5_mobile_rec`   | Cyrillic: Russian, Ukrainian, Bulgarian, Serbian, Kazakh, Mongolian |
| `eslav_PP-OCRv5_mobile_rec`      | East Slavic Cyrillic: Russian, Ukrainian, Belarusian                |
| `arabic_PP-OCRv5_mobile_rec`     | Arabic script: Arabic, Persian, Urdu, Uyghur                        |
| `devanagari_PP-OCRv5_mobile_rec` | Devanagari: Hindi, Marathi, Nepali                                  |
| `th_PP-OCRv5_mobile_rec`         | Thai                                                                |
| `el_PP-OCRv5_mobile_rec`         | Greek, including accented letters and final `ς`                     |
| `ta_PP-OCRv5_mobile_rec`         | Tamil                                                               |
| `te_PP-OCRv5_mobile_rec`         | Telugu                                                              |
| `latin_PP-OCRv5_mobile_rec`      | Latin-script languages, in a smaller and faster model than PP-OCRv6 |
| `en_PP-OCRv5_mobile_rec`         | English, in a smaller and faster model than PP-OCRv6                |

Pick the model for your text; OCR does not detect the language or switch models. No PP-OCRv5 model reads Hebrew, so
pass a [custom recognizer](#use-a-custom-recognizer) for it. See
[PaddleOCR's text recognition models](https://www.paddleocr.ai/latest/en/version3.x/module_usage/text_recognition.html)
for each model's accuracy and speed.

## Read Text Inside Your Own Detections

Pass any detection, OBB, or segmentation model as `model` to read the text inside its boxes, such as license plates or
serial numbers. OCR rectifies each detection and reads it as one line of text, so you do not crop anything yourself.
Each detection should hold a single line of text that runs along the long side of its box. OCR's `imgsz=1280` default
applies to your detector too, so pass the size it was trained at:

```python
import cv2

from ultralytics import solutions

ocr = solutions.OCR(model="path/to/license-plate-detector.pt", classes=[0], imgsz=640)  # the detector's training size
im0 = cv2.imread("path/to/image.jpg")
result = ocr(im0)
print([text["text"] for text in result.texts])
```

The
[number plate recognition notebook](https://github.com/ultralytics/notebooks/blob/main/notebooks/how-to-use-ultralytics-yolo-ocr-for-number-plate-recognition.ipynb)
runs this setup on a video.

## Read Pre-Cropped Text Lines in Batches

When another system already crops the text, such as label lines, form fields, or serial numbers, call `TextRecognizer`
to read the crops with PP-OCRv6 and skip detection. It takes a list of BGR `numpy.ndarray` crops, each holding one line
of text, and returns one `(text, confidence)` tuple per crop, in input order:

```python
import cv2

from ultralytics.solutions.ocr import TextRecognizer

recognize = TextRecognizer()  # PP-OCRv6 on ONNX Runtime, on a CUDA GPU when available, else CPU
crops = [cv2.imread(f) for f in ["path/to/line1.png", "path/to/line2.png", "path/to/line3.png"]]
for text, confidence in recognize(crops, batch=8):
    print(f"{text} ({confidence:.2f})")
```

The recognizer resizes each crop to 48 pixels high, sorts the crops by width, and reads them `batch` at a time (8 by
default), so crops of similar width share a batch. Pass a model from [Read Other Scripts](#read-other-scripts) with its
`.onnx` ending, such as `TextRecognizer("korean_PP-OCRv5_mobile_rec.onnx")`, to read another script.
`TextRecognizer(device=0, engine=True)` reads with the same FP16 TensorRT engine as `recognizer="tensorrt"`, and like
`OCR` it needs Python 3.11 or newer when it reads on ONNX Runtime.

!!! warning "Turn crops upright first"

    `TextRecognizer` does not check orientation as `OCR` does. It reads a crop at least 1.5 times taller than wide turned
    90° counterclockwise, which suits text that runs top to bottom, and an upside-down crop returns wrong text.

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

The confidence your callable returns decides the 0° or 180° turn, so return the model's score rather than a constant. A
`recognizer` that is neither a name nor a callable raises a `TypeError`, and an unknown name fails when its `.onnx` file
is not found. When no text is detected, `result.texts` is empty and the recognizer is not called.

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

`result.texts` holds one record per word. Group the records by `line` to get line strings, filter on `confidence` to
drop uncertain reads, and serialize the list directly as JSON:

```python
import json
from itertools import groupby

words = [text for text in result.texts if text["confidence"] >= 0.8]
lines = [" ".join(text["text"] for text in group) for _, group in groupby(words, key=lambda text: text["line"])]
print("\n".join(lines))
print(json.dumps(words))
```

On the [benchmark](#benchmarks) photos, keeping words with a recognition confidence of at least 0.8 raised word
precision from 77.7% to 86.8% and lowered recall from 77.2% to 76.1% with the default settings. Lines are ordered by
baseline across the whole page, so a multi-column page is returned row by row rather than column by column.

## Benchmarks

The benchmarks ran Ultralytics 8.4.216 on one **NVIDIA RTX PRO 6000 Blackwell Server Edition** GPU with ONNX Runtime
1.32 and TensorRT 11.5, on 300 photos sampled evenly from the
[HierText](https://github.com/google-research-datasets/hiertext) validation split, whose training split is part of the
[Scene Text OBB dataset](../datasets/obb/scene-text.md) that the YOLO26-Text detectors were trained on. The photos have
a long side of 480 to 1600 pixels and return 107 words each on average with the default settings. Accuracy is word F1 in
percent against the 29,543 legible ground-truth words: words are lowercased, reduced to letters and digits, kept when at
least two characters long, and matched within each image regardless of position. Latency is the mean time in
milliseconds of the full `ocr(im0)` call for one already decoded image, including the annotated image.

=== "Accuracy by Image Size"

    Word F1 with `recognizer="paddleocr"`, higher is better. The best value for each model is in bold. Photos smaller
    than `imgsz` run at their own size rounded up to a multiple of 32, so `imgsz=1920` ran these photos at up to 1600
    pixels.

    | Model        | `imgsz=640` | `imgsz=960` | `imgsz=1280` | `imgsz=1920` |
    | ------------ | ----------- | ----------- | ------------ | ------------ |
    | YOLO26n-Text | 66.5        | 71.7        | 73.6         | **74.2**     |
    | YOLO26s-Text | 69.6        | 73.3        | 75.2         | **76.6**     |
    | YOLO26m-Text | 71.6        | 74.7        | 76.5         | **77.5**     |
    | YOLO26l-Text | 71.9        | 74.8        | 76.6         | **77.6**     |
    | YOLO26x-Text | 72.7        | 75.1        | 77.4         | **78.0**     |

=== "Accuracy by recognizer"

    Word F1 at `imgsz=1280`, higher is better. The best value for each model is in bold.

    | Model        | `paddleocr` | `tensorrt` | `en_PP-OCRv5_mobile_rec` | `latin_PP-OCRv5_mobile_rec` |
    | ------------ | ----------- | ---------- | ------------------------ | --------------------------- |
    | YOLO26n-Text | **73.6**    | **73.6**   | 69.8                     | 69.9                        |
    | YOLO26s-Text | **75.2**    | **75.2**   | 71.5                     | 71.5                        |
    | YOLO26m-Text | **76.5**    | **76.5**   | 72.9                     | 72.4                        |
    | YOLO26l-Text | **76.6**    | **76.6**   | 72.5                     | 72.2                        |
    | YOLO26x-Text | **77.4**    | **77.4**   | 73.1                     | 73.0                        |

=== "Latency by recognizer"

    Milliseconds per image at `imgsz=1280`, lower is better: the median of three passes for each photo, averaged over
    the 300 photos. The detector column is the YOLO26-Text prediction step alone, with `recognizer="paddleocr"`.

    | Model        | Detector only | `paddleocr` | `tensorrt` | `en_PP-OCRv5_mobile_rec` | `latin_PP-OCRv5_mobile_rec` |
    | ------------ | ------------- | ----------- | ---------- | ------------------------ | --------------------------- |
    | YOLO26n-Text | 6.9           | 93.6        | 47.7       | 72.1                     | 73.3                        |
    | YOLO26s-Text | 7.1           | 95.8        | 47.8       | 73.4                     | 74.9                        |
    | YOLO26m-Text | 8.8           | 95.9        | 49.5       | 74.3                     | 76.5                        |
    | YOLO26l-Text | 10.6          | 99.6        | 52.4       | 77.6                     | 79.2                        |
    | YOLO26x-Text | 16.4          | 106.3       | 57.7       | 84.0                     | 85.3                        |

=== "Latency by word count"

    Milliseconds per image at `imgsz=1280` with `recognizer="paddleocr"`, lower is better. Photos are grouped by the
    number of words YOLO26x-Text returned: 81 photos with 32 words on average, 152 photos with 87, and 67 photos with
    242.

    | Model        | 50 or fewer words | 51 to 150 words | more than 150 words |
    | ------------ | ----------------- | --------------- | ------------------- |
    | YOLO26n-Text | 53.7              | 86.3            | 158.4               |
    | YOLO26s-Text | 53.9              | 88.6            | 162.6               |
    | YOLO26m-Text | 54.6              | 88.7            | 162.2               |
    | YOLO26l-Text | 58.7              | 92.5            | 164.9               |
    | YOLO26x-Text | 64.7              | 99.7            | 171.6               |

=== "CPU latency"

    Milliseconds per image at `imgsz=1280`, lower is better, on every sixth photo of the benchmark: 50 photos with 104
    words on average, one pass each, on an AMD EPYC 9655 host with the default 8 CPU threads. The GPU columns cover the
    same 50 photos. The English recognizer was measured on CPU for the smallest and the largest model.

    | Model        | CPU, `paddleocr` | CPU, `en_PP-OCRv5_mobile_rec` | GPU, `paddleocr` | GPU, `tensorrt` |
    | ------------ | ---------------- | ----------------------------- | ---------------- | --------------- |
    | YOLO26n-Text | 1048             | 437                           | 89.3             | 44.9            |
    | YOLO26s-Text | 1092             | -                             | 89.6             | 45.2            |
    | YOLO26m-Text | 1244             | -                             | 90.4             | 48.2            |
    | YOLO26l-Text | 1302             | -                             | 93.7             | 49.9            |
    | YOLO26x-Text | 1540             | 835                           | 103.4            | 57.2            |

- **Word count** set the latency more than the model did: YOLO26x-Text took 64.7 ms on photos with 50 or fewer words and
  171.6 ms on photos with more than 150, while the detector alone took 6.9 to 16.4 ms.
- **`recognizer="tensorrt"`** matched ONNX Runtime's word F1 to one decimal for all five models. With YOLO26x-Text it
  read 322 of 32,022 words differently from ONNX Runtime, and 160 of 32,019 differently between two runs with separately
  built engines; ONNX Runtime repeated all 32,022.

## Dependencies and Licenses

| Component       | Required                                              | Role                                                         | License                |
| --------------- | ----------------------------------------------------- | ------------------------------------------------------------ | ---------------------- |
| Ultralytics     | Yes                                                   | YOLO detection, crop geometry, and result orchestration      | AGPL-3.0               |
| ONNX Runtime    | For all but callables and `"tensorrt"` on NVIDIA GPUs | Recognition runtime, 1.26+ on Python 3.11+                   | MIT                    |
| PP-OCRv6 medium | For `"paddleocr"` and `"tensorrt"`                    | Recognition weights from the Ultralytics assets release      | Apache-2.0             |
| PP-OCRv5 mobile | For a PP-OCRv5 language model name                    | Language recognition weights from the Ultralytics assets     | Apache-2.0             |
| TensorRT        | Only for `recognizer="tensorrt"`                      | FP16 PP-OCRv6 engine on NVIDIA GPUs, instead of ONNX Runtime | NVIDIA proprietary     |
| Custom callable | No                                                    | User-owned recognition integration                           | Defined by the backend |

PP-OCRv6 and PP-OCRv5 weights download on first use. Review the model card, such as
[PP-OCRv6 medium](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec) or
[Korean PP-OCRv5 mobile](https://huggingface.co/PaddlePaddle/korean_PP-OCRv5_mobile_rec), and redistribution terms
before packaging or deploying them. Each ONNX model names its source card in its `source` metadata. Custom checkpoint
licenses are separate.

## FAQ

### Can I use my own OCR model instead of PP-OCRv6?

Yes. Pass a callable as `recognizer` and return `(text, confidence)` for each BGR crop. See
[Use a Custom Recognizer](#use-a-custom-recognizer) for the input and output contract.

### Can OCR read Cyrillic, Korean, or Arabic text?

Yes, with a language model. The default PP-OCRv6 covers Chinese, Japanese, and Latin-script languages, so pass
`recognizer="cyrillic_PP-OCRv5_mobile_rec"`, `"korean_PP-OCRv5_mobile_rec"`, or `"arabic_PP-OCRv5_mobile_rec"`.
YOLO26-Text detects words in any script. See [Read Other Scripts](#read-other-scripts).

### Does OCR read rotated or upside-down text?

Yes, at any rotation. OCR finds the page's text direction from the word boxes, crops words upright in that frame, and
reads the widest words both ways to settle 0° or 180°, so pages turned 90° or upside down return the same text in the
same reading order. See [How Scene Text OCR Works](#how-scene-text-ocr-works).

### Does OCR track text across video frames?

No. OCR treats every input image independently, without track IDs or cached recognition. Process each video frame as a
separate input when frame-by-frame OCR is required.

### Can I run PP-OCRv6 recognition with TensorRT?

Yes, on NVIDIA GPUs: `solutions.OCR(recognizer="tensorrt")` lowered the full call from 106.3 ms to 57.7 ms per image
with YOLO26x-Text in the [benchmarks](#benchmarks), at the same word F1 to one decimal and with 322 of 32,022 words read
differently from ONNX Runtime. The first run builds an FP16 TensorRT engine (about 30 seconds) and caches it next to the
PP-OCRv6 model for each TensorRT version and GPU architecture. The default `recognizer="paddleocr"` builds nothing and
starts in about a second. Without an NVIDIA GPU, `"tensorrt"` runs PP-OCRv6 on ONNX Runtime.

### Which YOLO26-Text model and recognizer should I use for OCR?

Start with the defaults, YOLO26x-Text with PP-OCRv6. On an NVIDIA GPU, pass `recognizer="tensorrt"` for lower latency at
the same word F1, and use YOLO26n-Text when the download size matters more than accuracy.
[Choose a Model and Recognizer](#choose-a-model-and-recognizer) gives the measured trade-offs.

### How do I read license plates with Ultralytics YOLO26 OCR?

Pass your license plate detector as `model`: `solutions.OCR(model="path/to/license-plate-detector.pt")` reads the text
inside every detected box and returns it in `result.texts`, with no manual cropping. See
[Read Text Inside Your Own Detections](#read-text-inside-your-own-detections).

### Can I export the complete OCR pipeline?

You can export the YOLO detector with the normal [Export mode](../modes/export.md), but the PP-OCRv6 or custom
recognizer is **not** embedded in the ONNX, TensorRT, or other detector artifact. Deploy the exported detector and recognition
runtime as separate components, or keep this Python composition layer.
