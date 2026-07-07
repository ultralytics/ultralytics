---
comments: true
description: Detect visual defects without labeled anomalies using Ultralytics YOLOA. Fit a memory bank on normal images, then predict, validate, and fine-tune with YOLO26.
keywords: anomaly detection, defect detection, YOLOA, YOLO26, visual inspection, quality control, one-class learning, memory bank, industrial AI, Ultralytics
---

# YOLOA: YOLO Anomaly Detection

<!-- BLOCKED: hero image pending an assets AVIF once the feature is public -->

[YOLOA](https://www.ultralytics.com/glossary/anomaly-detection) is a YOLO model group (like [YOLOE](yoloe.md) and [YOLO-World](yolo-world.md)) that adds training-free anomaly detection to the standard YOLO detection architecture. After fitting a memory bank on normal images alone, YOLOA fuses a heatmap prior into the detector at inference to find visual defects — scratches, dents, cracks, contamination — without labeled anomalies. Built on [YOLO26](yolo26.md), it outputs standard detection [bounding boxes](https://www.ultralytics.com/glossary/bounding-box) around anomalous regions, so the results plug into any pipeline that already consumes YOLO detections.

Unlike a regular [object detection](../tasks/detect.md) model, which requires labeled examples of every class it should find, YOLOA targets defect types you cannot enumerate in advance: it learns what _normal_ looks like and flags anything that deviates. This one-class approach fits industrial visual inspection, where normal samples are plentiful and defects are rare, diverse, and expensive to label.

!!! tip

    Import the model class with `from ultralytics import YOLOA`. YOLOA uses the standard `detect` task under the hood; model configurations use the `-anomaly` suffix, such as `yolo26n-anomaly.yaml`. The `fit()` step is Python-only and is not a CLI mode.

## Available Models, Supported Tasks, and Operating Modes

| Model Type | Model YAML            | Task Supported               | Inference | Validation | Training | Export |
| ---------- | --------------------- | ---------------------------- | --------- | ---------- | -------- | ------ |
| YOLOA-N    | `yolo26n-anomaly.yaml`  | [Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOA-S    | `yolo26s-anomaly.yaml`  | [Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOA-M    | `yolo26m-anomaly.yaml`  | [Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOA-L    | `yolo26l-anomaly.yaml`  | [Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |
| YOLOA-X    | `yolo26x-anomaly.yaml`  | [Detection](../tasks/detect.md) | ✅        | ✅         | ✅       | ✅     |

Pretrained YOLOA checkpoints and benchmark numbers have not been published yet. Build a model from the `yolo26-anomaly.yaml` configuration, which supports the standard n/s/m/l/x scales via the scale letter in the filename, for example `yolo26n-anomaly.yaml`.

<!-- BLOCKED: uncomment once published, measured numbers exist -->
<!-- {% include "macros/yolo-anomaly-perf.md" %} -->
<!-- Params and FLOPs values are for the fused model after `model.fuse()`, which merges Conv and BatchNorm layers. -->

## Two datasets, two "trainings"

YOLOA separates what most anomaly-detection newcomers conflate: the **normal-image set** consumed by `fit()` and the optional **labeled-defect set** consumed by `train()`.

|             | Dataset A: normal images                       | Dataset B: labeled defects                    |
| ----------- | ---------------------------------------------- | --------------------------------------------- |
| Contents    | Only good images, no labels                    | Defect bounding boxes in standard YOLO format |
| Consumed by | `fit()` → memory bank                          | `train()` → gradient fine-tuning              |
| "Training"  | No gradients, a single feature-extraction pass | Real backpropagation                          |
| Required?   | Required                                       | Optional                                      |

`fit()` is not training: it extracts backbone features from your normal images, compresses them into a memory bank, and calibrates an anomaly threshold — no gradients, no epochs. `train()` is an optional gradient fine-tune on a standard YOLO detection dataset of labeled defects that teaches the detector to convert anomaly evidence into tight boxes.

## Fit

- **Fit the memory bank**: Use a directory of normal (defect-free) images or a list of image paths.
- **Image Processing**: Images are read with the same loader as prediction and resized to 640 pixels internally.
- **Speed**: Training completes in seconds for a few dozen images, even on CPU.
- **Efficiency**: There are no gradients and no epochs required.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOA

        # Build a model from a YAML configuration
        model = YOLOA("yolo26n-anomaly.yaml")

        # Fit the memory bank on normal images and cache it for reuse
        model.fit("path/to/normal/images")

        # Save the fitted model; the memory bank is stored inside the checkpoint
        model.save("yolo26n-anomaly-bottle.pt")
        ```

A fitted checkpoint reloads with its memory bank intact, so `YOLOA("yolo26n-anomaly-bottle.pt")` predicts immediately without re-fitting.

### Key fit arguments

| Argument | Default | Description                                                                                                 |
| -------- | ------- | ----------------------------------------------------------------------------------------------------------- |
| `source` | —       | Directory of normal images or a list of image paths. Required.                                              |
| `batch`  | `8`     | Mini-batch size for feature extraction.                                                                     |
| `imgsz`  | `640`   | Resize images to this size (pixels) before feature extraction.                                              |

Bank-building hyperparameters (bank size 10,000 vectors, 5 nearest neighbors per query, sigmoid temperature 5.0) are baked into the model and are not configurable per call.

### Fit dataset format

The fit set is a plain folder of good images — no labels and no dataset YAML, similar to how [classification](../tasks/classify.md) trains from a folder. Supported extensions: `avif`, `bmp`, `dng`, `heic`, `heif`, `jp2`, `jpeg`, `jpeg2000`, `jpg`, `mpo`, `png`, `tif`, `tiff`, `webp`. See the [Anomaly Detection Dataset Guide](../datasets/anomaly/index.md) for details.

## Train (optional)

Fine-tune the detector on a standard YOLO detection dataset of labeled defects. During training the anomaly prior is rendered automatically from the ground-truth boxes (or polygon masks), augmented, and randomly dropped per sample so the model also performs without a prior.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOA

        model = YOLOA("yolo26n-anomaly.yaml")

        # Train on a standard YOLO detection dataset of labeled defects
        results = model.train(data="defects.yaml", epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        yolo train model=yolo26n-anomaly.yaml data=defects.yaml epochs=100 imgsz=640
        ```

The `data` argument takes a standard [detection dataset YAML](../datasets/detect/index.md) with `train`, `val`, `nc`, and `names` fields. See full `train` mode details in the [Train](../modes/train.md) page.

## Predict

Predict with a fitted model. When the memory bank is non-empty, each image is scored against the bank to produce an anomaly heatmap that is fused into the detector automatically — there is no prior argument to pass. Without a fitted bank, the model runs as a vanilla YOLO26 detector.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOA

        # Load a fitted checkpoint (bank included, no re-fit needed)
        model = YOLOA("yolo26n-anomaly-bottle.pt")

        # Predict on a test image
        results = model.predict("path/to/test/image.jpg")
        for r in results:
            print(r.boxes)  # standard detection boxes around anomalous regions
        ```

    === "CLI"

        ```bash
        yolo predict model=yolo26n-anomaly-bottle.pt source=path/to/test/image.jpg
        ```

Results are standard detection [Results](../reference/engine/results.md) objects with `boxes` populated. See full `predict` mode details in the [Predict](../modes/predict.md) page.

### Results Output

YOLOA returns one `Results` object per image with the same fields as [object detection](../tasks/detect.md) — downstream code written for YOLO detections works unchanged.

| Attribute           | Type            | Shape   | Description                                       |
| ------------------- | --------------- | ------- | ------------------------------------------------- |
| `result.boxes`      | `Boxes`         | `(N,6)` | Bounding boxes around anomalous regions.          |
| `result.boxes.data` | `torch.float32` | `(N,6)` | `x1, y1, x2, y2, confidence, class` for each box. |
| `result.boxes.xyxy` | `torch.float32` | `(N,4)` | Box coordinates in pixels.                        |
| `result.boxes.conf` | `torch.float32` | `(N,)`  | Confidence scores.                                |
| `result.masks`      | -               | -       | No masks.                                         |
| `result.probs`      | -               | -       | No classification probabilities.                  |

### YOLOA vs regular YOLO detection

| Aspect            | Object detection (`YOLO`)              | Anomaly detection (`YOLOA`)                                           |
| ----------------- | -------------------------------------- | --------------------------------------------------------------------- |
| Learns from       | Labeled boxes for every class          | Normal images alone via `fit()`; labeled defects optional (`train()`) |
| Detects           | Only classes seen during training      | Deviations from normal, including unseen defect types                 |
| Gradient training | Required                               | Optional fine-tune                                                    |
| Output field      | `result.boxes`                         | `result.boxes` (same format)                                          |
| Typical use       | General objects: people, cars, animals | Industrial inspection, quality control, defect screening              |

## Val

Validate on a dataset YAML whose `val` split contains labeled defect images. The validator reports two extra columns alongside the standard detection metrics — mAP10 and mAP25, computed at IoU 0.10 and 0.25 for coarse defect localization — populated when a fitted memory bank supplies the heatmap prior.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOA

        model = YOLOA("yolo26n-anomaly.yaml")
        model.fit("path/to/normal/images")

        # Validate with the heatmap prior
        metrics = model.val(data="defects.yaml")
        ```

    === "CLI"

        ```bash
        yolo val model=yolo26n-anomaly-bottle.pt data=defects.yaml
        ```

!!! warning

    `val()` resets the memory bank after it completes. Call `fit()` again (a cached bank reloads instantly) before running `predict()` on the same model instance, or reload the fitted checkpoint.

## Export

Export the model to a format like ONNX. The exported graph is built entirely from ONNX-native operators — no custom ops — with a `(1, 300, 6)` end-to-end output. Exporting a fitted model embeds the memory bank in the graph, so the exported model applies the anomaly prior standalone; exporting before fitting produces a plain detector graph.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLOA

        # Load a fitted checkpoint and export with the memory bank embedded
        model = YOLOA("yolo26n-anomaly-bottle.pt")
        model.export(format="onnx")
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n-anomaly-bottle.pt format=onnx
        ```

See full `export` details in the [Export](../modes/export.md) page.

## FAQ

### Can YOLOA detect defects without any labeled defect images?

Yes — YOLOA fits a memory bank on normal images alone — `model.fit("path/to/normal/images")` extracts backbone features, compresses them into up to 10,000 reference vectors, and calibrates an anomaly threshold without gradients or labels. At prediction time, regions whose features deviate from the bank are flagged as anomalies. Labeled defects are only needed for the optional `train()` fine-tune that sharpens box quality.

### What is the difference between fit() and train() in YOLOA?

`fit()` builds the memory bank from normal images in a single feature-extraction pass — no gradients, no epochs, and no labels. `train()` runs standard gradient fine-tuning on a YOLO detection dataset of labeled defect boxes. `fit()` is required for anomaly-aware inference; `train()` is optional and improves localization when labeled defects are available.

### How does anomaly detection differ from object detection?

[Object detection](../tasks/detect.md) learns to find classes it saw labeled during training, so it misses defect types absent from the training set. Anomaly detection inverts the problem: YOLOA models what normal looks like and flags deviations, so it can detect defect types never seen before. The output format is the same — bounding boxes with confidence scores.

### How do I control the anomaly prior at inference?

The prior is selected automatically: when the model has a fitted, non-empty memory bank, every `predict()` and `val()` call scores the image against the bank and fuses the resulting heatmap into the detector. Without a fitted bank, the model behaves like a vanilla YOLO26 detector. The prior is enabled automatically whenever the bank is fitted — there is no argument to turn it on.

### Can I export a YOLOA model to ONNX?

Yes — `model.export(format="onnx")` produces a graph with only ONNX-native operators and a `(1, 300, 6)` output. Exporting a fitted model embeds the memory bank, so the ONNX model scores images against the stored normal features and applies the anomaly prior on its own — no Python-side fitting at deployment. Exporting an unfitted model yields a plain YOLO26 detector graph.
