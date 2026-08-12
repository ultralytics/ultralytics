---
comments: true
description: Learn how to validate your YOLO26 model with precise metrics, easy-to-use tools, and custom settings for optimal performance.
keywords: Ultralytics, YOLO26, model validation, machine learning, object detection, mAP metrics, Python API, CLI
---

# Model Validation with Ultralytics YOLO

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Validation is a critical step in the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) pipeline, allowing you to assess the quality of your trained models. Val mode in Ultralytics YOLO26 provides a robust suite of tools and metrics for evaluating the performance of your [object detection](https://www.ultralytics.com/glossary/object-detection) models. This guide serves as a complete resource for understanding how to effectively use the Val mode to ensure that your models are both accurate and reliable.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?start=47"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics Modes Tutorial: Validation
</p>

## Why Validate with Ultralytics YOLO?

Here's why using YOLO26's Val mode is advantageous:

- **Precision:** Get accurate metrics like mAP50, mAP75, and mAP50-95 to comprehensively evaluate your model.
- **Convenience:** Utilize built-in features that remember training settings, simplifying the validation process.
- **Flexibility:** Validate your model with the same or different datasets and image sizes.
- **[Hyperparameter Tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning):** Use validation metrics to fine-tune your model for better performance.

### Key Features of Val Mode

These are the notable functionalities offered by YOLO26's Val mode:

- **Automated Settings:** Models remember their training configurations for straightforward validation.
- **Multi-Metric Support:** Evaluate your model based on a range of accuracy metrics.
- **CLI and Python API:** Choose from command-line interface or Python API based on your preference for validation.
- **Data Compatibility:** Works seamlessly with datasets used during the training phase as well as custom datasets.

!!! tip

    * YOLO26 models automatically remember their training settings, so you can validate a model at the same image size and on the original dataset easily with just `yolo val model=yolo26n.pt` or `YOLO("yolo26n.pt").val()`

## Usage Examples

Validate a trained YOLO26n model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes. See the Arguments section below for a full list of validation arguments.

!!! warning "Windows Multi-Processing Error"

    On Windows, you may receive a `RuntimeError` when launching the validation as a script. Add an `if __name__ == "__main__":` block before your validation code to resolve it.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list containing mAP50-95 for each category
        metrics.box.image_metrics  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo26n.pt      # val official model
        yolo detect val model=path/to/best.pt # val custom model
        ```

## Arguments for YOLO Model Validation

When validating YOLO models, several arguments can be fine-tuned to optimize the evaluation process. These arguments control aspects such as input image size, batch processing, and performance thresholds. Below is a detailed breakdown of each argument to help you customize your validation settings effectively.

| Argument        | Type            | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------- | --------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `data`          | `str`           | `None`  | Specifies the path to the dataset YAML (e.g., `coco8.yaml`), which should include the path to the [validation data](https://www.ultralytics.com/glossary/validation-data). Classification instead takes a dataset directory or a built-in dataset name (e.g., `imagenet10`).                                                                                                                                                                                              |
| `imgsz`         | `int`           | `640`   | Defines the size of input images. All images are resized to this dimension before processing. Larger sizes may improve accuracy for small objects but increase computation time.                                                                                                                                                                                                                                                                                          |
| `batch`         | `int`           | `16`    | Sets the number of images per batch. Higher values utilize GPU memory more efficiently but require more VRAM. Adjust based on available hardware resources.                                                                                                                                                                                                                                                                                                               |
| `save_json`     | `bool`          | `False` | If `True`, saves the results to a JSON file for further analysis, integration with other tools, or submission to evaluation servers like COCO.                                                                                                                                                                                                                                                                                                                            |
| `conf`          | `float`         | `0.001` | Sets the minimum confidence threshold for detections. Lower values increase recall but may introduce more false positives. Precision-recall curves default to `0.001`; detection confusion matrices use an explicit `conf` value, or `0.25` when it is omitted. Summary precision and recall use the max-F1 confidence, so they may differ from values derived from `confusion_matrix.png`. Defaults to `0.01` for OBB validation to reduce memory usage.                 |
| `iou`           | `float`         | `0.7`   | Sets the [Intersection Over Union](https://www.ultralytics.com/glossary/intersection-over-union-iou) threshold for [Non-Maximum Suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms). Controls duplicate detection elimination.                                                                                                                                                                                                                 |
| `max_det`       | `int`           | `300`   | Limits the maximum number of detections per image. Useful in dense scenes to prevent excessive detections and manage computational resources.                                                                                                                                                                                                                                                                                                                             |
| `quantize`      | `int` or `str`  | `None`  | Validation precision: `16`/`"fp16"` enables FP16 validation on supported GPUs; `32`/`"fp32"`/unset is FP32. INT8/PTQ quantization is configured during [export](../modes/export.md#quantization-options), then used by validating the exported model. Replaces the deprecated `half` flag.                                                                                                                                                                                |
| `device`        | `str`           | `None`  | Specifies the device for validation (`cpu`, `cuda:0`, `npu`, `npu:0`, etc.). When `None`, automatically selects the best available device. Multiple CUDA devices can be specified with comma separation.                                                                                                                                                                                                                                                                  |
| `dnn`           | `bool`          | `False` | If `True`, uses the [OpenCV](https://www.ultralytics.com/glossary/opencv) DNN module for ONNX model inference, offering an alternative to [PyTorch](https://www.ultralytics.com/glossary/pytorch) inference methods.                                                                                                                                                                                                                                                      |
| `plots`         | `bool`          | `True`  | When set to `True`, generates and saves plots of predictions versus ground truth, confusion matrices, and PR curves for visual evaluation of model performance.                                                                                                                                                                                                                                                                                                           |
| `classes`       | `list[int]`     | `None`  | Specifies a list of class IDs to evaluate. Useful for filtering out and focusing only on certain classes during evaluation.                                                                                                                                                                                                                                                                                                                                               |
| `rect`          | `bool`          | `True`  | If `True`, uses rectangular inference for batching, reducing padding and potentially increasing speed and efficiency by processing images in their original aspect ratio. Ignored for `depth` validation, which stretches every image to a fixed `imgsz` square instead of padding.                                                                                                                                                                                       |
| `split`         | `str`           | `'val'` | Determines the dataset split to use for validation (`val`, `test`, or `train`). Allows flexibility in choosing the data segment for performance evaluation.                                                                                                                                                                                                                                                                                                               |
| `project`       | `str`           | `None`  | Name of the project directory where validation outputs are saved. Helps organize results from different experiments or models.                                                                                                                                                                                                                                                                                                                                            |
| `name`          | `str`           | `None`  | Name of the validation run. Used for creating a subdirectory within the project folder, where validation logs and outputs are stored.                                                                                                                                                                                                                                                                                                                                     |
| `verbose`       | `bool`          | `True`  | If `True`, displays detailed information during the validation process, including per-class metrics, batch progress, and additional debugging information.                                                                                                                                                                                                                                                                                                                |
| `save_txt`      | `bool`          | `False` | If `True`, saves detection results in text files, with one file per image, useful for further analysis, custom post-processing, or integration with other systems.                                                                                                                                                                                                                                                                                                        |
| `save_conf`     | `bool`          | `False` | If `True`, includes confidence values in the saved text files when `save_txt` is enabled, providing more detailed output for analysis and filtering.                                                                                                                                                                                                                                                                                                                      |
| `workers`       | `int`           | `8`     | Number of worker threads for data loading. Higher values can speed up data preprocessing but may increase CPU usage. Setting to 0 uses main thread, which can be more stable in some environments.                                                                                                                                                                                                                                                                        |
| `augment`       | `bool`          | `False` | Enables test-time augmentation (TTA) during validation, potentially improving detection accuracy at the cost of inference speed by running inference on transformed versions of the input. Only available for Ultralytics PyTorch models.                                                                                                                                                                                                                                 |
| `agnostic_nms`  | `bool`          | `False` | Enables class-agnostic [Non-Maximum Suppression](https://www.ultralytics.com/glossary/non-maximum-suppression-nms), suppressing lower-scoring overlapping boxes regardless of their predicted class. Useful for instance-focused applications. For end-to-end models (YOLO26, YOLOv10), this only prevents the same detection from appearing with multiple class labels (IoU=1.0 duplicates) and does not perform IoU-threshold-based suppression between distinct boxes. |
| `single_cls`    | `bool`          | `False` | Treats all classes as a single class during validation. Useful for evaluating model performance on binary detection tasks or when class distinctions aren't important.                                                                                                                                                                                                                                                                                                    |
| `visualize`     | `bool`          | `False` | Visualizes the ground truths, true positives, false positives, and false negatives for each image. Useful for debugging and model interpretation.                                                                                                                                                                                                                                                                                                                         |
| `show_labels`   | `bool`          | `True`  | Displays class labels in validation visualizations when `visualize=True`. Set to `False` for a cleaner view of matches and errors.                                                                                                                                                                                                                                                                                                                                        |
| `show_conf`     | `bool`          | `True`  | Displays confidence scores in validation visualizations when `visualize=True`. Set to `False` for a cleaner view of matches and errors.                                                                                                                                                                                                                                                                                                                                   |
| `compile`       | `bool` or `str` | `False` | Enables PyTorch 2.x `torch.compile` graph compilation with `backend='inductor'`. Accepts `True` → `"default"`, `False` → disables, or a string mode such as `"default"`, `"reduce-overhead"`, `"max-autotune-no-cudagraphs"`. Falls back to eager with a warning if unsupported.                                                                                                                                                                                          |
| `channels_last` | `bool`          | `False` | Uses the channels_last (NHWC) memory format for convolutions during validation, speeding up CUDA Tensor Core GPUs with no change in results. Applies to native PyTorch models only; ignored for CPU, MPS, and exported formats such as TensorRT and ONNX.                                                                                                                                                                                                                 |
| `end2end`       | `bool`          | `None`  | Overrides the end-to-end mode in YOLO models that support NMS-free inference (YOLO26, YOLOv10). Setting it to `False` lets you run validation using the traditional NMS pipeline, additionally allowing you to make use of the `iou` argument.                                                                                                                                                                                                                            |

Each of these settings plays a vital role in the validation process, allowing for a customizable and efficient evaluation of YOLO models. Adjusting these parameters according to your specific needs and resources can help achieve the best balance between accuracy and performance.

### Example Validation with Arguments

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/zHxwDkYShNc"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> How to Export Model Validation Results in CSV, JSON, SQL, Polars DataFrame & More
</p>

<a href="https://github.com/ultralytics/notebooks/blob/main/notebooks/how-to-export-the-validation-results-into-dataframe-csv-sql-and-other-formats.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Explore model validation and different export methods in Google Colab"></a>

The below examples showcase YOLO model validation with custom arguments in Python and CLI.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo26n.pt")

        # Customize validation settings
        metrics = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.7, device="0")
        ```

    === "CLI"

        ```bash
        yolo val model=yolo26n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.7 device=0
        ```

!!! tip "Export ConfusionMatrix"

    You can also save the ConfusionMatrix results in different formats using the provided code.

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo26n.pt")

    results = model.val(data="coco8.yaml", plots=True)
    print(results.confusion_matrix.to_df())
    ```

!!! tip "Per-Image Precision, Recall, and F1"

    Validation stores per-image precision, recall, F1, TP, FP, and FN metrics (at IoU threshold 0.5) for all tasks
    except classification. Access them through `results.box.image_metrics` for detection and OBB, `results.seg.image_metrics`
    for segmentation, and `results.pose.image_metrics` for pose after validation completes.

    ```python
    from ultralytics import YOLO

    # Load a model
    model = YOLO("yolo26n.pt")

    # Validate and access per-image metrics
    results = model.val(data="coco8.yaml")

    # image_metrics is a dictionary with image filenames as keys
    print(results.box.image_metrics)
    # Output: {'image1.jpg': {'precision': 0.85, 'recall': 0.92, 'f1': 0.88, 'tp': 17, 'fp': 3, 'fn': 1}, ...}

    # Access metrics for a specific image
    results.box.image_metrics["image1.jpg"]  # {'precision': 0.85, 'recall': 0.92, 'f1': 0.88, 'tp': 17, 'fp': 3, 'fn': 1}
    ```

    Each entry in `image_metrics` contains the following keys:

    | Key         | Description                                       |
    |-------------|---------------------------------------------------|
    | `precision` | Precision score for the image (`tp / (tp + fp)`). |
    | `recall`    | Recall score for the image (`tp / (tp + fn)`).    |
    | `f1`        | Harmonic mean of precision and recall.            |
    | `tp`        | Number of true positives for the image.           |
    | `fp`        | Number of false positives for the image.          |
    | `fn`        | Number of false negatives for the image.          |

    This feature is available for detection, segmentation, pose, and OBB tasks.

| Method      | Return Type            | Description                                                                |
| ----------- | ---------------------- | -------------------------------------------------------------------------- |
| `summary()` | `List[Dict[str, Any]]` | Converts validation results to a summarized dictionary.                    |
| `to_df()`   | `DataFrame`            | Returns the validation results as a structured Polars DataFrame.           |
| `to_csv()`  | `str`                  | Exports the validation results in CSV format and returns the CSV string.   |
| `to_json()` | `str`                  | Exports the validation results in JSON format and returns the JSON string. |

For more details see the [`DataExportMixin` class documentation](../reference/utils/__init__.md#ultralytics.utils.__init__.DataExportMixin).

## What's Next

Happy with the metrics? [Export the model](export.md) to a deployment format. If accuracy is off, [go back and retrain](train.md) with different [hyperparameters](../guides/hyperparameter-tuning.md) or more training data.

## FAQ

### How do I validate my YOLO26 model with Ultralytics?

To validate your YOLO26 model, you can use the Val mode provided by Ultralytics. For example, using the Python API, you can load a model and run validation with:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # map50-95
```

Alternatively, you can use the command-line interface (CLI):

```bash
yolo val model=yolo26n.pt
```

For further customization, you can adjust various arguments like `imgsz`, `batch`, and `conf` in both Python and CLI modes. Check the [Arguments for YOLO Model Validation](#arguments-for-yolo-model-validation) section for the full list of parameters.

### What metrics can I get from YOLO26 model validation?

YOLO26 model validation provides several key metrics to assess model performance. These include:

- mAP50 (mean Average Precision at IoU threshold 0.5)
- mAP75 (mean Average Precision at IoU threshold 0.75)
- mAP50-95 (mean Average Precision across multiple IoU thresholds from 0.5 to 0.95)

Using the Python API, you can access these metrics as follows:

```python
metrics = model.val()  # assumes `model` has been loaded
print(metrics.box.map)  # mAP50-95
print(metrics.box.map50)  # mAP50
print(metrics.box.map75)  # mAP75
print(metrics.box.maps)  # list of mAP50-95 for each category
print(metrics.box.image_metrics)  # per-image metrics dictionary with precision, recall, F1, TP, FP, and FN
```

For a complete performance evaluation, it's crucial to review all these metrics. For more details, refer to the [Key Features of Val Mode](#key-features-of-val-mode).

### What are the advantages of using Ultralytics YOLO for validation?

Using Ultralytics YOLO for validation provides several advantages:

- **[Precision](https://www.ultralytics.com/glossary/precision):** YOLO26 offers accurate performance metrics including mAP50, mAP75, and mAP50-95.
- **Convenience:** The models remember their training settings, making validation straightforward.
- **Flexibility:** You can validate against the same or different datasets and image sizes.
- **Hyperparameter Tuning:** Validation metrics help in fine-tuning models for better performance.

These benefits ensure that your models are evaluated thoroughly and can be optimized for superior results. Learn more about these advantages in the [Why Validate with Ultralytics YOLO](#why-validate-with-ultralytics-yolo) section.

### Can I validate my YOLO26 model using a custom dataset?

Yes, you can validate your YOLO26 model using a [custom dataset](../datasets/index.md). Specify the `data` argument with the path to your dataset YAML, which should include the path to the [validation data](https://www.ultralytics.com/glossary/validation-data). Classification instead takes a dataset directory or a built-in dataset name (e.g., `imagenet10`).

!!! note

    Validation is performed using the model's own class names, which you can view using `model.names`, and which may be different to those the dataset itself defines.

Example in Python:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Validate with a custom dataset
metrics = model.val(data="path/to/your/custom_dataset.yaml")
print(metrics.box.map)  # map50-95
```

Example using CLI:

```bash
yolo val model=yolo26n.pt data=path/to/your/custom_dataset.yaml
```

For more customizable options during validation, see the [Example Validation with Arguments](#example-validation-with-arguments) section.

### How do I save validation results to a JSON file in YOLO26?

To save the validation results to a JSON file, you can set the `save_json` argument to `True` when running validation. This can be done in both the Python API and CLI.

Example in Python:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n.pt")

# Save validation results to JSON
metrics = model.val(save_json=True)
```

Example using CLI:

```bash
yolo val model=yolo26n.pt save_json=True
```

This functionality is particularly useful for further analysis or integration with other tools. Check the [Arguments for YOLO Model Validation](#arguments-for-yolo-model-validation) for more details.
