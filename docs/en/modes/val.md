---
comments: true
description: Learn how to validate your YOLO11 model with precise metrics, easy-to-use tools, and custom settings for optimal performance.
keywords: Ultralytics, YOLO11, model validation, machine learning, object detection, mAP metrics, Python API, CLI
---

# Model Validation with Ultralytics YOLO

<img width="1024" src="https://github.com/ultralytics/docs/releases/download/0/ultralytics-yolov8-ecosystem-integrations.avif" alt="Ultralytics YOLO ecosystem and integrations">

## Introduction

Validation is a critical step in the [machine learning](https://www.ultralytics.com/glossary/machine-learning-ml) pipeline, allowing you to assess the quality of your trained models. Val mode in Ultralytics YOLO11 provides a robust suite of tools and metrics for evaluating the performance of your [object detection](https://www.ultralytics.com/glossary/object-detection) models. This guide serves as a complete resource for understanding how to effectively use the Val mode to ensure that your models are both accurate and reliable.

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

Here's why using YOLO11's Val mode is advantageous:

- **Precision:** Get accurate metrics like mAP50, mAP75, and mAP50-95 to comprehensively evaluate your model.
- **Convenience:** Utilize built-in features that remember training settings, simplifying the validation process.
- **Flexibility:** Validate your model with the same or different datasets and image sizes.
- **[Hyperparameter Tuning](https://www.ultralytics.com/glossary/hyperparameter-tuning):** Use validation metrics to fine-tune your model for better performance.

### Key Features of Val Mode

These are the notable functionalities offered by YOLO11's Val mode:

- **Automated Settings:** Models remember their training configurations for straightforward validation.
- **Multi-Metric Support:** Evaluate your model based on a range of accuracy metrics.
- **CLI and Python API:** Choose from command-line interface or Python API based on your preference for validation.
- **Data Compatibility:** Works seamlessly with datasets used during the training phase as well as custom datasets.

!!! tip

    * YOLO11 models automatically remember their training settings, so you can validate a model at the same image size and on the original dataset easily with just `yolo val model=yolo11n.pt` or `YOLO("yolo11n.pt").val()`

## Usage Examples

Validate a trained YOLO11n model [accuracy](https://www.ultralytics.com/glossary/accuracy) on the COCO8 dataset. No arguments are needed as the `model` retains its training `data` and arguments as model attributes. See the Arguments section below for a full list of validation arguments.

!!! warning "Windows Multi-Processing Error"

    On Windows, you may receive a `RuntimeError` when launching the validation as a script. Add an `if __name__ == "__main__":` block before your validation code to resolve it.

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a model
        model = YOLO("yolo11n.pt")  # load an official model
        model = YOLO("path/to/best.pt")  # load a custom model

        # Validate the model
        metrics = model.val()  # no arguments needed, dataset and settings remembered
        metrics.box.map  # map50-95
        metrics.box.map50  # map50
        metrics.box.map75  # map75
        metrics.box.maps  # a list containing mAP50-95 for each category
        ```

    === "CLI"

        ```bash
        yolo detect val model=yolo11n.pt      # val official model
        yolo detect val model=path/to/best.pt # val custom model
        ```

## Arguments for YOLO Model Validation

When validating YOLO models, several arguments can be fine-tuned to optimize the evaluation process. These arguments control aspects such as input image size, batch processing, and performance thresholds. Below is a detailed breakdown of each argument to help you customize your validation settings effectively.

{% include "macros/validation-args.md" %}

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
        model = YOLO("yolo11n.pt")

        # Customize validation settings
        metrics = model.val(data="coco8.yaml", imgsz=640, batch=16, conf=0.25, iou=0.6, device="0")
        ```

    === "CLI"

        ```bash
        yolo val model=yolo11n.pt data=coco8.yaml imgsz=640 batch=16 conf=0.25 iou=0.6 device=0
        ```

!!! tip "Export ConfusionMatrix"

    You can also save the ConfusionMatrix results in different formats using the provided code.

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    results = model.val(data="coco8.yaml", plots=True)
    print(results.confusion_matrix.to_df())
    ```

| Method      | Return Type            | Description                                                                |
| ----------- | ---------------------- | -------------------------------------------------------------------------- |
| `summary()` | `List[Dict[str, Any]]` | Converts validation results to a summarized dictionary.                    |
| `to_df()`   | `DataFrame`            | Returns the validation results as a structured Polars DataFrame.           |
| `to_csv()`  | `str`                  | Exports the validation results in CSV format and returns the CSV string.   |
| `to_json()` | `str`                  | Exports the validation results in JSON format and returns the JSON string. |

For more details see the [`DataExportMixin` class documentation](../reference/utils/__init__.md/#ultralytics.utils.DataExportMixin).

## FAQ

### How do I validate my YOLO11 model with Ultralytics?

To validate your YOLO11 model, you can use the Val mode provided by Ultralytics. For example, using the Python API, you can load a model and run validation with:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Validate the model
metrics = model.val()
print(metrics.box.map)  # map50-95
```

Alternatively, you can use the command-line interface (CLI):

```bash
yolo val model=yolo11n.pt
```

For further customization, you can adjust various arguments like `imgsz`, `batch`, and `conf` in both Python and CLI modes. Check the [Arguments for YOLO Model Validation](#arguments-for-yolo-model-validation) section for the full list of parameters.

### What metrics can I get from YOLO11 model validation?

YOLO11 model validation provides several key metrics to assess model performance. These include:

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
```

For a complete performance evaluation, it's crucial to review all these metrics. For more details, refer to the [Key Features of Val Mode](#key-features-of-val-mode).

### What are the advantages of using Ultralytics YOLO for validation?

Using Ultralytics YOLO for validation provides several advantages:

- **[Precision](https://www.ultralytics.com/glossary/precision):** YOLO11 offers accurate performance metrics including mAP50, mAP75, and mAP50-95.
- **Convenience:** The models remember their training settings, making validation straightforward.
- **Flexibility:** You can validate against the same or different datasets and image sizes.
- **Hyperparameter Tuning:** Validation metrics help in fine-tuning models for better performance.

These benefits ensure that your models are evaluated thoroughly and can be optimized for superior results. Learn more about these advantages in the [Why Validate with Ultralytics YOLO](#why-validate-with-ultralytics-yolo) section.

### Can I validate my YOLO11 model using a custom dataset?

Yes, you can validate your YOLO11 model using a [custom dataset](https://docs.ultralytics.com/datasets/). Specify the `data` argument with the path to your dataset configuration file. This file should include the path to the [validation data](https://www.ultralytics.com/glossary/validation-data).

!!! note

    Validation is performed using the model's own class names, which you can view using `model.names`, and which may be different to those specified in the dataset configuration file.

Example in Python:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Validate with a custom dataset
metrics = model.val(data="path/to/your/custom_dataset.yaml")
print(metrics.box.map)  # map50-95
```

Example using CLI:

```bash
yolo val model=yolo11n.pt data=path/to/your/custom_dataset.yaml
```

For more customizable options during validation, see the [Example Validation with Arguments](#example-validation-with-arguments) section.

### How do I save validation results to a JSON file in YOLO11?

To save the validation results to a JSON file, you can set the `save_json` argument to `True` when running validation. This can be done in both the Python API and CLI.

Example in Python:

```python
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")

# Save validation results to JSON
metrics = model.val(save_json=True)
```

Example using CLI:

```bash
yolo val model=yolo11n.pt save_json=True
```

This functionality is particularly useful for further analysis or integration with other tools. Check the [Arguments for YOLO Model Validation](#arguments-for-yolo-model-validation) for more details.
