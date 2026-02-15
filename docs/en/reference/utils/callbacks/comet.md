---
description: Explore the integration of Comet callbacks in Ultralytics YOLO, enabling advanced logging and monitoring for your machine learning experiments.
keywords: Ultralytics, YOLO, Comet, callbacks, logging, machine learning, monitoring, integration
---

# Reference for `ultralytics/utils/callbacks/comet.py`

!!! success "Improvements"

    This page is sourced from [https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py). Have an improvement or example to add? Open a [Pull Request](https://docs.ultralytics.com/help/contributing/) — thank you! 🙏

<br>

!!! abstract "Summary"

    === "<span class="doc-kind doc-kind-function">Functions</span>"

        - [`_get_comet_mode`](#ultralytics.utils.callbacks.comet._get_comet_mode)
        - [`_get_comet_model_name`](#ultralytics.utils.callbacks.comet._get_comet_model_name)
        - [`_get_eval_batch_logging_interval`](#ultralytics.utils.callbacks.comet._get_eval_batch_logging_interval)
        - [`_get_max_image_predictions_to_log`](#ultralytics.utils.callbacks.comet._get_max_image_predictions_to_log)
        - [`_scale_confidence_score`](#ultralytics.utils.callbacks.comet._scale_confidence_score)
        - [`_should_log_confusion_matrix`](#ultralytics.utils.callbacks.comet._should_log_confusion_matrix)
        - [`_should_log_image_predictions`](#ultralytics.utils.callbacks.comet._should_log_image_predictions)
        - [`_resume_or_create_experiment`](#ultralytics.utils.callbacks.comet._resume_or_create_experiment)
        - [`_fetch_trainer_metadata`](#ultralytics.utils.callbacks.comet._fetch_trainer_metadata)
        - [`_scale_bounding_box_to_original_image_shape`](#ultralytics.utils.callbacks.comet._scale_bounding_box_to_original_image_shape)
        - [`_format_ground_truth_annotations_for_detection`](#ultralytics.utils.callbacks.comet._format_ground_truth_annotations_for_detection)
        - [`_format_prediction_annotations`](#ultralytics.utils.callbacks.comet._format_prediction_annotations)
        - [`_extract_segmentation_annotation`](#ultralytics.utils.callbacks.comet._extract_segmentation_annotation)
        - [`_fetch_annotations`](#ultralytics.utils.callbacks.comet._fetch_annotations)
        - [`_create_prediction_metadata_map`](#ultralytics.utils.callbacks.comet._create_prediction_metadata_map)
        - [`_log_confusion_matrix`](#ultralytics.utils.callbacks.comet._log_confusion_matrix)
        - [`_log_images`](#ultralytics.utils.callbacks.comet._log_images)
        - [`_log_image_predictions`](#ultralytics.utils.callbacks.comet._log_image_predictions)
        - [`_log_plots`](#ultralytics.utils.callbacks.comet._log_plots)
        - [`_log_model`](#ultralytics.utils.callbacks.comet._log_model)
        - [`_log_image_batches`](#ultralytics.utils.callbacks.comet._log_image_batches)
        - [`_log_asset`](#ultralytics.utils.callbacks.comet._log_asset)
        - [`_log_table`](#ultralytics.utils.callbacks.comet._log_table)
        - [`on_pretrain_routine_start`](#ultralytics.utils.callbacks.comet.on_pretrain_routine_start)
        - [`on_train_epoch_end`](#ultralytics.utils.callbacks.comet.on_train_epoch_end)
        - [`on_fit_epoch_end`](#ultralytics.utils.callbacks.comet.on_fit_epoch_end)
        - [`on_train_end`](#ultralytics.utils.callbacks.comet.on_train_end)


## Function `ultralytics.utils.callbacks.comet._get_comet_mode` {#ultralytics.utils.callbacks.comet.\_get\_comet\_mode}

```python
def _get_comet_mode() -> str
```

Return the Comet mode from environment variables, defaulting to 'online'.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L44-L56"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_comet_mode() -> str:
    """Return the Comet mode from environment variables, defaulting to 'online'."""
    comet_mode = os.getenv("COMET_MODE")
    if comet_mode is not None:
        LOGGER.warning(
            "The COMET_MODE environment variable is deprecated. "
            "Please use COMET_START_ONLINE to set the Comet experiment mode. "
            "To start an offline Comet experiment, use 'export COMET_START_ONLINE=0'. "
            "If COMET_START_ONLINE is not set or is set to '1', an online Comet experiment will be created."
        )
        return comet_mode

    return "online"
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._get_comet_model_name` {#ultralytics.utils.callbacks.comet.\_get\_comet\_model\_name}

```python
def _get_comet_model_name() -> str
```

Return the Comet model name from environment variable or default to 'Ultralytics'.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L59-L61"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_comet_model_name() -> str:
    """Return the Comet model name from environment variable or default to 'Ultralytics'."""
    return os.getenv("COMET_MODEL_NAME", "Ultralytics")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._get_eval_batch_logging_interval` {#ultralytics.utils.callbacks.comet.\_get\_eval\_batch\_logging\_interval}

```python
def _get_eval_batch_logging_interval() -> int
```

Get the evaluation batch logging interval from environment variable or use default value 1.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L64-L66"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_eval_batch_logging_interval() -> int:
    """Get the evaluation batch logging interval from environment variable or use default value 1."""
    return int(os.getenv("COMET_EVAL_BATCH_LOGGING_INTERVAL", 1))
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._get_max_image_predictions_to_log` {#ultralytics.utils.callbacks.comet.\_get\_max\_image\_predictions\_to\_log}

```python
def _get_max_image_predictions_to_log() -> int
```

Get the maximum number of image predictions to log from environment variables.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L69-L71"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _get_max_image_predictions_to_log() -> int:
    """Get the maximum number of image predictions to log from environment variables."""
    return int(os.getenv("COMET_MAX_IMAGE_PREDICTIONS", 100))
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._scale_confidence_score` {#ultralytics.utils.callbacks.comet.\_scale\_confidence\_score}

```python
def _scale_confidence_score(score: float) -> float
```

Scale the confidence score by a factor specified in environment variable.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `score` | `float` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L74-L77"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _scale_confidence_score(score: float) -> float:
    """Scale the confidence score by a factor specified in environment variable."""
    scale = float(os.getenv("COMET_MAX_CONFIDENCE_SCORE", 100.0))
    return score * scale
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._should_log_confusion_matrix` {#ultralytics.utils.callbacks.comet.\_should\_log\_confusion\_matrix}

```python
def _should_log_confusion_matrix() -> bool
```

Determine if the confusion matrix should be logged based on environment variable settings.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L80-L82"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _should_log_confusion_matrix() -> bool:
    """Determine if the confusion matrix should be logged based on environment variable settings."""
    return os.getenv("COMET_EVAL_LOG_CONFUSION_MATRIX", "false").lower() == "true"
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._should_log_image_predictions` {#ultralytics.utils.callbacks.comet.\_should\_log\_image\_predictions}

```python
def _should_log_image_predictions() -> bool
```

Determine whether to log image predictions based on environment variable.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L85-L87"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _should_log_image_predictions() -> bool:
    """Determine whether to log image predictions based on environment variable."""
    return os.getenv("COMET_EVAL_LOG_IMAGE_PREDICTIONS", "true").lower() == "true"
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._resume_or_create_experiment` {#ultralytics.utils.callbacks.comet.\_resume\_or\_create\_experiment}

```python
def _resume_or_create_experiment(args: SimpleNamespace) -> None
```

Resume CometML experiment or create a new experiment based on args.

Ensures that the experiment object is only created in a single process during distributed training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `args` | `SimpleNamespace` | Training arguments containing project configuration and other parameters. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L90-L122"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _resume_or_create_experiment(args: SimpleNamespace) -> None:
    """Resume CometML experiment or create a new experiment based on args.

    Ensures that the experiment object is only created in a single process during distributed training.

    Args:
        args (SimpleNamespace): Training arguments containing project configuration and other parameters.
    """
    if RANK not in {-1, 0}:
        return

    # Set environment variable (if not set by the user) to configure the Comet experiment's online mode under the hood.
    # IF COMET_START_ONLINE is set by the user it will override COMET_MODE value.
    if os.getenv("COMET_START_ONLINE") is None:
        comet_mode = _get_comet_mode()
        os.environ["COMET_START_ONLINE"] = "1" if comet_mode != "offline" else "0"

    try:
        _project_name = os.getenv("COMET_PROJECT_NAME", args.project)
        experiment = comet_ml.start(project_name=_project_name)
        experiment.log_parameters(vars(args))
        experiment.log_others(
            {
                "eval_batch_logging_interval": _get_eval_batch_logging_interval(),
                "log_confusion_matrix_on_eval": _should_log_confusion_matrix(),
                "log_image_predictions": _should_log_image_predictions(),
                "max_image_predictions": _get_max_image_predictions_to_log(),
            }
        )
        experiment.log_other("Created from", "ultralytics")

    except Exception as e:
        LOGGER.warning(f"Comet installed but not initialized correctly, not logging this run. {e}")
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._fetch_trainer_metadata` {#ultralytics.utils.callbacks.comet.\_fetch\_trainer\_metadata}

```python
def _fetch_trainer_metadata(trainer) -> dict
```

Return metadata for YOLO training including epoch and asset saving status.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The YOLO trainer object containing training state and config. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `dict` | Dictionary containing current epoch, step, save assets flag, and final epoch flag. |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L125-L145"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _fetch_trainer_metadata(trainer) -> dict:
    """Return metadata for YOLO training including epoch and asset saving status.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The YOLO trainer object containing training state and config.

    Returns:
        (dict): Dictionary containing current epoch, step, save assets flag, and final epoch flag.
    """
    curr_epoch = trainer.epoch + 1

    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs

    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and not final_epoch

    return dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=save_assets, final_epoch=final_epoch)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._scale_bounding_box_to_original_image_shape` {#ultralytics.utils.callbacks.comet.\_scale\_bounding\_box\_to\_original\_image\_shape}

```python
def _scale_bounding_box_to_original_image_shape(
    box, resized_image_shape, original_image_shape, ratio_pad
) -> list[float]
```

Scale bounding box from resized image coordinates to original image coordinates.

YOLO resizes images during training and the label values are normalized based on this resized shape. This function rescales the bounding box labels to the original image shape.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `box` | `torch.Tensor` | Bounding box in normalized xywh format. | *required* |
| `resized_image_shape` | `tuple` | Shape of the resized image (height, width). | *required* |
| `original_image_shape` | `tuple` | Shape of the original image (height, width). | *required* |
| `ratio_pad` | `tuple` | Ratio and padding information for scaling. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[float]` | Scaled bounding box coordinates in xywh format with top-left corner adjustment. |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L148-L177"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _scale_bounding_box_to_original_image_shape(
    box, resized_image_shape, original_image_shape, ratio_pad
) -> list[float]:
    """Scale bounding box from resized image coordinates to original image coordinates.

    YOLO resizes images during training and the label values are normalized based on this resized shape. This function
    rescales the bounding box labels to the original image shape.

    Args:
        box (torch.Tensor): Bounding box in normalized xywh format.
        resized_image_shape (tuple): Shape of the resized image (height, width).
        original_image_shape (tuple): Shape of the original image (height, width).
        ratio_pad (tuple): Ratio and padding information for scaling.

    Returns:
        (list[float]): Scaled bounding box coordinates in xywh format with top-left corner adjustment.
    """
    resized_image_height, resized_image_width = resized_image_shape

    # Convert normalized xywh format predictions to xyxy in resized scale format
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    # Scale box predictions from resized image scale back to original image scale
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    # Convert bounding box format from xyxy to xywh for Comet logging
    box = ops.xyxy2xywh(box)
    # Adjust xy center to correspond top-left corner
    box[:2] -= box[2:] / 2
    box = box.tolist()

    return box
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._format_ground_truth_annotations_for_detection` {#ultralytics.utils.callbacks.comet.\_format\_ground\_truth\_annotations\_for\_detection}

```python
def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map = None) -> dict | None
```

Format ground truth annotations for object detection.

This function processes ground truth annotations from a batch of images for object detection tasks. It extracts bounding boxes, class labels, and other metadata for a specific image in the batch, and formats them for visualization or evaluation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_idx` | `int` | Index of the image in the batch to process. | *required* |
| `image_path` | `str | Path` | Path to the image file. | *required* |
| `batch` | `dict` | Batch dictionary containing detection data with keys:<br>    - 'batch_idx': Tensor of batch indices<br>    - 'bboxes': Tensor of bounding boxes in normalized xywh format<br>    - 'cls': Tensor of class labels<br>    - 'ori_shape': Original image shapes<br>    - 'resized_shape': Resized image shapes<br>    - 'ratio_pad': Ratio and padding information | *required* |
| `class_name_map` | `dict, optional` | Mapping from class indices to class names. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict | None` | Formatted ground truth annotations with keys 'name' and 'data', where 'data' is a list of |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L180-L229"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None) -> dict | None:
    """Format ground truth annotations for object detection.

    This function processes ground truth annotations from a batch of images for object detection tasks. It extracts
    bounding boxes, class labels, and other metadata for a specific image in the batch, and formats them for
    visualization or evaluation.

    Args:
        img_idx (int): Index of the image in the batch to process.
        image_path (str | Path): Path to the image file.
        batch (dict): Batch dictionary containing detection data with keys:
            - 'batch_idx': Tensor of batch indices
            - 'bboxes': Tensor of bounding boxes in normalized xywh format
            - 'cls': Tensor of class labels
            - 'ori_shape': Original image shapes
            - 'resized_shape': Resized image shapes
            - 'ratio_pad': Ratio and padding information
        class_name_map (dict, optional): Mapping from class indices to class names.

    Returns:
        (dict | None): Formatted ground truth annotations with keys 'name' and 'data', where 'data' is a list of
            annotation dicts each containing 'boxes', 'label', and 'score' keys. Returns None if no bounding boxes are
            found for the image.
    """
    indices = batch["batch_idx"] == img_idx
    bboxes = batch["bboxes"][indices]
    if len(bboxes) == 0:
        LOGGER.debug(f"Comet Image: {image_path} has no bounding boxes labels")
        return None

    cls_labels = batch["cls"][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]

    original_image_shape = batch["ori_shape"][img_idx]
    resized_image_shape = batch["resized_shape"][img_idx]
    ratio_pad = batch["ratio_pad"][img_idx]

    data = []
    for box, label in zip(bboxes, cls_labels):
        box = _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)
        data.append(
            {
                "boxes": [box],
                "label": f"gt_{label}",
                "score": _scale_confidence_score(1.0),
            }
        )

    return {"name": "ground_truth", "data": data}
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._format_prediction_annotations` {#ultralytics.utils.callbacks.comet.\_format\_prediction\_annotations}

```python
def _format_prediction_annotations(image_path, metadata, class_label_map = None, class_map = None) -> dict | None
```

Format YOLO predictions for object detection visualization.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `image_path` | `Path` | Path to the image file. | *required* |
| `metadata` | `dict` | Prediction metadata containing bounding boxes and class information. | *required* |
| `class_label_map` | `dict, optional` | Mapping from class indices to class names. | `None` |
| `class_map` | `dict, optional` | Additional class mapping for label conversion. | `None` |

**Returns**

| Type | Description |
| --- | --- |
| `dict | None` | Formatted prediction annotations or None if no predictions exist. |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L232-L281"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _format_prediction_annotations(image_path, metadata, class_label_map=None, class_map=None) -> dict | None:
    """Format YOLO predictions for object detection visualization.

    Args:
        image_path (Path): Path to the image file.
        metadata (dict): Prediction metadata containing bounding boxes and class information.
        class_label_map (dict, optional): Mapping from class indices to class names.
        class_map (dict, optional): Additional class mapping for label conversion.

    Returns:
        (dict | None): Formatted prediction annotations or None if no predictions exist.
    """
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem

    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(f"Comet Image: {image_path} has no bounding boxes predictions")
        return None

    # apply the mapping that was used to map the predicted classes when the JSON was created
    if class_label_map and class_map:
        class_label_map = {class_map[k]: v for k, v in class_label_map.items()}
    try:
        # import pycotools utilities to decompress annotations for various tasks, e.g. segmentation
        from faster_coco_eval.core.mask import decode
    except ImportError:
        decode = None

    data = []
    for prediction in predictions:
        boxes = prediction["bbox"]
        score = _scale_confidence_score(prediction["score"])
        cls_label = prediction["category_id"]
        if class_label_map:
            cls_label = str(class_label_map[cls_label])

        annotation_data = {"boxes": [boxes], "label": cls_label, "score": score}

        if decode is not None:
            # do segmentation processing only if we are able to decode it
            segments = prediction.get("segmentation", None)
            if segments is not None:
                segments = _extract_segmentation_annotation(segments, decode)
            if segments is not None:
                annotation_data["points"] = segments

        data.append(annotation_data)

    return {"name": "prediction", "data": data}
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._extract_segmentation_annotation` {#ultralytics.utils.callbacks.comet.\_extract\_segmentation\_annotation}

```python
def _extract_segmentation_annotation(segmentation_raw: str, decode: Callable) -> list[list[Any]] | None
```

Extract segmentation annotation from compressed segmentations as list of polygons.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `segmentation_raw` | `str` | Raw segmentation data in compressed format. | *required* |
| `decode` | `Callable` | Function to decode the compressed segmentation data. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list[list[Any]] | None` | List of polygon points or None if extraction fails. |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L284-L301"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _extract_segmentation_annotation(segmentation_raw: str, decode: Callable) -> list[list[Any]] | None:
    """Extract segmentation annotation from compressed segmentations as list of polygons.

    Args:
        segmentation_raw (str): Raw segmentation data in compressed format.
        decode (Callable): Function to decode the compressed segmentation data.

    Returns:
        (list[list[Any]] | None): List of polygon points or None if extraction fails.
    """
    try:
        mask = decode(segmentation_raw)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        annotations = [np.array(polygon).squeeze() for polygon in contours if len(polygon) >= 3]
        return [annotation.ravel().tolist() for annotation in annotations]
    except Exception as e:
        LOGGER.warning(f"Comet Failed to extract segmentation annotation: {e}")
    return None
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._fetch_annotations` {#ultralytics.utils.callbacks.comet.\_fetch\_annotations}

```python
def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map, class_map) -> list | None
```

Join the ground truth and prediction annotations if they exist.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `img_idx` | `int` | Index of the image in the batch. | *required* |
| `image_path` | `Path` | Path to the image file. | *required* |
| `batch` | `dict` | Batch data containing ground truth annotations. | *required* |
| `prediction_metadata_map` | `dict` | Map of prediction metadata by image ID. | *required* |
| `class_label_map` | `dict` | Mapping from class indices to class names. | *required* |
| `class_map` | `dict` | Additional class mapping for label conversion. | *required* |

**Returns**

| Type | Description |
| --- | --- |
| `list | None` | List of annotation dictionaries or None if no annotations exist. |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L304-L328"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map, class_map) -> list | None:
    """Join the ground truth and prediction annotations if they exist.

    Args:
        img_idx (int): Index of the image in the batch.
        image_path (Path): Path to the image file.
        batch (dict): Batch data containing ground truth annotations.
        prediction_metadata_map (dict): Map of prediction metadata by image ID.
        class_label_map (dict): Mapping from class indices to class names.
        class_map (dict): Additional class mapping for label conversion.

    Returns:
        (list | None): List of annotation dictionaries or None if no annotations exist.
    """
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(
        img_idx, image_path, batch, class_label_map
    )
    prediction_annotations = _format_prediction_annotations(
        image_path, prediction_metadata_map, class_label_map, class_map
    )

    annotations = [
        annotation for annotation in [ground_truth_annotations, prediction_annotations] if annotation is not None
    ]
    return [annotations] if annotations else None
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._create_prediction_metadata_map` {#ultralytics.utils.callbacks.comet.\_create\_prediction\_metadata\_map}

```python
def _create_prediction_metadata_map(model_predictions) -> dict
```

Create metadata map for model predictions by grouping them based on image ID.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `model_predictions` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L331-L338"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _create_prediction_metadata_map(model_predictions) -> dict:
    """Create metadata map for model predictions by grouping them based on image ID."""
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction["image_id"], [])
        pred_metadata_map[prediction["image_id"]].append(prediction)

    return pred_metadata_map
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_confusion_matrix` {#ultralytics.utils.callbacks.comet.\_log\_confusion\_matrix}

```python
def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch) -> None
```

Log the confusion matrix to Comet experiment.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` |  |  | *required* |
| `trainer` |  |  | *required* |
| `curr_step` |  |  | *required* |
| `curr_epoch` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L341-L347"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch) -> None:
    """Log the confusion matrix to Comet experiment."""
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = [*list(trainer.data["names"].values()), "background"]
    experiment.log_confusion_matrix(
        matrix=conf_mat, labels=names, max_categories=len(names), epoch=curr_epoch, step=curr_step
    )
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_images` {#ultralytics.utils.callbacks.comet.\_log\_images}

```python
def _log_images(experiment, image_paths, curr_step: int | None, annotations = None) -> None
```

Log images to the experiment with optional annotations.

This function logs images to a Comet ML experiment, optionally including annotation data for visualization such as bounding boxes or segmentation masks.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` | `comet_ml.CometExperiment` | The Comet ML experiment to log images to. | *required* |
| `image_paths` | `list[Path]` | List of paths to images that will be logged. | *required* |
| `curr_step` | `int | None` | Current training step/iteration for tracking in the experiment timeline. | *required* |
| `annotations` | `list[list[dict]], optional` | Nested list of annotation dictionaries for each image. Each annotation<br>    contains visualization data like bounding boxes, labels, and confidence scores. | `None` |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L350-L369"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_images(experiment, image_paths, curr_step: int | None, annotations=None) -> None:
    """Log images to the experiment with optional annotations.

    This function logs images to a Comet ML experiment, optionally including annotation data for visualization such as
    bounding boxes or segmentation masks.

    Args:
        experiment (comet_ml.CometExperiment): The Comet ML experiment to log images to.
        image_paths (list[Path]): List of paths to images that will be logged.
        curr_step (int | None): Current training step/iteration for tracking in the experiment timeline.
        annotations (list[list[dict]], optional): Nested list of annotation dictionaries for each image. Each annotation
            contains visualization data like bounding boxes, labels, and confidence scores.
    """
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)

    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_image_predictions` {#ultralytics.utils.callbacks.comet.\_log\_image\_predictions}

```python
def _log_image_predictions(experiment, validator, curr_step) -> None
```

Log image predictions to a Comet ML experiment during model validation.

This function processes validation data and formats both ground truth and prediction annotations for visualization in the Comet dashboard. The function respects configured limits on the number of images to log.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` | `comet_ml.CometExperiment` | The Comet ML experiment to log to. | *required* |
| `validator` | `BaseValidator` | The validator instance containing validation data and predictions. | *required* |
| `curr_step` | `int` | The current training step for logging timeline. | *required* |

!!! note "Notes"

    This function uses global state to track the number of logged predictions across calls.
    It only logs predictions for supported tasks defined in COMET_SUPPORTED_TASKS.
    The number of logged images is limited by the COMET_MAX_IMAGE_PREDICTIONS environment variable.

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L372-L430"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_image_predictions(experiment, validator, curr_step) -> None:
    """Log image predictions to a Comet ML experiment during model validation.

    This function processes validation data and formats both ground truth and prediction annotations for visualization
    in the Comet dashboard. The function respects configured limits on the number of images to log.

    Args:
        experiment (comet_ml.CometExperiment): The Comet ML experiment to log to.
        validator (BaseValidator): The validator instance containing validation data and predictions.
        curr_step (int): The current training step for logging timeline.

    Notes:
        This function uses global state to track the number of logged predictions across calls.
        It only logs predictions for supported tasks defined in COMET_SUPPORTED_TASKS.
        The number of logged images is limited by the COMET_MAX_IMAGE_PREDICTIONS environment variable.
    """
    global _comet_image_prediction_count

    task = validator.args.task
    if task not in COMET_SUPPORTED_TASKS:
        return

    jdict = validator.jdict
    if not jdict:
        return

    predictions_metadata_map = _create_prediction_metadata_map(jdict)
    dataloader = validator.dataloader
    class_label_map = validator.names
    class_map = getattr(validator, "class_map", None)

    batch_logging_interval = _get_eval_batch_logging_interval()
    max_image_predictions = _get_max_image_predictions_to_log()

    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % batch_logging_interval != 0:
            continue

        image_paths = batch["im_file"]
        for img_idx, image_path in enumerate(image_paths):
            if _comet_image_prediction_count >= max_image_predictions:
                return

            image_path = Path(image_path)
            annotations = _fetch_annotations(
                img_idx,
                image_path,
                batch,
                predictions_metadata_map,
                class_label_map,
                class_map=class_map,
            )
            _log_images(
                experiment,
                [image_path],
                curr_step,
                annotations=annotations,
            )
            _comet_image_prediction_count += 1
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_plots` {#ultralytics.utils.callbacks.comet.\_log\_plots}

```python
def _log_plots(experiment, trainer) -> None
```

Log evaluation plots and label plots for the experiment.

This function logs various evaluation plots and confusion matrices to the experiment tracking system. It handles different types of metrics (SegmentMetrics, PoseMetrics, DetMetrics, OBBMetrics) and logs the appropriate plots for each type.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` | `comet_ml.CometExperiment` | The Comet ML experiment to log plots to. | *required* |
| `trainer` | `ultralytics.engine.trainer.BaseTrainer` | The trainer object containing validation metrics and save<br>    directory information. | *required* |

**Examples**

```python
>>> from ultralytics.utils.callbacks.comet import _log_plots
>>> _log_plots(experiment, trainer)
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L433-L477"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_plots(experiment, trainer) -> None:
    """Log evaluation plots and label plots for the experiment.

    This function logs various evaluation plots and confusion matrices to the experiment tracking system. It handles
    different types of metrics (SegmentMetrics, PoseMetrics, DetMetrics, OBBMetrics) and logs the appropriate plots for
    each type.

    Args:
        experiment (comet_ml.CometExperiment): The Comet ML experiment to log plots to.
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer object containing validation metrics and save
            directory information.

    Examples:
        >>> from ultralytics.utils.callbacks.comet import _log_plots
        >>> _log_plots(experiment, trainer)
    """
    plot_filenames = None
    if isinstance(trainer.validator.metrics, SegmentMetrics):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in SEGMENT_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, PoseMetrics):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in POSE_METRICS_PLOT_PREFIX
        ]
    elif isinstance(trainer.validator.metrics, (DetMetrics, OBBMetrics)):
        plot_filenames = [
            trainer.save_dir / f"{prefix}{plots}.png"
            for plots in EVALUATION_PLOT_NAMES
            for prefix in DETECTION_METRICS_PLOT_PREFIX
        ]

    if plot_filenames is not None:
        _log_images(experiment, plot_filenames, None)

    confusion_matrix_filenames = [trainer.save_dir / f"{plots}.png" for plots in CONFUSION_MATRIX_PLOT_NAMES]
    _log_images(experiment, confusion_matrix_filenames, None)

    if not isinstance(trainer.validator.metrics, ClassifyMetrics):
        label_plot_filenames = [trainer.save_dir / f"{labels}.jpg" for labels in LABEL_PLOT_NAMES]
        _log_images(experiment, label_plot_filenames, None)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_model` {#ultralytics.utils.callbacks.comet.\_log\_model}

```python
def _log_model(experiment, trainer) -> None
```

Log the best-trained model to Comet.ml.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` |  |  | *required* |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L480-L483"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_model(experiment, trainer) -> None:
    """Log the best-trained model to Comet.ml."""
    model_name = _get_comet_model_name()
    experiment.log_model(model_name, file_or_folder=str(trainer.best), file_name="best.pt", overwrite=True)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_image_batches` {#ultralytics.utils.callbacks.comet.\_log\_image\_batches}

```python
def _log_image_batches(experiment, trainer, curr_step: int) -> None
```

Log samples of image batches for train and validation.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` |  |  | *required* |
| `trainer` |  |  | *required* |
| `curr_step` | `int` |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L486-L489"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_image_batches(experiment, trainer, curr_step: int) -> None:
    """Log samples of image batches for train and validation."""
    _log_images(experiment, trainer.save_dir.glob("train_batch*.jpg"), curr_step)
    _log_images(experiment, trainer.save_dir.glob("val_batch*.jpg"), curr_step)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_asset` {#ultralytics.utils.callbacks.comet.\_log\_asset}

```python
def _log_asset(experiment, asset_path) -> None
```

Logs a specific asset file to the given experiment.

This function facilitates logging an asset, such as a file, to the provided experiment. It enables integration with experiment tracking platforms.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` | `comet_ml.CometExperiment` | The experiment instance to which the asset will be logged. | *required* |
| `asset_path` | `Path` | The file path of the asset to log. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L492-L502"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_asset(experiment, asset_path) -> None:
    """Logs a specific asset file to the given experiment.

    This function facilitates logging an asset, such as a file, to the provided
    experiment. It enables integration with experiment tracking platforms.

    Args:
        experiment (comet_ml.CometExperiment): The experiment instance to which the asset will be logged.
        asset_path (Path): The file path of the asset to log.
    """
    experiment.log_asset(asset_path)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet._log_table` {#ultralytics.utils.callbacks.comet.\_log\_table}

```python
def _log_table(experiment, table_path) -> None
```

Logs a table to the provided experiment.

This function is used to log a table file to the given experiment. The table is identified by its file path.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `experiment` | `comet_ml.CometExperiment` | The experiment object where the table file will be logged. | *required* |
| `table_path` | `Path` | The file path of the table to be logged. | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L505-L514"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def _log_table(experiment, table_path) -> None:
    """Logs a table to the provided experiment.

    This function is used to log a table file to the given experiment. The table is identified by its file path.

    Args:
        experiment (comet_ml.CometExperiment): The experiment object where the table file will be logged.
        table_path (Path): The file path of the table to be logged.
    """
    experiment.log_table(str(table_path))
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet.on_pretrain_routine_start` {#ultralytics.utils.callbacks.comet.on\_pretrain\_routine\_start}

```python
def on_pretrain_routine_start(trainer) -> None
```

Create or resume a CometML experiment at the start of a YOLO pre-training routine.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L517-L519"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_pretrain_routine_start(trainer) -> None:
    """Create or resume a CometML experiment at the start of a YOLO pre-training routine."""
    _resume_or_create_experiment(trainer.args)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet.on_train_epoch_end` {#ultralytics.utils.callbacks.comet.on\_train\_epoch\_end}

```python
def on_train_epoch_end(trainer) -> None
```

Log metrics and save batch images at the end of training epochs.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L522-L532"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_epoch_end(trainer) -> None:
    """Log metrics and save batch images at the end of training epochs."""
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]

    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix="train"), step=curr_step, epoch=curr_epoch)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet.on_fit_epoch_end` {#ultralytics.utils.callbacks.comet.on\_fit\_epoch\_end}

```python
def on_fit_epoch_end(trainer) -> None
```

Log model assets at the end of each epoch during training.

This function is called at the end of each training epoch to log metrics, learning rates, and model information to a Comet ML experiment. It also logs model assets, confusion matrices, and image predictions based on configuration settings.

The function retrieves the current Comet ML experiment and logs various training metrics. If it's the first epoch, it also logs model information. On specified save intervals, it logs the model, confusion matrix (if enabled), and image predictions (if enabled).

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` | `BaseTrainer` | The YOLO trainer object containing training state, metrics, and configuration. | *required* |

**Examples**

```python
>>> # Inside a training loop
>>> on_fit_epoch_end(trainer)  # Log metrics and assets to Comet ML
```

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L535-L576"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_fit_epoch_end(trainer) -> None:
    """Log model assets at the end of each epoch during training.

    This function is called at the end of each training epoch to log metrics, learning rates, and model information to a
    Comet ML experiment. It also logs model assets, confusion matrices, and image predictions based on configuration
    settings.

    The function retrieves the current Comet ML experiment and logs various training metrics. If it's the first epoch,
    it also logs model information. On specified save intervals, it logs the model, confusion matrix (if enabled), and
    image predictions (if enabled).

    Args:
        trainer (BaseTrainer): The YOLO trainer object containing training state, metrics, and configuration.

    Examples:
        >>> # Inside a training loop
        >>> on_fit_epoch_end(trainer)  # Log metrics and assets to Comet ML
    """
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    save_assets = metadata["save_assets"]

    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)
    if curr_epoch == 1:
        from ultralytics.utils.torch_utils import model_info_for_loggers

        experiment.log_metrics(model_info_for_loggers(trainer), step=curr_step, epoch=curr_epoch)

    if not save_assets:
        return

    _log_model(experiment, trainer)
    if _should_log_confusion_matrix():
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    if _should_log_image_predictions():
        _log_image_predictions(experiment, trainer.validator, curr_step)
```
</details>


<br><br><hr><br>

## Function `ultralytics.utils.callbacks.comet.on_train_end` {#ultralytics.utils.callbacks.comet.on\_train\_end}

```python
def on_train_end(trainer) -> None
```

Perform operations at the end of training.

**Args**

| Name | Type | Description | Default |
| --- | --- | --- | --- |
| `trainer` |  |  | *required* |

<details>
<summary>Source code in <code>ultralytics/utils/callbacks/comet.py</code></summary>

<a href="https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/callbacks/comet.py#L579-L610"><i class="fa-brands fa-github" aria-hidden="true" style="margin-right:6px;"></i>View on GitHub</a>
```python
def on_train_end(trainer) -> None:
    """Perform operations at the end of training."""
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]
    plots = trainer.args.plots

    _log_model(experiment, trainer)
    if plots:
        _log_plots(experiment, trainer)

    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    _log_image_predictions(experiment, trainer.validator, curr_step)
    _log_image_batches(experiment, trainer, curr_step)
    # log results table
    table_path = trainer.save_dir / RESULTS_TABLE_NAME
    if table_path.exists():
        _log_table(experiment, table_path)

    # log arguments YAML
    args_path = trainer.save_dir / ARGS_YAML_NAME
    if args_path.exists():
        _log_asset(experiment, args_path)

    experiment.end()

    global _comet_image_prediction_count
    _comet_image_prediction_count = 0
```
</details>

<br><br>
