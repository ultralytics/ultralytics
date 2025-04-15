# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, List, Optional

import cv2
import numpy as np

from ultralytics.utils import LOGGER, RANK, SETTINGS, TESTS_RUNNING, ops
from ultralytics.utils.metrics import ClassifyMetrics, DetMetrics, OBBMetrics, PoseMetrics, SegmentMetrics

try:
    assert not TESTS_RUNNING  # do not log pytest
    assert SETTINGS["comet"] is True  # verify integration is enabled
    import comet_ml

    assert hasattr(comet_ml, "__version__")  # verify package is not directory

    import os
    from pathlib import Path

    # Ensures certain logging functions only run for supported tasks
    COMET_SUPPORTED_TASKS = ["detect", "segment"]

    # Names of plots created by Ultralytics that are logged to Comet
    CONFUSION_MATRIX_PLOT_NAMES = "confusion_matrix", "confusion_matrix_normalized"
    EVALUATION_PLOT_NAMES = "F1_curve", "P_curve", "R_curve", "PR_curve"
    LABEL_PLOT_NAMES = "labels", "labels_correlogram"
    SEGMENT_METRICS_PLOT_PREFIX = "Box", "Mask"
    POSE_METRICS_PLOT_PREFIX = "Box", "Pose"

    _comet_image_prediction_count = 0

except (ImportError, AssertionError):
    comet_ml = None


def _get_comet_mode() -> str:
    """Returns the mode of comet set in the environment variables, defaults to 'online' if not set."""
    comet_mode = os.getenv("COMET_MODE")
    if comet_mode is not None:
        LOGGER.warning(
            "WARNING ⚠️ The COMET_MODE environment variable is deprecated. "
            "Please use COMET_START_ONLINE to set the Comet experiment mode. "
            "To start an offline Comet experiment, use 'export COMET_START_ONLINE=0'. "
            "If COMET_START_ONLINE is not set or is set to '1', an online Comet experiment will be created."
        )
        return comet_mode

    return "online"


def _get_comet_model_name() -> str:
    """Returns the model name for Comet from the environment variable COMET_MODEL_NAME or defaults to 'Ultralytics'."""
    return os.getenv("COMET_MODEL_NAME", "Ultralytics")


def _get_eval_batch_logging_interval() -> int:
    """Get the evaluation batch logging interval from environment variable or use default value 1."""
    return int(os.getenv("COMET_EVAL_BATCH_LOGGING_INTERVAL", 1))


def _get_max_image_predictions_to_log() -> int:
    """Get the maximum number of image predictions to log from the environment variables."""
    return int(os.getenv("COMET_MAX_IMAGE_PREDICTIONS", 100))


def _scale_confidence_score(score: float) -> float:
    """Scales the given confidence score by a factor specified in an environment variable."""
    scale = float(os.getenv("COMET_MAX_CONFIDENCE_SCORE", 100.0))
    return score * scale


def _should_log_confusion_matrix() -> bool:
    """Determines if the confusion matrix should be logged based on the environment variable settings."""
    return os.getenv("COMET_EVAL_LOG_CONFUSION_MATRIX", "false").lower() == "true"


def _should_log_image_predictions() -> bool:
    """Determines whether to log image predictions based on a specified environment variable."""
    return os.getenv("COMET_EVAL_LOG_IMAGE_PREDICTIONS", "true").lower() == "true"


def _resume_or_create_experiment(args: SimpleNamespace) -> None:
    """
    Resumes CometML experiment or creates a new experiment based on args.

    Ensures that the experiment object is only created in a single process during distributed training.
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
        LOGGER.warning(f"WARNING ⚠️ Comet installed but not initialized correctly, not logging this run. {e}")


def _fetch_trainer_metadata(trainer) -> dict:
    """Returns metadata for YOLO training including epoch and asset saving status."""
    curr_epoch = trainer.epoch + 1

    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs

    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and not final_epoch

    return dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=save_assets, final_epoch=final_epoch)


def _scale_bounding_box_to_original_image_shape(
    box, resized_image_shape, original_image_shape, ratio_pad
) -> List[float]:
    """
    YOLO resizes images during training and the label values are normalized based on this resized shape.

    This function rescales the bounding box labels to the original image shape.
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


def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None) -> Optional[dict]:
    """
    Format ground truth annotations for object detection.

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
        class_name_map (dict | None, optional): Mapping from class indices to class names.

    Returns:
        (dict | None): Formatted ground truth annotations with the following structure:
            - 'boxes': List of box coordinates [x, y, width, height]
            - 'label': Label string with format "gt_{class_name}"
            - 'score': Confidence score (always 1.0, scaled by _scale_confidence_score)
            Returns None if no bounding boxes are found for the image.
    """
    indices = batch["batch_idx"] == img_idx
    bboxes = batch["bboxes"][indices]
    if len(bboxes) == 0:
        LOGGER.debug(f"COMET WARNING: Image: {image_path} has no bounding boxes labels")
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


def _format_prediction_annotations(image_path, metadata, class_label_map=None, class_map=None) -> Optional[dict]:
    """Format YOLO predictions for object detection visualization."""
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem

    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(f"COMET WARNING: Image: {image_path} has no bounding boxes predictions")
        return None

    # apply the mapping that was used to map the predicted classes when the JSON was created
    if class_label_map and class_map:
        class_label_map = {class_map[k]: v for k, v in class_label_map.items()}
    try:
        # import pycotools utilities to decompress annotations for various tasks, e.g. segmentation
        from pycocotools.mask import decode  # noqa
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


def _extract_segmentation_annotation(segmentation_raw: str, decode: Callable) -> Optional[List[List[Any]]]:
    """
    Extracts segmentation annotation from compressed segmentations as list of polygons.

    Args:
        segmentation_raw: Raw segmentation data in compressed format.
        decode: Function to decode the compressed segmentation data.

    Returns:
        (Optional[List[List[Any]]]): List of polygon points or None if extraction fails.
    """
    try:
        mask = decode(segmentation_raw)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        annotations = [np.array(polygon).squeeze() for polygon in contours if len(polygon) >= 3]
        return [annotation.ravel().tolist() for annotation in annotations]
    except Exception as e:
        LOGGER.warning(f"COMET WARNING: Failed to extract segmentation annotation: {e}")
    return None


def _fetch_annotations(
    img_idx, image_path, batch, prediction_metadata_map, class_label_map, class_map
) -> Optional[List]:
    """Join the ground truth and prediction annotations if they exist."""
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


def _create_prediction_metadata_map(model_predictions) -> dict:
    """Create metadata map for model predictions by groupings them based on image ID."""
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction["image_id"], [])
        pred_metadata_map[prediction["image_id"]].append(prediction)

    return pred_metadata_map


def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch) -> None:
    """Log the confusion matrix to Comet experiment."""
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = list(trainer.data["names"].values()) + ["background"]
    experiment.log_confusion_matrix(
        matrix=conf_mat, labels=names, max_categories=len(names), epoch=curr_epoch, step=curr_step
    )


def _log_images(experiment, image_paths, curr_step, annotations=None) -> None:
    """
    Log images to the experiment with optional annotations.

    This function logs images to a Comet ML experiment, optionally including annotation data for visualization
    such as bounding boxes or segmentation masks.

    Args:
        experiment (comet_ml.Experiment): The Comet ML experiment to log images to.
        image_paths (List[Path]): List of paths to images that will be logged.
        curr_step (int): Current training step/iteration for tracking in the experiment timeline.
        annotations (List[List[dict]], optional): Nested list of annotation dictionaries for each image. Each annotation
            contains visualization data like bounding boxes, labels, and confidence scores.

    Returns:
        None
    """
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)

    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)


def _log_image_predictions(experiment, validator, curr_step) -> None:
    """
    Log predicted boxes for a single image during training.

    This function logs image predictions to a Comet ML experiment during model validation. It processes
    validation data and formats both ground truth and prediction annotations for visualization in the Comet
    dashboard. The function respects configured limits on the number of images to log.

    Args:
        experiment (comet_ml.Experiment): The Comet ML experiment to log to.
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


def _log_plots(experiment, trainer) -> None:
    """
    Log evaluation plots and label plots for the experiment.

    This function logs various evaluation plots and confusion matrices to the experiment tracking system. It handles
    different types of metrics (SegmentMetrics, PoseMetrics, DetMetrics, OBBMetrics) and logs the appropriate plots
    for each type.

    Args:
        experiment (comet_ml.Experiment): The Comet ML experiment to log plots to.
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer object containing validation metrics and save
            directory information.

    Examples:
        >>> from ultralytics.utils.callbacks.comet import _log_plots
        >>> _log_plots(experiment, trainer)
    """
    plot_filenames = None
    if isinstance(trainer.validator.metrics, SegmentMetrics) and trainer.validator.metrics.task == "segment":
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
        plot_filenames = [trainer.save_dir / f"{plots}.png" for plots in EVALUATION_PLOT_NAMES]

    if plot_filenames is not None:
        _log_images(experiment, plot_filenames, None)

    confusion_matrix_filenames = [trainer.save_dir / f"{plots}.png" for plots in CONFUSION_MATRIX_PLOT_NAMES]
    _log_images(experiment, confusion_matrix_filenames, None)

    if not isinstance(trainer.validator.metrics, ClassifyMetrics):
        label_plot_filenames = [trainer.save_dir / f"{labels}.jpg" for labels in LABEL_PLOT_NAMES]
        _log_images(experiment, label_plot_filenames, None)


def _log_model(experiment, trainer) -> None:
    """Log the best-trained model to Comet.ml."""
    model_name = _get_comet_model_name()
    experiment.log_model(model_name, file_or_folder=str(trainer.best), file_name="best.pt", overwrite=True)


def _log_image_batches(experiment, trainer, curr_step: int) -> None:
    """Log samples of images batches for train, validation, and test."""
    _log_images(experiment, trainer.save_dir.glob("train_batch*.jpg"), curr_step)
    _log_images(experiment, trainer.save_dir.glob("val_batch*.jpg"), curr_step)


def on_pretrain_routine_start(trainer) -> None:
    """Creates or resumes a CometML experiment at the start of a YOLO pre-training routine."""
    _resume_or_create_experiment(trainer.args)


def on_train_epoch_end(trainer) -> None:
    """Log metrics and save batch images at the end of training epochs."""
    experiment = comet_ml.get_running_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata["curr_epoch"]
    curr_step = metadata["curr_step"]

    experiment.log_metrics(trainer.label_loss_items(trainer.tloss, prefix="train"), step=curr_step, epoch=curr_epoch)


def on_fit_epoch_end(trainer) -> None:
    """
    Log model assets at the end of each epoch during training.

    This function is called at the end of each training epoch to log metrics, learning rates, and model information
    to a Comet ML experiment. It also logs model assets, confusion matrices, and image predictions based on
    configuration settings.

    The function retrieves the current Comet ML experiment and logs various training metrics. If it's the first epoch,
    it also logs model information. On specified save intervals, it logs the model, confusion matrix (if enabled),
    and image predictions (if enabled).

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
    experiment.end()

    global _comet_image_prediction_count
    _comet_image_prediction_count = 0


callbacks = (
    {
        "on_pretrain_routine_start": on_pretrain_routine_start,
        "on_train_epoch_end": on_train_epoch_end,
        "on_fit_epoch_end": on_fit_epoch_end,
        "on_train_end": on_train_end,
    }
    if comet_ml
    else {}
)
