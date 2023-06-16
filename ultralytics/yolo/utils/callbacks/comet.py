# Ultralytics YOLO 🚀, AGPL-3.0 license

import os
from pathlib import Path

from ultralytics.yolo.utils import LOGGER, RANK, TESTS_RUNNING, ops
from ultralytics.yolo.utils.torch_utils import model_info_for_loggers

try:
    import comet_ml

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(comet_ml, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    comet_ml = None

# Ensures certain logging functions only run for supported tasks
COMET_SUPPORTED_TASKS = ['detect']

# Names of plots created by YOLOv8 that are logged to Comet
EVALUATION_PLOT_NAMES = 'F1_curve', 'P_curve', 'R_curve', 'PR_curve', 'confusion_matrix'
LABEL_PLOT_NAMES = 'labels', 'labels_correlogram'

_comet_image_prediction_count = 0


def _get_comet_mode():
    return os.getenv('COMET_MODE', 'online')


def _get_comet_model_name():
    return os.getenv('COMET_MODEL_NAME', 'YOLOv8')


def _get_eval_batch_logging_interval():
    return int(os.getenv('COMET_EVAL_BATCH_LOGGING_INTERVAL', 1))


def _get_max_image_predictions_to_log():
    return int(os.getenv('COMET_MAX_IMAGE_PREDICTIONS', 100))


def _scale_confidence_score(score):
    scale = float(os.getenv('COMET_MAX_CONFIDENCE_SCORE', 100.0))
    return score * scale


def _should_log_confusion_matrix():
    return os.getenv('COMET_EVAL_LOG_CONFUSION_MATRIX', 'false').lower() == 'true'


def _should_log_image_predictions():
    return os.getenv('COMET_EVAL_LOG_IMAGE_PREDICTIONS', 'true').lower() == 'true'


def _get_experiment_type(mode, project_name):
    """Return an experiment based on mode and project name."""
    if mode == 'offline':
        return comet_ml.OfflineExperiment(project_name=project_name)

    return comet_ml.Experiment(project_name=project_name)


def _create_experiment(args):
    """Ensures that the experiment object is only created in a single process during distributed training."""
    if RANK not in (-1, 0):
        return
    try:
        comet_mode = _get_comet_mode()
        _project_name = os.getenv('COMET_PROJECT_NAME', args.project)
        experiment = _get_experiment_type(comet_mode, _project_name)
        experiment.log_parameters(vars(args))
        experiment.log_others({
            'eval_batch_logging_interval': _get_eval_batch_logging_interval(),
            'log_confusion_matrix_on_eval': _should_log_confusion_matrix(),
            'log_image_predictions': _should_log_image_predictions(),
            'max_image_predictions': _get_max_image_predictions_to_log(), })
        experiment.log_other('Created from', 'yolov8')

    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ Comet installed but not initialized correctly, not logging this run. {e}')


def _fetch_trainer_metadata(trainer):
    """Returns metadata for YOLO training including epoch and asset saving status."""
    curr_epoch = trainer.epoch + 1

    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs

    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and not final_epoch

    return dict(
        curr_epoch=curr_epoch,
        curr_step=curr_step,
        save_assets=save_assets,
        final_epoch=final_epoch,
    )


def _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad):
    """YOLOv8 resizes images during training and the label values
    are normalized based on this resized shape. This function rescales the
    bounding box labels to the original image shape.
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


def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None):
    """Format ground truth annotations for detection."""
    indices = batch['batch_idx'] == img_idx
    bboxes = batch['bboxes'][indices]
    if len(bboxes) == 0:
        LOGGER.debug(f'COMET WARNING: Image: {image_path} has no bounding boxes labels')
        return None

    cls_labels = batch['cls'][indices].squeeze(1).tolist()
    if class_name_map:
        cls_labels = [str(class_name_map[label]) for label in cls_labels]

    original_image_shape = batch['ori_shape'][img_idx]
    resized_image_shape = batch['resized_shape'][img_idx]
    ratio_pad = batch['ratio_pad'][img_idx]

    data = []
    for box, label in zip(bboxes, cls_labels):
        box = _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad)
        data.append({
            'boxes': [box],
            'label': f'gt_{label}',
            'score': _scale_confidence_score(1.0), })

    return {'name': 'ground_truth', 'data': data}


def _format_prediction_annotations_for_detection(image_path, metadata, class_label_map=None):
    """Format YOLO predictions for object detection visualization."""
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem

    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(f'COMET WARNING: Image: {image_path} has no bounding boxes predictions')
        return None

    data = []
    for prediction in predictions:
        boxes = prediction['bbox']
        score = _scale_confidence_score(prediction['score'])
        cls_label = prediction['category_id']
        if class_label_map:
            cls_label = str(class_label_map[cls_label])

        data.append({'boxes': [boxes], 'label': cls_label, 'score': score})

    return {'name': 'prediction', 'data': data}


def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map):
    """Join the ground truth and prediction annotations if they exist."""
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(img_idx, image_path, batch,
                                                                              class_label_map)
    prediction_annotations = _format_prediction_annotations_for_detection(image_path, prediction_metadata_map,
                                                                          class_label_map)

    annotations = [
        annotation for annotation in [ground_truth_annotations, prediction_annotations] if annotation is not None]
    return [annotations] if annotations else None


def _create_prediction_metadata_map(model_predictions):
    """Create metadata map for model predictions by groupings them based on image ID."""
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction['image_id'], [])
        pred_metadata_map[prediction['image_id']].append(prediction)

    return pred_metadata_map


def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch):
    """Log the confusion matrix to Comet experiment."""
    conf_mat = trainer.validator.confusion_matrix.matrix
    names = list(trainer.data['names'].values()) + ['background']
    experiment.log_confusion_matrix(
        matrix=conf_mat,
        labels=names,
        max_categories=len(names),
        epoch=curr_epoch,
        step=curr_step,
    )


def _log_images(experiment, image_paths, curr_step, annotations=None):
    """Logs images to the experiment with optional annotations."""
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)

    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)


def _log_image_predictions(experiment, validator, curr_step):
    """Logs predicted boxes for a single image during training."""
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

    batch_logging_interval = _get_eval_batch_logging_interval()
    max_image_predictions = _get_max_image_predictions_to_log()

    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % batch_logging_interval != 0:
            continue

        image_paths = batch['im_file']
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
            )
            _log_images(
                experiment,
                [image_path],
                curr_step,
                annotations=annotations,
            )
            _comet_image_prediction_count += 1


def _log_plots(experiment, trainer):
    """Logs evaluation plots and label plots for the experiment."""
    plot_filenames = [trainer.save_dir / f'{plots}.png' for plots in EVALUATION_PLOT_NAMES]
    _log_images(experiment, plot_filenames, None)

    label_plot_filenames = [trainer.save_dir / f'{labels}.jpg' for labels in LABEL_PLOT_NAMES]
    _log_images(experiment, label_plot_filenames, None)


def _log_model(experiment, trainer):
    """Log the best-trained model to Comet.ml."""
    model_name = _get_comet_model_name()
    experiment.log_model(
        model_name,
        file_or_folder=str(trainer.best),
        file_name='best.pt',
        overwrite=True,
    )


def on_pretrain_routine_start(trainer):
    """Creates or resumes a CometML experiment at the start of a YOLO pre-training routine."""
    experiment = comet_ml.get_global_experiment()
    is_alive = getattr(experiment, 'alive', False)
    if not experiment or not is_alive:
        _create_experiment(trainer.args)


def on_train_epoch_end(trainer):
    """Log metrics and save batch images at the end of training epochs."""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']

    experiment.log_metrics(
        trainer.label_loss_items(trainer.tloss, prefix='train'),
        step=curr_step,
        epoch=curr_epoch,
    )

    if curr_epoch == 1:
        _log_images(experiment, trainer.save_dir.glob('train_batch*.jpg'), curr_step)


def on_fit_epoch_end(trainer):
    """Logs model assets at the end of each epoch."""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']
    save_assets = metadata['save_assets']

    experiment.log_metrics(trainer.metrics, step=curr_step, epoch=curr_epoch)
    experiment.log_metrics(trainer.lr, step=curr_step, epoch=curr_epoch)
    if curr_epoch == 1:
        experiment.log_metrics(model_info_for_loggers(trainer), step=curr_step, epoch=curr_epoch)

    if not save_assets:
        return

    _log_model(experiment, trainer)
    if _should_log_confusion_matrix():
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    if _should_log_image_predictions():
        _log_image_predictions(experiment, trainer.validator, curr_step)


def on_train_end(trainer):
    """Perform operations at the end of training."""
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        return

    metadata = _fetch_trainer_metadata(trainer)
    curr_epoch = metadata['curr_epoch']
    curr_step = metadata['curr_step']
    plots = trainer.args.plots

    _log_model(experiment, trainer)
    if plots:
        _log_plots(experiment, trainer)

    _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    _log_image_predictions(experiment, trainer.validator, curr_step)
    experiment.end()

    global _comet_image_prediction_count
    _comet_image_prediction_count = 0


callbacks = {
    'on_pretrain_routine_start': on_pretrain_routine_start,
    'on_train_epoch_end': on_train_epoch_end,
    'on_fit_epoch_end': on_fit_epoch_end,
    'on_train_end': on_train_end} if comet_ml else {}
