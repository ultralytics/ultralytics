# Ultralytics YOLO 🚀, GPL-3.0 license
import os
from pathlib import Path

from ultralytics.yolo.utils import LOGGER, RANK, TESTS_RUNNING, ops
from ultralytics.yolo.utils.torch_utils import get_flops, get_num_params

try:
    import comet_ml

    assert not TESTS_RUNNING  # do not log pytest
    assert hasattr(comet_ml, '__version__')  # verify package is not directory
except (ImportError, AssertionError):
    comet_ml = None

COMET_MODE = os.getenv('COMET_MODE', 'online')
COMET_MODEL_NAME = os.getenv('COMET_MODEL_NAME', 'YOLOv8')
# determines how many batches of image predictions to log from the validation set
COMET_EVAL_BATCH_LOGGING_INTERVAL = int(os.getenv('COMET_EVAL_BATCH_LOGGING_INTERVAL', 1))
# determines whether to log confusion matrix every evaluation epoch
COMET_EVAL_LOG_CONFUSION_MATRIX = (os.getenv('COMET_EVAL_LOG_CONFUSION_MATRIX', 'true').lower() == 'true')
# determines whether to log image predictions every evaluation epoch
COMET_EVAL_LOG_IMAGE_PREDICTIONS = (os.getenv('COMET_EVAL_LOG_IMAGE_PREDICTIONS', 'true').lower() == 'true')
COMET_MAX_IMAGE_PREDICTIONS = int(os.getenv('COMET_MAX_IMAGE_PREDICTIONS', 100))

# ensures certain logging functions only run for supported tasks
COMET_SUPPORTED_TASKS = ['detect']
# scales reported confidence scores (0.0-1.0) by this value
COMET_MAX_CONFIDENCE_SCORE = int(os.getenv('COMET_MAX_CONFIDENCE_SCORE', 100))

# names of plots created by YOLOv8 that are logged to Comet
EVALUATION_PLOT_NAMES = 'F1_curve', 'P_curve', 'R_curve', 'PR_curve', 'confusion_matrix'
LABEL_PLOT_NAMES = 'labels', 'labels_correlogram'

_comet_image_prediction_count = 0


def _get_experiment_type(mode, project_name):
    if mode == 'offline':
        return comet_ml.OfflineExperiment(project_name=project_name)

    return comet_ml.Experiment(project_name=project_name)


def _create_experiment(args):
    # Ensures that the experiment object is only created in a single process during distributed training.
    if RANK not in (-1, 0):
        return
    try:
        experiment = _get_experiment_type(COMET_MODE, args.project)
        experiment.log_parameters(vars(args))
        experiment.log_others({
            'eval_batch_logging_interval': COMET_EVAL_BATCH_LOGGING_INTERVAL,
            'log_confusion_matrix': COMET_EVAL_LOG_CONFUSION_MATRIX,
            'log_image_predictions': COMET_EVAL_LOG_IMAGE_PREDICTIONS,
            'max_image_predictions': COMET_MAX_IMAGE_PREDICTIONS, })
        experiment.log_other('Created from', 'yolov8')

    except Exception as e:
        LOGGER.warning(f'WARNING ⚠️ Comet installed but not initialized correctly, not logging this run. {e}')


def _fetch_trainer_metadata(trainer):
    curr_epoch = trainer.epoch + 1

    train_num_steps_per_epoch = len(trainer.train_loader.dataset) // trainer.batch_size
    curr_step = curr_epoch * train_num_steps_per_epoch
    final_epoch = curr_epoch == trainer.epochs

    save = trainer.args.save
    save_period = trainer.args.save_period
    save_interval = curr_epoch % save_period == 0
    save_assets = save and save_period > 0 and save_interval and not final_epoch

    return dict(curr_epoch=curr_epoch, curr_step=curr_step, save_assets=save_assets, final_epoch=final_epoch)


def _scale_bounding_box_to_original_image_shape(box, resized_image_shape, original_image_shape, ratio_pad):
    """YOLOv8 resizes images during training and the label values
    are normalized based on this resized shape. This function rescales the
    bounding box labels to the original image shape.
    """

    resized_image_height, resized_image_width = resized_image_shape

    # convert normalized xywh format predictions to xyxy in resized scale format
    box = ops.xywhn2xyxy(box, h=resized_image_height, w=resized_image_width)
    # scale box predictions from resized image scale back to original image scale
    box = ops.scale_boxes(resized_image_shape, box, original_image_shape, ratio_pad)
    # Convert bounding box format from xyxy to xywh for Comet logging
    box = ops.xyxy2xywh(box)
    # adjust xy center to correspond top-left corner
    box[:2] -= box[2:] / 2
    box = box.tolist()

    return box


def _format_ground_truth_annotations_for_detection(img_idx, image_path, batch, class_name_map=None):
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
        data.append({'boxes': [box], 'label': f'gt_{label}', 'score': COMET_MAX_CONFIDENCE_SCORE})

    return {'name': 'ground_truth', 'data': data}


def _format_prediction_annotations_for_detection(image_path, metadata, class_label_map=None):
    stem = image_path.stem
    image_id = int(stem) if stem.isnumeric() else stem

    predictions = metadata.get(image_id)
    if not predictions:
        LOGGER.debug(f'COMET WARNING: Image: {image_path} has no bounding boxes predictions')
        return None

    data = []
    for prediction in predictions:
        boxes = prediction['bbox']
        score = prediction['score'] * COMET_MAX_CONFIDENCE_SCORE
        cls_label = prediction['category_id']
        if class_label_map:
            cls_label = str(class_label_map[cls_label])

        data.append({'boxes': [boxes], 'label': cls_label, 'score': score})

    return {'name': 'prediction', 'data': data}


def _fetch_annotations(img_idx, image_path, batch, prediction_metadata_map, class_label_map):
    ground_truth_annotations = _format_ground_truth_annotations_for_detection(img_idx, image_path, batch,
                                                                              class_label_map)
    prediction_annotations = _format_prediction_annotations_for_detection(image_path, prediction_metadata_map,
                                                                          class_label_map)

    annotations = [
        annotation for annotation in [ground_truth_annotations, prediction_annotations] if annotation is not None]
    return [annotations] if annotations else None


def _create_prediction_metadata_map(model_predictions):
    pred_metadata_map = {}
    for prediction in model_predictions:
        pred_metadata_map.setdefault(prediction['image_id'], [])
        pred_metadata_map[prediction['image_id']].append(prediction)

    return pred_metadata_map


def _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch):
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
    if annotations:
        for image_path, annotation in zip(image_paths, annotations):
            experiment.log_image(image_path, name=image_path.stem, step=curr_step, annotations=annotation)

    else:
        for image_path in image_paths:
            experiment.log_image(image_path, name=image_path.stem, step=curr_step)


def _log_image_predictions(experiment, validator, curr_step):
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

    for batch_idx, batch in enumerate(dataloader):
        if (batch_idx + 1) % COMET_EVAL_BATCH_LOGGING_INTERVAL != 0:
            continue

        image_paths = batch['im_file']
        for img_idx, image_path in enumerate(image_paths):
            if _comet_image_prediction_count >= COMET_MAX_IMAGE_PREDICTIONS:
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
    plot_filenames = [trainer.save_dir / f'{plots}.png' for plots in EVALUATION_PLOT_NAMES]
    _log_images(experiment, plot_filenames, None)

    label_plot_filenames = [trainer.save_dir / f'{labels}.jpg' for labels in LABEL_PLOT_NAMES]
    _log_images(experiment, label_plot_filenames, None)


def _log_model(experiment, trainer):
    experiment.log_model(
        COMET_MODEL_NAME,
        file_or_folder=str(trainer.best),
        file_name='best.pt',
        overwrite=True,
    )


def on_pretrain_routine_start(trainer):
    experiment = comet_ml.get_global_experiment()
    if not experiment:
        _create_experiment(trainer.args)


def on_train_epoch_end(trainer):
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
        model_info = {
            'model/parameters': get_num_params(trainer.model),
            'model/GFLOPs': round(get_flops(trainer.model), 3),
            'model/speed(ms)': round(trainer.validator.speed['inference'], 3)}
        experiment.log_metrics(model_info, step=curr_step, epoch=curr_epoch)

    if not save_assets:
        return

    _log_model(experiment, trainer)
    if COMET_EVAL_LOG_CONFUSION_MATRIX:
        _log_confusion_matrix(experiment, trainer, curr_step, curr_epoch)
    if COMET_EVAL_LOG_IMAGE_PREDICTIONS:
        _log_image_predictions(experiment, trainer.validator, curr_step)


def on_train_end(trainer):
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
