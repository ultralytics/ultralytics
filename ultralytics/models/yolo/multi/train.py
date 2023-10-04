# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import MultiTaskModel
from ultralytics.utils import DEFAULT_CFG, LOGGER
from ultralytics.utils.plotting import plot_images, plot_results


class MultiTaskTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a pose model.

    Example:
        ```python
        from ultralytics.models.yolo.multi import MultiTaskTrainer

        args = dict(model='yolov8n-multi.pt', data='coco8-multi.yaml', epochs=3)
        trainer = MultiTaskTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'multi-task'
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == 'mps':
            # TODO: check if MultiTaskTrainer also has this bug
            LOGGER.warning("WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                           'See https://github.com/ultralytics/ultralytics/issues/4031.')

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = MultiTaskModel(cfg, ch=3, nc=self.data['nc'], data_kpt_shape=self.data['kpt_shape'], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data['kpt_shape']

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = ('box_loss', 'pose_loss', 'kobj_loss', 'seg_loss', 'cls_loss', 'dfl_loss')
        return yolo.multi.MultiTaskValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, batch_number):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        plot_images(
            batch['img'],
            batch['batch_idx'],
            batch['cls'].squeeze(-1),
            batch['bboxes'],
            kpts=batch['keypoints'],
            masks=batch['masks'],
            paths=batch['im_file'],
            fname=self.save_dir / f'train_batch{batch_number}.jpg',
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, pose=True, on_plot=self.on_plot)
