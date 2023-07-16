# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.plotting import plot_images, plot_results


# BaseTrainer python usage
class PoseTrainer(yolo.detect.DetectionTrainer):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a PoseTrainer object with specified configurations and overrides."""
        if overrides is None:
            overrides = {}
        overrides['task'] = 'pose'
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get pose estimation model with specified configuration and weights."""
        model = PoseModel(cfg, ch=3, nc=self.data['nc'], data_kpt_shape=self.data['kpt_shape'], verbose=verbose)
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of PoseModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data['kpt_shape']

    def get_validator(self):
        """Returns an instance of the PoseValidator class for validation."""
        self.loss_names = 'box_loss', 'pose_loss', 'kobj_loss', 'cls_loss', 'dfl_loss'
        return yolo.pose.PoseValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Plot a batch of training samples with annotated class labels, bounding boxes, and keypoints."""
        images = batch['img']
        kpts = batch['keypoints']
        cls = batch['cls'].squeeze(-1)
        bboxes = batch['bboxes']
        paths = batch['im_file']
        batch_idx = batch['batch_idx']
        plot_images(images,
                    batch_idx,
                    cls,
                    bboxes,
                    kpts=kpts,
                    paths=paths,
                    fname=self.save_dir / f'train_batch{ni}.jpg',
                    on_plot=self.on_plot)

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, pose=True, on_plot=self.on_plot)  # save results.png


def train(cfg=DEFAULT_CFG, use_python=False):
    """Train the YOLO model on the given data and device."""
    model = cfg.model or 'yolov8n-pose.yaml'
    data = cfg.data or 'coco8-pose.yaml'
    device = cfg.device if cfg.device is not None else ''

    args = dict(model=model, data=data, device=device)
    if use_python:
        from ultralytics import YOLO
        YOLO(model).train(**args)
    else:
        trainer = PoseTrainer(overrides=args)
        trainer.train()


if __name__ == '__main__':
    train()
