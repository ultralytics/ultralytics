# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import SegmentationPoseModel
from ultralytics.utils import DEFAULT_CFG, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class SegmentationPoseTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a segmentation-pose model.

    Example:
        ```python
        from ultralytics.models.yolo.segment_pose import SegmentationPoseTrainer

        args = dict(model='yolov8n-segpose.pt', data='coco8-segpose.yaml', epochs=3)
        trainer = SegmentationPoseTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a SegmentationPoseTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "segment_pose"
        super().__init__(cfg, overrides, _callbacks)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return SegmentationPoseModel initialized with specified config and weights."""
        model = SegmentationPoseModel(cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data['kpt_shape'], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        return model

    def get_validator(self):
        """Return an instance of SegmentationPoseValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", 'pose_loss', 'kobj_loss'
        return yolo.segment_pose.SegmentationPoseValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates."""
        plot_images(
            batch["img"],
            batch["batch_idx"],
            batch["cls"].squeeze(-1),
            batch["bboxes"],
            masks=batch["masks"],
            kpts=batch["keypoints"],
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, pose=True, on_plot=self.on_plot)  # save results.png
