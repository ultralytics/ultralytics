# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import copy

from ultralytics.models import yolo
from ultralytics.nn.tasks import MultiTaskModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import plot_images, plot_results


class MultiTaskTrainer(yolo.detect.DetectionTrainer):
    """
    A class extending the DetectionTrainer class for training based on a MultiTask model.

    Example:
        ```python
        from ultralytics.models.yolo.multitask import MultiTaskTrainer

        args = dict(model='yolov8n-multitask.pt', data='coco8-multitask.yaml', epochs=3)
        trainer = MultiTaskTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a MultiTaskTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "multitask"
        super().__init__(cfg, overrides, _callbacks)

        if isinstance(self.args.device, str) and self.args.device.lower() == "mps":
            LOGGER.warning(
                "WARNING ⚠️ Apple MPS known Pose bug. Recommend 'device=cpu' for Pose models. "
                "See https://github.com/ultralytics/ultralytics/issues/4031."
            )

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return MultiTaskModel initialized with specified config and weights."""
        model = MultiTaskModel(
            cfg, ch=3, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1
        )
        if weights:
            model.load(weights)

        return model

    def set_model_attributes(self):
        """Sets keypoints shape attribute of MultiTaskModel."""
        super().set_model_attributes()
        self.model.kpt_shape = self.data["kpt_shape"]

    def get_validator(self):
        """Return an instance of MultiTaskValidator for validation of YOLO model."""
        self.loss_names = "box_loss", "seg_loss", "cls_loss", "dfl_loss", "pose_loss", "kobj_loss"
        return yolo.multitask.MultiTaskValidator(self.test_loader, save_dir=self.save_dir, args=copy(self.args))

    def plot_training_samples(self, batch, ni):
        """Creates a plot of training sample images with labels and box coordinates, masks and keypoints."""
        images = batch["img"]
        cls = batch["cls"].squeeze(-1)
        bboxes = batch["bboxes"]
        masks = batch["masks"]
        kpts = batch["keypoints"]
        paths = batch["im_file"]
        batch_idx = batch["batch_idx"]

        plot_images(
            images,
            batch_idx,
            cls,
            bboxes,
            masks=masks,
            kpts=kpts,
            paths=paths,
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_metrics(self):
        """Plots training/val metrics."""
        plot_results(file=self.csv, segment=True, pose=True, on_plot=self.on_plot)  # save results.png
