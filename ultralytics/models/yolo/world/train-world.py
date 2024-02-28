from ultralytics.data import build_yolomultimodal_dataset, build_yolo_dataset, GroundingDataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils import DEFAULT_CFG


class WorldTrainerFromScratch(WorldTrainer):
    """
    A class extending the WorldTrainer class for training a world model from scratch on open-set dataset.

    Example:
        ```python
        from ultralytics.models.yolo.world import WorldTrainerFromScratch

        args = dict(model='yolov8s-world.pt', data='coco8.yaml', epochs=3)
        trainer = WorldTrainerFromScratch(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainer object with given arguments."""
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            pass
