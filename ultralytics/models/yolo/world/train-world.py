from ultralytics.data import build_yolomultimodal_dataset, build_yolo_dataset, build_grounding, YOLOConcatDataset
from ultralytics.data.utils import check_det_dataset
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
        # NOTE: debug
        self.flickr30k_data = dict(
            img_path="../datasets/flickr30k/images",
            json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
        )
        self.gqa_data = dict(
            img_path="../datasets/GQA/images",
            json_file="../datasets/GQA/final_flickr_separateGT_train.json",
        )

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
            # yolo_multimodal = build_yolomultimodal_dataset(self.args, img_path, batch, self.data["train"], stride=gs)
            flickr30k = build_grounding(
                self.args,
                self.flickr30k_data["img_path"],
                self.flickr30k_data["json_file"],
                batch,
                stride=gs,
            )
            gqa = build_grounding(
                self.args,
                self.gqa_data["img_path"],
                self.gqa_data["json_file"],
                batch,
                self.data["train"],
                stride=gs,
            )
            return YOLOConcatDataset([flickr30k, gqa])
        else:
            return build_yolo_dataset(
                self.args, img_path, batch, self.data["val"], mode=mode, rect=mode == "val", stride=gs
            )

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        data_yaml = self.args.data
        if isinstance(data_yaml, dict):
            assert data_yaml.get("train", False)  # object365.yaml
            assert data_yaml.get("val", False)  # lvis.yaml
            data = {k: check_det_dataset(v) for k, v in data_yaml.items()}
            if data["val"].get("minival", None) is not None:
                data["val"]["minival"] = str(data["val"]["path"] / data["val"]["minival"])
        else:
            data = check_det_dataset(self.args.data)
        # NOTE: to make training work smoothly, set `nc` and `names`
        self.data = data
        self.data["nc"] = data["train"]["nc"]
        self.data["names"] = data["train"]["names"]
        return data["train"]["train"], data["val"].get("val") or data["val"].get("minival")

    def plot_training_labels(self):
        """DO NOT plot labels."""
        pass

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO model."""
        self.validator.args.data = self.args.data["val"]
        return super().final_eval()


if __name__ == "__main__":
    from ultralytics import YOLOWorld

    model = YOLOWorld("yolov8s-worldv2.yaml")
    data = dict(train="Objects365.yaml", val="lvis.yaml")
    model.train(data=data, batch=128, exist_ok=True, epochs=1, trainer=WorldTrainerFromScratch)
