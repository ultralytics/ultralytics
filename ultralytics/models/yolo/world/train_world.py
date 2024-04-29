from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import de_parallel


class WorldTrainerFromScratch(WorldTrainer):
    """
    A class extending the WorldTrainer class for training a world model from scratch on open-set dataset.

    Example:
        ```python
        from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        from ultralytics import YOLOWorld

        data = dict(
            train=dict(
                yolo_data=["Objects365.yaml"],
                grounding_data=[
                    dict(
                        img_path="../datasets/flickr30k/images",
                        json_file="../datasets/flickr30k/final_flickr_separateGT_train.json",
                    ),
                    dict(
                        img_path="../datasets/GQA/images",
                        json_file="../datasets/GQA/final_mixed_train_no_coco.json",
                    ),
                ],
            ),
            val=dict(yolo_data=["lvis.yaml"]),
        )

        model = YOLOWorld("yolov8s-worldv2.yaml")
        model.train(data=data, trainer=WorldTrainerFromScratch)
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
            img_path (List[str] | str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`. Defaults to None.
        """
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        if mode == "train":
            dataset = [
                build_yolo_dataset(self.args, im_path, batch, self.data, stride=gs, multi_modal=True)
                if isinstance(im_path, str)
                else build_grounding(self.args, im_path["img_path"], im_path["json_file"], batch, stride=gs)
                for im_path in img_path
            ]
            return YOLOConcatDataset(dataset) if len(dataset) > 1 else dataset[0]
        else:
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataset(self):
        """
        Get train, val path from data dict if it exists.

        Returns None if data format is not recognized.
        """
        final_data = dict()
        data_yaml = self.args.data
        assert data_yaml.get("train", False)  # object365.yaml
        assert data_yaml.get("val", False)  # lvis.yaml
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:  # for lvis dataset
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in ["train", "val"]:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # save grounding data if there's one
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = [grounding_data] if not isinstance(grounding_data, list) else grounding_data
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
            final_data[s] += grounding_data
        # NOTE: to make training work properly, set `nc` and `names`
        final_data["nc"] = data["val"][0]["nc"]
        final_data["names"] = data["val"][0]["names"]
        self.data = final_data
        return final_data["train"], final_data["val"][0]

    def plot_training_labels(self):
        """DO NOT plot labels."""
        pass

    def final_eval(self):
        """Performs final evaluation and validation for object detection YOLO-World model."""
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()
