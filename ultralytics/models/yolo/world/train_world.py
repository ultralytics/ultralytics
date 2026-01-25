# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from pathlib import Path

from ultralytics.data import YOLOConcatDataset, build_grounding, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.world import WorldTrainer
from ultralytics.utils import DATASETS_DIR, DEFAULT_CFG, LOGGER
from ultralytics.utils.torch_utils import unwrap_model


class WorldTrainerFromScratch(WorldTrainer):
    """A class extending the WorldTrainer for training a world model from scratch on open-set datasets.

    This trainer specializes in handling mixed datasets including both object detection and grounding datasets,
    supporting training YOLO-World models with combined vision-language capabilities.

    Attributes:
        cfg (dict): Configuration dictionary with default parameters for model training.
        overrides (dict): Dictionary of parameter overrides to customize the configuration.
        _callbacks (list): List of callback functions to be executed during different stages of training.
        data (dict): Final processed data configuration containing train/val paths and metadata.
        training_data (dict): Dictionary mapping training dataset paths to their configurations.

    Methods:
        build_dataset: Build YOLO Dataset for training or validation with mixed dataset support.
        get_dataset: Get train and validation paths from data dictionary.
        plot_training_labels: Skip label plotting for YOLO-World training.
        final_eval: Perform final evaluation and validation for the YOLO-World model.

    Examples:
        >>> from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch
        >>> from ultralytics import YOLOWorld
        >>> data = dict(
        ...     train=dict(
        ...         yolo_data=["Objects365.yaml"],
        ...         grounding_data=[
        ...             dict(
        ...                 img_path="flickr30k/images",
        ...                 json_file="flickr30k/final_flickr_separateGT_train.json",
        ...             ),
        ...             dict(
        ...                 img_path="GQA/images",
        ...                 json_file="GQA/final_mixed_train_no_coco.json",
        ...             ),
        ...         ],
        ...     ),
        ...     val=dict(yolo_data=["lvis.yaml"]),
        ... )
        >>> model = YOLOWorld("yolov8s-worldv2.yaml")
        >>> model.train(data=data, trainer=WorldTrainerFromScratch)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a WorldTrainerFromScratch object.

        This initializes a trainer for YOLO-World models from scratch, supporting mixed datasets including both object
        detection and grounding datasets for vision-language capabilities.

        Args:
            cfg (dict): Configuration dictionary with default parameters for model training.
            overrides (dict, optional): Dictionary of parameter overrides to customize the configuration.
            _callbacks (list, optional): List of callback functions to be executed during different stages of training.
        """
        if overrides is None:
            overrides = {}
        super().__init__(cfg, overrides, _callbacks)

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset for training or validation.

        This method constructs appropriate datasets based on the mode and input paths, handling both standard YOLO
        datasets and grounding datasets with different formats.

        Args:
            img_path (list[str] | str): Path to the folder containing images or list of paths.
            mode (str): 'train' mode or 'val' mode, allowing customized augmentations for each mode.
            batch (int, optional): Size of batches, used for rectangular training/validation.

        Returns:
            (YOLOConcatDataset | Dataset): The constructed dataset for training or validation.
        """
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        if mode != "train":
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=False, stride=gs)
        datasets = [
            build_yolo_dataset(self.args, im_path, batch, self.training_data[im_path], stride=gs, multi_modal=True)
            if isinstance(im_path, str)
            else build_grounding(
                # assign `nc` from validation set to max number of text samples for training consistency
                self.args,
                im_path["img_path"],
                im_path["json_file"],
                batch,
                stride=gs,
                max_samples=self.data["nc"],
            )
            for im_path in img_path
        ]
        self.set_text_embeddings(datasets, batch)  # cache text embeddings to accelerate training
        return YOLOConcatDataset(datasets) if len(datasets) > 1 else datasets[0]
    def check_data_config(self):
        """
        Check and load the data configuration from a YAML file or dictionary.
        
        Returns:
            (dict): Data configuration dictionary loaded from YAML file or passed directly.
            
        Raises:
            TypeError: If data is not a string path or dictionary.
            FileNotFoundError: If the specified YAML file does not exist.
            ValueError: If the YAML file format is invalid.
        """
        # If already a dictionary, return as-is
        if isinstance(self.args.data, dict):
            LOGGER.info("Using data config from dictionary")
            return self.args.data
        
        # If string, load from YAML file
        if isinstance(self.args.data, str):
            data_path = Path(self.args.data)
            
            # If data is just a filename (no path), look in default datasets dir
            if not data_path.is_absolute() and data_path.parent == Path('.'):
                from ultralytics.utils import ROOT
                # Remove .yaml/.yml extension if present, then add .yaml
                stem = data_path.stem if data_path.suffix in {'.yaml', '.yml'} else data_path.name
                data_path = ROOT / 'cfg' / 'datasets' / f'{stem}.yaml'
            
            # Validate YAML file exists
            if not data_path.exists():
                raise FileNotFoundError(f"Data config file not found: {data_path}")
            
            # Validate YAML file extension
            if data_path.suffix not in {'.yaml', '.yml'}:
                raise ValueError(f"Data file must be YAML format (.yaml/.yml), got {data_path.suffix}")
            
            # Load YAML with error handling
            try:
                from ultralytics.utils import YAML
                data_yaml = YAML.load(str(data_path))
                
                # Validate loaded data is a dictionary
                if not isinstance(data_yaml, dict):
                    raise ValueError(f"YAML file must contain a dictionary, got {type(data_yaml).__name__}")
                
                LOGGER.info(f"Loaded data config from {data_path}")
                
                self.args.data=data_yaml  # update args.data to the loaded dict 
                return data_yaml
                
            except Exception as e:
                raise ValueError(f"Failed to load YAML file {data_path}: {e}")
        
        # Invalid type
        raise TypeError(
            f"data must be a YAML file path (str) or dict, "
            f"got {type(self.args.data).__name__}"
        )

    def get_dataset(self):
        """Get train and validation paths from data dictionary.

        Processes the data configuration to extract paths for training and validation datasets, handling both YOLO
        detection datasets and grounding datasets.

        Returns:
            train_path (str): Train dataset path.
            val_path (str): Validation dataset path.

        Raises:
            AssertionError: If train or validation datasets are not found, or if validation has multiple datasets.
        """
        final_data = {}
        data_yaml=self.check_data_config()
        assert data_yaml.get("train", False), "train dataset not found"  # object365.yaml
        assert data_yaml.get("val", False), "validation dataset not found"  # lvis.yaml
        data = {k: [check_det_dataset(d) for d in v.get("yolo_data", [])] for k, v in data_yaml.items()}
        assert len(data["val"]) == 1, f"Only support validating on 1 dataset for now, but got {len(data['val'])}."
        val_split = "minival" if "lvis" in data["val"][0]["val"] else "val"
        for d in data["val"]:
            if d.get("minival") is None:  # for lvis dataset
                continue
            d["minival"] = str(d["path"] / d["minival"])
        for s in {"train", "val"}:
            final_data[s] = [d["train" if s == "train" else val_split] for d in data[s]]
            # save grounding data if there's one
            grounding_data = data_yaml[s].get("grounding_data")
            if grounding_data is None:
                continue
            grounding_data = grounding_data if isinstance(grounding_data, list) else [grounding_data]
            for g in grounding_data:
                assert isinstance(g, dict), f"Grounding data should be provided in dict format, but got {type(g)}"
                for k in {"img_path", "json_file"}:
                    path = Path(g[k])
                    if not path.exists() and not path.is_absolute():
                        g[k] = str((DATASETS_DIR / g[k]).resolve())  # path relative to DATASETS_DIR
            final_data[s] += grounding_data
        # assign the first val dataset as currently only one validation set is supported
        data["val"] = data["val"][0]
        final_data["val"] = final_data["val"][0]
        # NOTE: to make training work properly, set `nc` and `names`
        final_data["nc"] = data["val"]["nc"]
        final_data["names"] = data["val"]["names"]
        # NOTE: add path with lvis path
        final_data["path"] = data["val"]["path"]
        final_data["channels"] = data["val"]["channels"]
        self.data = final_data
        if self.args.single_cls:  # consistent with base trainer
            LOGGER.info("Overriding class names with single class.")
            self.data["names"] = {0: "object"}
            self.data["nc"] = 1
        self.training_data = {}
        for d in data["train"]:
            if self.args.single_cls:
                d["names"] = {0: "object"}
                d["nc"] = 1
            self.training_data[d["train"]] = d
        return final_data

    def plot_training_labels(self):
        """Skip label plotting for YOLO-World training."""
        pass

    def final_eval(self):
        """Perform final evaluation and validation for the YOLO-World model.

        Configures the validator with appropriate dataset and split information before running evaluation.

        Returns:
            (dict): Dictionary containing evaluation metrics and results.
        """

        # Ensure self.args.data is a dict (should be after get_dataset call)
        assert isinstance(self.args.data, dict), "self.args.data should be a dict at this point"
        val = self.args.data["val"]["yolo_data"][0]
        self.validator.args.data = val
        self.validator.args.split = "minival" if isinstance(val, str) and "lvis" in val else "val"
        return super().final_eval()
