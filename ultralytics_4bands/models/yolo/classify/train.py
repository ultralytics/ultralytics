# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch
import torchvision

from ultralytics_4bands.data import ClassificationDataset, build_dataloader
from ultralytics_4bands.engine.trainer import BaseTrainer
from ultralytics_4bands.models import yolo
from ultralytics_4bands.nn.tasks import ClassificationModel, attempt_load_one_weight
from ultralytics_4bands.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics_4bands.utils.plotting import plot_images, plot_results
from ultralytics_4bands.utils.torch_utils import is_parallel, strip_optimizer, torch_distributed_zero_first


class ClassificationTrainer(BaseTrainer):
    """
    A class extending the BaseTrainer class for training based on a classification model.

    Notes:
        - Torchvision classification models can also be passed to the 'model' argument, i.e. model='resnet18'.

    Example:
        ```python
        from ultralytics_4bands.models.yolo.classify import ClassificationTrainer

        args = dict(model='yolov8n-cls.pt', data='imagenet10', epochs=3)
        trainer = ClassificationTrainer(overrides=args)
        trainer.train()
        ```
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize a ClassificationTrainer object with optional configuration overrides and callbacks."""
        if overrides is None:
            overrides = {}
        overrides["task"] = "classify"
        if overrides.get("imgsz") is None:
            overrides["imgsz"] = 224
        super().__init__(cfg, overrides, _callbacks)

    def set_model_attributes(self):
        """Set the YOLO model's class names from the loaded dataset."""
        self.model.names = self.data["names"]

    def modify_input_channels(self, model, new_channels):
        # Access the Conv module which contains the convolutional layer, batch normalization, and activation
        try:
            conv_module = model.model[0]
            index = 0
        except AttributeError:
            conv_module = model.model[1]
            index = 1

        # Access the actual Conv2d layer
        conv1 = conv_module.conv

        old_channels = conv1.in_channels
        if new_channels == old_channels:
            return model

        # Create a new Conv2d layer with the desired number of input channels
        new_conv = nn.Conv2d(new_channels, conv1.out_channels,
                             kernel_size=conv1.kernel_size,
                             stride=conv1.stride,
                             padding=conv1.padding,
                             bias=conv1.bias is not None)

        # Transfer the weights from the old conv layer to the new one
        with torch.no_grad():
            # Copy the weights for the first 3 channels
            new_conv.weight[:, :3, :, :].copy_(conv1.weight)
            # If there are more channels, initialize them
            if new_channels > 3:
                nn.init.xavier_uniform_(new_conv.weight[:, 3:, :, :])
            # If the original convolutional layer had a bias, copy it
            if conv1.bias is not None:
                new_conv.bias.copy_(conv1.bias)

        # Replace the convolutional layer within the Conv module
        conv_module.conv = new_conv

        model.model[index] = conv_module

        return model

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Returns a modified PyTorch model configured for training YOLO."""
        model = ClassificationModel(cfg, nc=self.data["nc"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)

        model = self.modify_input_channels(model, 4)
        for m in model.modules():
            if not self.args.pretrained and hasattr(m, "reset_parameters"):
                m.reset_parameters()
            if isinstance(m, torch.nn.Dropout) and self.args.dropout:
                m.p = self.args.dropout  # set dropout
        for p in model.parameters():
            p.requires_grad = True  # for training
        return model

    def setup_model(self):
        """Load, create or download model for any task."""
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        model, ckpt = str(self.model), None
        # Load a YOLO model locally, from torchvision, or from Ultralytics assets
        if model.endswith(".pt"):
            self.model, ckpt = attempt_load_one_weight(model, device="cpu")
            for p in self.model.parameters():
                p.requires_grad = True  # for training
        elif model.split(".")[-1] in ("yaml", "yml"):
            self.model = self.get_model(cfg=model)
        elif model in torchvision.models.__dict__:
            self.model = torchvision.models.__dict__[model](weights="IMAGENET1K_V1" if self.args.pretrained else None)
        else:
            FileNotFoundError(f"ERROR: model={model} not found locally or online. Please check model name.")
        ClassificationModel.reshape_outputs(self.model, self.data["nc"])

        return ckpt

    def build_dataset(self, img_path, mode="train", batch=None):
        """Creates a ClassificationDataset instance given an image path, and mode (train/test etc.)."""
        return ClassificationDataset(root=img_path, args=self.args, augment=mode == "train", prefix=mode)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode)

        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank)
        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader

    def preprocess_batch(self, batch):
        """Preprocesses a batch of images and classes."""
        batch["img"] = batch["img"].to(self.device)
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def progress_string(self):
        """Returns a formatted string showing training progress."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_validator(self):
        """Returns an instance of ClassificationValidator for validation."""
        self.loss_names = ["loss"]
        return yolo.classify.ClassificationValidator(self.test_loader, self.save_dir, _callbacks=self.callbacks)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Returns a loss dict with labelled training loss items tensor.

        Not needed for classification but necessary for segmentation & detection
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is None:
            return keys
        loss_items = [round(float(loss_items), 5)]
        return dict(zip(keys, loss_items))

    def plot_metrics(self):
        """Plots metrics from a CSV file."""
        plot_results(file=self.csv, classify=True, on_plot=self.on_plot)  # save results.png

    def final_eval(self):
        """Evaluate trained model and save validation results."""
        for f in self.last, self.best:
            if f.exists():
                strip_optimizer(f)  # strip optimizers
                if f is self.best:
                    LOGGER.info(f"\nValidating {f}...")
                    self.validator.args.data = self.args.data
                    self.validator.args.plots = self.args.plots
                    self.metrics = self.validator(model=f)
                    self.metrics.pop("fitness", None)
                    self.run_callbacks("on_fit_epoch_end")
        LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")

    def plot_training_samples(self, batch, ni):
        """Plots training samples with their annotations."""
        plot_images(
            images=batch["img"],
            batch_idx=torch.arange(len(batch["img"])),
            cls=batch["cls"].view(-1),  # warning: use .view(), not .squeeze() for Classify models
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )
