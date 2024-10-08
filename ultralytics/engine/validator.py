# Ultralytics YOLO 🚀, AGPL-3.0 license

import json
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data import ClassificationDataset, build_dataloader
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode


class BaseValidator:
    """
    BaseValidator.

    A base class for creating validators.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): Dataloader to use for validation.
        pbar (tqdm): Progress bar to update during validation.
        device (torch.device): Device to use for validation.
        callbacks (dict): Dictionary to store various callback functions.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
    """

    def __init__(self, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        """
        Initializes a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader): Dataloader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            pbar (tqdm.tqdm): Progress bar for displaying progress.
            args (SimpleNamespace): Configuration for the validator.
            _callbacks (dict): Dictionary to store various callback functions.
        """
        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.pbar = pbar
        self.device = None
        self.callbacks = _callbacks or callbacks.get_default_callbacks()
        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        self.plots = {}
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}
        self.training = False
        self.loss = None
        self.names = None
        self.metrics = None

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Executes validation process, running inference on dataloader and computing performance metrics."""
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device
            self.args.half = model.fp16
            imgsz = check_imgsz(self.args.imgsz, stride=model.stride)
            if not model.pt and not model.jit:
                self.args.batch = model.metadata.get("batch", 1)
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            else:
                self.data = check_cls_dataset(self.args.data, split=self.args.split)

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)
            model.eval()
            model.warmup(imgsz=(1 if model.pt else self.args.batch, 3, imgsz, imgsz))

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            with dt[0]:
                batch = self.preprocess(batch)

            with dt[1]:
                preds = model(batch["img"], augment=augment)

            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)
                stats = self.eval_json(stats)
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def preprocess(self, batch):
        """Preprocesses an input batch."""
        batch["img"] = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = batch["img"].half() if self.args.half else batch["img"].float()
        batch["cls"] = batch["cls"].to(self.device)
        return batch

    def postprocess(self, preds):
        """Postprocesses the predictions."""
        return preds

    def build_dataset(self, img_path):
        """Build dataset."""
        # Implement dataset building here, ensuring compatibility with u16 images
        return ClassificationDataset(root=img_path, args=self.args, augment=False, prefix=self.args.split)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        if dataset_path is None:
            raise FileNotFoundError("Validation dataset path is None. Please provide a valid dataset path.")
        dataset = self.build_dataset(dataset_path)
        return build_dataloader(dataset, batch_size, self.args.workers, rank=-1)

    def init_metrics(self, model):
        """Initialize performance metrics for the model."""
        # Initialize any metrics needed for validation
        self.names = model.names
        self.nc = len(self.names)
        self.targets = []
        self.pred = []
        # Initialize confusion matrix or any other metrics here if needed

    def update_metrics(self, preds, batch):
        """Updates metrics based on predictions and batch."""
        # Implement metric updates here
        # Example for classification:
        n5 = min(len(self.names), 5)
        self.pred.append(preds.argsort(1, descending=True)[:, :n5].type(torch.int32).cpu())
        self.targets.append(batch["cls"].type(torch.int32).cpu())

    def finalize_metrics(self):
        """Finalizes and returns all metrics."""
        # Finalize metrics computation
        pass

    def get_stats(self):
        """Returns statistics about the model's performance."""
        # Compute and return statistics
        # Example for classification:
        self.metrics = ClassifyMetrics()
        self.metrics.process(self.targets, self.pred)
        return self.metrics.results_dict

    def check_stats(self, stats):
        """Checks statistics."""
        # Check for NaNs or invalid values in stats
        if not all(np.isfinite(v) for v in stats.values()):
            raise ValueError("Invalid metric values detected during validation.")

    def print_results(self):
        """Prints the results of the model's predictions."""
        # Print the computed metrics
        # Example for classification:
        pf = "%22s" + "%11.3g" * len(self.metrics.keys)
        LOGGER.info(pf % ("all", self.metrics.top1, self.metrics.top5))

    def get_desc(self):
        """Get description for the progress bar."""
        # Return a description string for the progress bar
        return "Validation"

    def plot_val_samples(self, batch, ni):
        """Plots validation samples during training."""
        # Implement plotting without using PIL
        # For example, using OpenCV and matplotlib
        import cv2
        import matplotlib.pyplot as plt

        images = batch["img"].cpu().numpy()
        cls = batch["cls"].cpu().numpy()

        # Convert images from CHW to HWC and from BGR to RGB
        images = images.transpose(0, 2, 3, 1)  # NCHW to NHWC

        for i, img in enumerate(images):
            # Handle uint16 images
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = self.names[int(cls[i])]
            plt.imshow(img)
            plt.title(f"Label: {label}")
            plt.axis("off")
            plt.savefig(str(self.save_dir / f"val_batch{ni}_sample{i}.jpg"))
            plt.close()

    def plot_predictions(self, batch, preds, ni):
        """Plots model predictions on batch images."""
        # Similar to plot_val_samples but include predictions
        import cv2
        import matplotlib.pyplot as plt

        images = batch["img"].cpu().numpy()
        pred_labels = torch.argmax(preds, dim=1).cpu().numpy()

        # Convert images from CHW to HWC and from BGR to RGB
        images = images.transpose(0, 2, 3, 1)  # NCHW to NHWC

        for i, img in enumerate(images):
            # Handle uint16 images
            if img.dtype == np.uint16:
                img = (img / 65535.0 * 255).astype(np.uint8)
            else:
                img = img.astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            label = self.names[int(pred_labels[i])]
            plt.imshow(img)
            plt.title(f"Predicted: {label}")
            plt.axis("off")
            plt.savefig(str(self.save_dir / f"val_batch{ni}_pred{i}.jpg"))
            plt.close()

    def on_plot(self, name, data=None):
        """Registers plots (e.g., to be consumed in callbacks)."""
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def run_callbacks(self, event: str):
        """Runs all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)
