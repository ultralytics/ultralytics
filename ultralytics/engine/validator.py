# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Check a model's accuracy on a test or val split of a dataset.

Usage:
    $ yolo mode=val model=yolo26n.pt data=coco8.yaml imgsz=640

Usage - formats:
    $ yolo mode=val model=yolo26n.pt                 # PyTorch
                          yolo26n.torchscript        # TorchScript
                          yolo26n.onnx               # ONNX Runtime or OpenCV DNN with dnn=True
                          yolo26n_openvino_model     # OpenVINO
                          yolo26n.engine             # TensorRT
                          yolo26n.mlpackage          # CoreML (macOS-only)
                          yolo26n_saved_model        # TensorFlow SavedModel
                          yolo26n.pb                 # TensorFlow GraphDef
                          yolo26n.tflite             # TensorFlow Lite
                          yolo26n_edgetpu.tflite     # TensorFlow Edge TPU
                          yolo26n_paddle_model       # PaddlePaddle
                          yolo26n.mnn                # MNN
                          yolo26n_ncnn_model         # NCNN
                          yolo26n_imx_model          # Sony IMX
                          yolo26n_rknn_model         # Rockchip RKNN
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.utils import LOGGER, RANK, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.checks import check_imgsz
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import attempt_compile, select_device, smart_inference_mode, unwrap_model


class BaseValidator:
    """A base class for creating validators.

    This class provides the foundation for validation processes, including model evaluation, metric computation, and
    result visualization.

    Attributes:
        args (SimpleNamespace): Configuration for the validator.
        dataloader (DataLoader): DataLoader to use for validation.
        model (nn.Module): Model to validate.
        data (dict): Data dictionary containing dataset information.
        device (torch.device): Device to use for validation.
        batch_i (int): Current batch index.
        training (bool): Whether the model is in training mode.
        names (dict): Class names mapping.
        seen (int): Number of images seen so far during validation.
        stats (dict): Statistics collected during validation.
        confusion_matrix: Confusion matrix for classification evaluation.
        nc (int): Number of classes.
        iouv (torch.Tensor): IoU thresholds from 0.50 to 0.95 in spaces of 0.05.
        jdict (list): List to store JSON validation results.
        speed (dict): Dictionary with keys 'preprocess', 'inference', 'loss', 'postprocess' and their respective batch
            processing times in milliseconds.
        save_dir (Path): Directory to save results.
        plots (dict): Dictionary to store plots for visualization.
        callbacks (dict): Dictionary to store various callback functions.
        stride (int): Model stride for padding calculations.
        loss (torch.Tensor): Accumulated loss during training validation.

    Methods:
        __call__: Execute validation process, running inference on dataloader and computing performance metrics.
        match_predictions: Match predictions to ground truth objects using IoU.
        add_callback: Append the given callback to the specified event.
        run_callbacks: Run all callbacks associated with a specified event.
        get_dataloader: Get data loader from dataset path and batch size.
        build_dataset: Build dataset from image path.
        preprocess: Preprocess an input batch.
        postprocess: Postprocess the predictions.
        init_metrics: Initialize performance metrics for the YOLO model.
        update_metrics: Update metrics based on predictions and batch.
        finalize_metrics: Finalize and return all metrics.
        get_stats: Return statistics about the model's performance.
        print_results: Print the results of the model's predictions.
        get_desc: Get description of the YOLO model.
        on_plot: Register plots for visualization.
        plot_val_samples: Plot validation samples during training.
        plot_predictions: Plot YOLO model predictions on batch images.
        pred_to_json: Convert predictions to JSON format.
        eval_json: Evaluate and return JSON format of prediction statistics.
    """

    def __init__(self, dataloader=None, save_dir=None, args=None, _callbacks=None):
        """Initialize a BaseValidator instance.

        Args:
            dataloader (torch.utils.data.DataLoader, optional): DataLoader to be used for validation.
            save_dir (Path, optional): Directory to save results.
            args (SimpleNamespace, optional): Configuration for the validator.
            _callbacks (dict, optional): Dictionary to store various callback functions.
        """
        import torchvision  # noqa (import here so torchvision import time not recorded in postprocess time)

        self.args = get_cfg(overrides=args)
        self.dataloader = dataloader
        self.stride = None
        self.data = None
        self.device = None
        self.batch_i = None
        self.training = True
        self.names = None
        self.seen = None
        self.stats = None
        self.confusion_matrix = None
        self.nc = None
        self.iouv = None
        self.jdict = None
        self.speed = {"preprocess": 0.0, "inference": 0.0, "loss": 0.0, "postprocess": 0.0}

        self.save_dir = save_dir or get_save_dir(self.args)
        (self.save_dir / "labels" if self.args.save_txt else self.save_dir).mkdir(parents=True, exist_ok=True)
        if self.args.conf is None:
            self.args.conf = 0.01 if self.args.task == "obb" else 0.001  # reduce OBB val memory usage
        self.args.imgsz = check_imgsz(self.args.imgsz, max_dim=1)

        self.plots = {}
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            if trainer.args.compile and hasattr(model, "_orig_mod"):
                model = model._orig_mod  # validate non-compiled original model to avoid issues
            model = model.half() if self.args.half else model.float()
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            if hasattr(model, "end2end"):
                if self.args.end2end is not None:
                    model.end2end = self.args.end2end
                if model.end2end:
                    model.set_head_attr(max_det=self.args.max_det, agnostic_nms=self.args.agnostic_nms)
            model = AutoBackend(
                model=model or self.args.model,
                device=select_device(self.args.device) if RANK == -1 else torch.device("cuda", RANK),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit = model.stride, model.pt, model.jit
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if not (pt or jit or getattr(model, "dynamic", False)):
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not (pt or (getattr(model, "dynamic", False) and not model.imx)):
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            if self.args.compile:
                model = attempt_compile(model, device=self.device)
            model.warmup(imgsz=(1 if pt else self.args.batch, self.data["channels"], imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(unwrap_model(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds = self.postprocess(preds)

            self.update_metrics(preds, batch)
            if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")

        stats = {}
        self.gather_stats()
        if RANK in {-1, 0}:
            stats = self.get_stats()
            self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
            self.finalize_metrics()
            self.print_results()
            self.run_callbacks("on_val_end")

        if self.training:
            model.float()
            # Reduce loss across all GPUs
            loss = self.loss.clone().detach()
            if trainer.world_size > 1:
                dist.reduce(loss, dst=0, op=dist.ReduceOp.AVG)
            if RANK > 0:
                return
            results = {**stats, **trainer.label_loss_items(loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            if RANK > 0:
                return stats
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def match_predictions(
        self, pred_classes: torch.Tensor, true_classes: torch.Tensor, iou: torch.Tensor, use_scipy: bool = False
    ) -> torch.Tensor:
        """Match predictions to ground truth objects using IoU.

        Args:
            pred_classes (torch.Tensor): Predicted class indices of shape (N,).
            true_classes (torch.Tensor): Target class indices of shape (M,).
            iou (torch.Tensor): An NxM tensor containing the pairwise IoU values for predictions and ground truth.
            use_scipy (bool, optional): Whether to use scipy for matching (more precise).

        Returns:
            (torch.Tensor): Correct tensor of shape (N, 10) for 10 IoU thresholds.
        """
        # Dx10 matrix, where D - detections, 10 - IoU thresholds
        correct = np.zeros((pred_classes.shape[0], self.iouv.shape[0])).astype(bool)
        # LxD matrix where L - labels (rows), D - detections (columns)
        correct_class = true_classes[:, None] == pred_classes
        iou = iou * correct_class  # zero out the wrong classes
        iou = iou.cpu().numpy()
        for i, threshold in enumerate(self.iouv.cpu().tolist()):
            if use_scipy:
                # WARNING: known issue that reduces mAP in https://github.com/ultralytics/ultralytics/pull/4708
                import scipy  # scope import to avoid importing for all commands

                cost_matrix = iou * (iou >= threshold)
                if cost_matrix.any():
                    labels_idx, detections_idx = scipy.optimize.linear_sum_assignment(cost_matrix)
                    valid = cost_matrix[labels_idx, detections_idx] > 0
                    if valid.any():
                        correct[detections_idx[valid], i] = True
            else:
                matches = np.nonzero(iou >= threshold)  # IoU > threshold and classes match
                matches = np.array(matches).T
                if matches.shape[0]:
                    if matches.shape[0] > 1:
                        matches = matches[iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                        matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                        matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                    correct[matches[:, 1].astype(int), i] = True
        return torch.tensor(correct, dtype=torch.bool, device=pred_classes.device)

    def add_callback(self, event: str, callback):
        """Append the given callback to the specified event."""
        self.callbacks[event].append(callback)

    def run_callbacks(self, event: str):
        """Run all callbacks associated with a specified event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def get_dataloader(self, dataset_path, batch_size):
        """Get data loader from dataset path and batch size."""
        raise NotImplementedError("get_dataloader function not implemented for this validator")

    def build_dataset(self, img_path):
        """Build dataset from image path."""
        raise NotImplementedError("build_dataset function not implemented in validator")

    def preprocess(self, batch):
        """Preprocess an input batch."""
        return batch

    def postprocess(self, preds):
        """Postprocess the predictions."""
        return preds

    def init_metrics(self, model):
        """Initialize performance metrics for the YOLO model."""
        pass

    def update_metrics(self, preds, batch):
        """Update metrics based on predictions and batch."""
        pass

    def finalize_metrics(self):
        """Finalize and return all metrics."""
        pass

    def get_stats(self):
        """Return statistics about the model's performance."""
        return {}

    def gather_stats(self):
        """Gather statistics from all the GPUs during DDP training to GPU 0."""
        pass

    def print_results(self):
        """Print the results of the model's predictions."""
        pass

    def get_desc(self):
        """Get description of the YOLO model."""
        pass

    @property
    def metric_keys(self):
        """Return the metric keys used in YOLO training/validation."""
        return []

    def on_plot(self, name, data=None):
        """Register plots for visualization, deduplicating by type."""
        plot_type = data.get("type") if data else None
        if plot_type and any((v.get("data") or {}).get("type") == plot_type for v in self.plots.values()):
            return  # Skip duplicate plot types
        self.plots[Path(name)] = {"data": data, "timestamp": time.time()}

    def plot_val_samples(self, batch, ni):
        """Plot validation samples during training."""
        pass

    def plot_predictions(self, batch, preds, ni):
        """Plot YOLO model predictions on batch images."""
        pass

    def pred_to_json(self, preds, batch):
        """Convert predictions to JSON format."""
        pass

    def eval_json(self, stats):
        """Evaluate and return JSON format of prediction statistics."""
        pass
