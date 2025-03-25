# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import deepcopy

import torch
from torch.nn import functional as F

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.model import YOLOEModel
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEValidatorMixin:
    """
    A mixin class for YOLOE model validation that handles both text and visual prompt embeddings.

    This mixin provides functionality to validate YOLOE models using either text or visual prompt embeddings.
    It includes methods for extracting visual prompt embeddings from samples, preprocessing batches, and
    running validation with different prompt types.

    Attributes:
        device (torch.device): The device on which validation is performed.
        args (namespace): Configuration arguments for validation.
        dataloader (DataLoader): DataLoader for validation data.
    """

    @smart_inference_mode()
    def get_visual_pe(self, dataloader, model):
        """
        Extract visual prompt embeddings from training samples.

        Args:
            model (YOLOEModel): The YOLOE model from which to extract visual prompt embeddings.

        Returns:
            (torch.Tensor): Visual prompt embeddings with shape (1, num_classes, embed_dim).
        """
        assert isinstance(model, YOLOEModel)
        names = [name.split("/")[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        pbar = TQDM(dataloader, total=len(dataloader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])  # (B, max_n, embed_dim)

            batch_idx = batch["batch_idx"]
            for i in range(preds.shape[0]):
                cls = batch["cls"][batch_idx == i].squeeze(-1).to(torch.int).unique(sorted=True)
                pad_cls = torch.ones(preds.shape[1], device=self.device) * -1
                pad_cls[: len(cls)] = cls
                for c in cls:
                    visual_pe[c] += preds[i][pad_cls == c].sum(0) / cls_visual_num[c]

        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def preprocess(self, batch):
        """Preprocess batch data, ensuring visuals are on the same device as images."""
        batch = super().preprocess(batch)
        if "visuals" in batch:
            batch["visuals"] = batch["visuals"].to(batch["img"].device)
        return batch

    def get_vpe_dataloader(self, data):
        """
        Create a dataloader for LVIS training visual prompt samples.

        Args:
            model (YOLOEModel): The YOLOE model for which to create the dataloader.

        Returns:
            (tuple): A tuple containing:
                - lvis_train_vps_loader (DataLoader): DataLoader for LVIS training visual prompt samples.
                - names (list): List of class names.
        """
        vps_data = check_det_dataset(data)
        dataset = build_yolo_dataset(
            self.args,
            vps_data.get(self.args.split, vps_data.get("val")),
            self.args.batch,
            vps_data,
            mode="val",
            rect=False,
        )
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        vps_loader = build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )
        return vps_loader

    def build_vp_dataset(self, img_path, mode="val", batch=None):
        """
        Build YOLO Dataset.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): `train` mode or `val` mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for `rect`.

        Returns:
            (Dataset): YOLO dataset.
        """
        dataset = super().build_dataset(img_path, mode, batch)
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return dataset

    def add_prefix_for_metric(self, stats, prefix):
        """
        Add a prefix to metric keys in stats dictionary.

        Args:
            stats (dict): Statistics dictionary containing metrics.
            prefix (str): Prefix to add to metric keys.

        Returns:
            (dict): Statistics dictionary with prefixed metric keys.
        """
        prefix_stats = {}
        for k, v in stats.items():
            if k.startswith("metrics"):
                prefix_stats[f"{prefix}_{k}"] = v
            else:
                prefix_stats[k] = v
        return prefix_stats

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, refer_data=None, load_vp=False):
        """
        Run validation on the model using either text or visual prompt embeddings.

        Args:
            trainer (object, optional): Trainer object containing the model and device.
            model (YOLOEModel, optional): Model to validate.

        Returns:
            (dict): Validation statistics.
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/")[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                # directly the same dataloader for visual embeddings extracting during training
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
                vp_stats = super().__call__(trainer, model)
                vp_stats = self.add_prefix_for_metric(vp_stats, "vp")
                stats = vp_stats
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                tp_stats = super().__call__(trainer, model)
                tp_stats = self.add_prefix_for_metric(tp_stats, "tp")
                stats = tp_stats
        else:
            if isinstance(model, YOLOEModel) and not hasattr(model, "pe"):
                self.device = select_device(self.args.device, self.args.batch)

                model.eval().to(self.device)
                data = check_det_dataset(self.args.data)
                names = [name.split("/")[0] for name in list(data["names"].values())]

                if load_vp:  # TODO
                    LOGGER.info("Validate using the visual prompt.")
                    self.args.half = False
                    # TODO: need to check if the names from refer data is consistent with the evaluted dataset
                    # could use same dataset or refer to extract visual prompt embeddings
                    dataloader = self.get_vpe_dataloader(refer_data or self.args.data)
                    vpe = self.get_visual_pe(dataloader, model)
                    model.set_classes(names, vpe)
                    vp_stats = super().__call__(model=deepcopy(model))
                    vp_stats = self.add_prefix_for_metric(vp_stats, "vp")
                    stats = vp_stats
                else:
                    LOGGER.info("Validate using the text prompt.")
                    tpe = model.get_text_pe(names)
                    model.set_classes(names, tpe)
                    tp_stats = super().__call__(model=deepcopy(model))
                    tp_stats = self.add_prefix_for_metric(tp_stats, "tp")
                    stats = tp_stats
            else:
                return super().__call__(trainer, model)
        return stats


class YOLOEDetectValidator(YOLOEValidatorMixin, DetectionValidator):
    """YOLOE detection validator that supports both text and visual prompt embeddings."""

    pass


class YOLOESegValidator(YOLOEValidatorMixin, SegmentationValidator):
    """YOLOE segmentation validator that supports both text and visual prompt embeddings."""

    pass


class YOLOEPEFreeDetectValidator(DetectionValidator):
    """YOLOE detection validator that doesn't require prompt embeddings."""

    def eval_json(self, stats):
        """Return stats without additional processing."""
        return stats
