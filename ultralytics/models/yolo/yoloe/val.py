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
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEDetectValidator(DetectionValidator):
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

        This function processes a dataloader to compute visual prompt embeddings for each class
        using a YOLOE model. It normalizes the embeddings and handles cases where no samples
        exist for a class.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader providing training samples.
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

        This function prepares a dataloader for visual prompt embeddings (VPE) using the LVIS dataset.
        It applies necessary transformations and configurations to the dataset and returns a dataloader
        for validation purposes.

        Args:
            data (dict): Dataset configuration dictionary containing paths and settings.

        Returns:
            (torch.utils.data.DataLoader): The dataLoader for visual prompt samples.
        """
        dataset = build_yolo_dataset(
            self.args,
            data.get(self.args.split, data.get("val")),
            self.args.batch,
            data,
            mode="val",
            rect=False,
        )
        if isinstance(dataset, YOLOConcatDataset):
            for d in dataset.datasets:
                d.transforms.append(LoadVisualPrompt())
        else:
            dataset.transforms.append(LoadVisualPrompt())
        return build_dataloader(
            dataset,
            self.args.batch,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None, refer_data=None, load_vp=False):
        """
        Run validation on the model using either text or visual prompt embeddings.

        This method validates the model using either text prompts or visual prompts, depending
        on the `load_vp` flag. It supports validation during training (using a trainer object)
        or standalone validation with a provided model.

        Args:
            trainer (object, optional): Trainer object containing the model and device.
            model (YOLOEModel, optional): Model to validate. Required if `trainer` is not provided.
            refer_data (str, optional): Path to reference data for visual prompts.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/")[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                # Directly use the same dataloader for visual embeddings extracted during training
                vpe = self.get_visual_pe(self.dataloader, model)
                model.set_classes(names, vpe)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
            stats = super().__call__(trainer, model)
        else:
            if refer_data is not None:
                assert load_vp, "Refer data is only used for visual prompt validation."
            self.device = select_device(self.args.device)

            if isinstance(model, str):
                from ultralytics.nn.tasks import attempt_load_weights

                model = attempt_load_weights(model, device=self.device, inplace=True)
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/")[0] for name in list(data["names"].values())]

            if load_vp:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                # TODO: need to check if the names from refer data is consistent with the evaluated dataset
                # could use same dataset or refer to extract visual prompt embeddings
                dataloader = self.get_vpe_dataloader(data)
                vpe = self.get_visual_pe(dataloader, model)
                model.set_classes(names, vpe)
                stats = super().__call__(model=deepcopy(model))
            elif isinstance(model.model[-1], YOLOEDetect) and hasattr(model.model[-1], "lrpc"):  # prompt-free
                return super().__call__(trainer, model)
            else:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                stats = super().__call__(model=deepcopy(model))
        return stats


class YOLOESegValidator(YOLOEDetectValidator, SegmentationValidator):
    """YOLOE segmentation validator that supports both text and visual prompt embeddings."""

    pass
