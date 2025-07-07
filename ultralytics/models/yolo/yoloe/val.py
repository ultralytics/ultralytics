# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from torch.nn import functional as F

from ultralytics.data import YOLOConcatDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.augment import LoadVisualPrompt
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.nn.modules.head import YOLOEDetect
from ultralytics.nn.tasks import YOLOEModel
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEDetectValidator(DetectionValidator):
    """
    A validator class for YOLOE detection models that handles both text and visual prompt embeddings.

    This class extends DetectionValidator to provide specialized validation functionality for YOLOE models.
    It supports validation using either text prompts or visual prompt embeddings extracted from training samples,
    enabling flexible evaluation strategies for prompt-based object detection.

    Attributes:
        device (torch.device): The device on which validation is performed.
        args (namespace): Configuration arguments for validation.
        dataloader (DataLoader): DataLoader for validation data.

    Methods:
        get_visual_pe: Extract visual prompt embeddings from training samples.
        preprocess: Preprocess batch data ensuring visuals are on the same device as images.
        get_vpe_dataloader: Create a dataloader for LVIS training visual prompt samples.
        __call__: Run validation using either text or visual prompt embeddings.

    Examples:
        Validate with text prompts
        >>> validator = YOLOEDetectValidator()
        >>> stats = validator(model=model, load_vp=False)

        Validate with visual prompts
        >>> stats = validator(model=model, refer_data="path/to/data.yaml", load_vp=True)
    """

    @smart_inference_mode()
    def get_visual_pe(self, dataloader: torch.utils.data.DataLoader, model: YOLOEModel) -> torch.Tensor:
        """
        Extract visual prompt embeddings from training samples.

        This method processes a dataloader to compute visual prompt embeddings for each class using a YOLOE model.
        It normalizes the embeddings and handles cases where no samples exist for a class by setting their
        embeddings to zero.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader providing training samples.
            model (YOLOEModel): The YOLOE model from which to extract visual prompt embeddings.

        Returns:
            (torch.Tensor): Visual prompt embeddings with shape (1, num_classes, embed_dim).
        """
        assert isinstance(model, YOLOEModel)
        names = [name.split("/", 1)[0] for name in list(dataloader.dataset.data["names"].values())]
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        # Count samples per class
        for batch in dataloader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        # Extract visual prompt embeddings
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

        # Normalize embeddings for classes with samples, set others to zero
        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def preprocess(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess batch data, ensuring visuals are on the same device as images."""
        batch = super().preprocess(batch)
        if "visuals" in batch:
            batch["visuals"] = batch["visuals"].to(batch["img"].device)
        return batch

    def get_vpe_dataloader(self, data: Dict[str, Any]) -> torch.utils.data.DataLoader:
        """
        Create a dataloader for LVIS training visual prompt samples.

        This method prepares a dataloader for visual prompt embeddings (VPE) using the specified dataset.
        It applies necessary transformations including LoadVisualPrompt and configurations to the dataset
        for validation purposes.

        Args:
            data (dict): Dataset configuration dictionary containing paths and settings.

        Returns:
            (torch.utils.data.DataLoader): The dataloader for visual prompt samples.
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
    def __call__(
        self,
        trainer: Optional[Any] = None,
        model: Optional[Union[YOLOEModel, str]] = None,
        refer_data: Optional[str] = None,
        load_vp: bool = False,
    ) -> Dict[str, Any]:
        """
        Run validation on the model using either text or visual prompt embeddings.

        This method validates the model using either text prompts or visual prompts, depending on the load_vp flag.
        It supports validation during training (using a trainer object) or standalone validation with a provided
        model. For visual prompts, reference data can be specified to extract embeddings from a different dataset.

        Args:
            trainer (object, optional): Trainer object containing the model and device.
            model (YOLOEModel | str, optional): Model to validate. Required if trainer is not provided.
            refer_data (str, optional): Path to reference data for visual prompts.
            load_vp (bool): Whether to load visual prompts. If False, text prompts are used.

        Returns:
            (dict): Validation statistics containing metrics computed during validation.
        """
        if trainer is not None:
            self.device = trainer.device
            model = trainer.ema.ema
            names = [name.split("/", 1)[0] for name in list(self.dataloader.dataset.data["names"].values())]

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

            if isinstance(model, (str, Path)):
                from ultralytics.nn.tasks import attempt_load_weights

                model = attempt_load_weights(model, device=self.device, inplace=True)
            model.eval().to(self.device)
            data = check_det_dataset(refer_data or self.args.data)
            names = [name.split("/", 1)[0] for name in list(data["names"].values())]

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
