# Ultralytics YOLO 🚀, AGPL-3.0 license

from copy import deepcopy

import torch
from torch.nn import functional as F

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.models.yolo.model import YOLOEModel
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import LOGGER, TQDM
from ultralytics.utils.torch_utils import select_device, smart_inference_mode


class YOLOEValidatorMixin:
    @smart_inference_mode()
    def get_visual_pe(self, model):
        assert isinstance(model, YOLOEModel)
        data_loader, names = self.get_lvis_train_vps_loader(model)
        visual_pe = torch.zeros(len(names), model.model[-1].embed, device=self.device)
        cls_visual_num = torch.zeros(len(names))

        desc = "Get visual prompt embeddings from samples"

        for batch in data_loader:
            cls = batch["cls"].squeeze(-1).to(torch.int).unique()
            count = torch.bincount(cls, minlength=len(names))
            cls_visual_num += count

        cls_visual_num = cls_visual_num.to(self.device)

        pbar = TQDM(data_loader, total=len(data_loader), desc=desc)
        for batch in pbar:
            batch = self.preprocess(batch)
            preds = model.get_visual_pe(batch["img"], visual=batch["visuals"])
            assert preds.shape[0] == 1

            cls = batch["cls"].squeeze(-1).to(torch.int).unique(sorted=True)
            assert len(cls) == 1 and preds.shape[1] == 1
            visual_pe[cls] += preds[0] / cls_visual_num[cls]

        visual_pe[cls_visual_num != 0] = F.normalize(visual_pe[cls_visual_num != 0], dim=-1, p=2)
        visual_pe[cls_visual_num == 0] = 0
        return visual_pe.unsqueeze(0)

    def preprocess(self, batch):
        batch = super().preprocess(batch)
        if "visuals" in batch:
            batch["visuals"] = batch["visuals"].to(batch["img"].device)
        return batch

    def get_lvis_train_vps_loader(self, model):
        lvis_train_vps_data = check_det_dataset("lvis_train_vps.yaml")
        lvis_train_vps_loader = build_dataloader(
            build_yolo_dataset(
                self.args,
                lvis_train_vps_data.get("val"),
                1,
                lvis_train_vps_data,
                mode="val",
                stride=max(int(model.stride.max()), 32),
                rect=False,
                load_vp=True,
            ),
            1,
            self.args.workers,
            shuffle=False,
            rank=-1,
        )
        return lvis_train_vps_loader, lvis_train_vps_data["names"]

    def add_prefix_for_metric(self, stats, prefix):
        prefix_stats = {}
        for k, v in stats.items():
            if k.startswith("metrics"):
                prefix_stats[f"{prefix}_{k}"] = v
            else:
                prefix_stats[k] = v
        return prefix_stats

    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        if trainer is not None:
            self.device = trainer.device

            model = trainer.ema.ema
            assert isinstance(model, YOLOEModel)
            assert not model.training

            names = [name.split("/")[0] for name in list(self.dataloader.dataset.data["names"].values())]

            if not self.args.load_vp:
                LOGGER.info("Validate using the text prompt.")
                tpe = model.get_text_pe(names)
                model.set_classes(names, tpe)
                tp_stats = super().__call__(trainer, model)
                tp_stats = self.add_prefix_for_metric(tp_stats, "tp")
                stats = tp_stats
            else:
                LOGGER.info("Validate using the visual prompt.")
                self.args.half = False
                vpe = self.get_visual_pe(model)
                model.set_classes(names, vpe)
                vp_stats = super().__call__(trainer, model)
                vp_stats = self.add_prefix_for_metric(vp_stats, "vp")
                stats = vp_stats

            return stats
        else:
            if isinstance(model, YOLOEModel) and not hasattr(model, "pe"):
                self.device = select_device(self.args.device, self.args.batch)

                model.eval().to(self.device)
                data = check_det_dataset(self.args.data)
                names = [name.split("/")[0] for name in list(data["names"].values())]

                if not self.args.load_vp:
                    LOGGER.info("Validate using the text prompt.")
                    tpe = model.get_text_pe(names)
                    model.set_classes(names, tpe)
                    tp_stats = super().__call__(trainer, deepcopy(model))
                    tp_stats = self.add_prefix_for_metric(tp_stats, "tp")
                    stats = tp_stats
                else:
                    LOGGER.info("Validate using the visual prompt.")
                    self.args.half = False
                    vpe = self.get_visual_pe(model)
                    model.set_classes(names, vpe)
                    vp_stats = super().__call__(trainer, deepcopy(model))
                    vp_stats = self.add_prefix_for_metric(vp_stats, "vp")
                    stats = vp_stats

                return stats
            else:
                return super().__call__(trainer, model)


class YOLOEDetectValidator(YOLOEValidatorMixin, DetectionValidator):
    pass


class YOLOESegValidator(YOLOEValidatorMixin, SegmentationValidator):
    pass


class YOLOEPEFreeDetectValidator(DetectionValidator):
    def eval_json(self, stats):
        return stats
