# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from ultralytics.models.yolo.detect import DetectionTrainer

from ultralytics.data import build_dataloader
from ultralytics.utils import LOGGER
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME, DETECTION_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.detect.utils import build_tlc_yolo_dataset, tlc_check_det_dataset
from ultralytics.utils.tlc.detect.validator import TLCDetectionValidator
from ultralytics.utils.tlc.engine.trainer import TLCTrainerMixin
from ultralytics.utils.tlc.utils import create_sampler
from ultralytics.utils.torch_utils import de_parallel, torch_distributed_zero_first


class TLCDetectionTrainer(TLCTrainerMixin, DetectionTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = DETECTION_LABEL_COLUMN_NAME
    """Trainer class for YOLOv8 object detection with 3LC"""

    def get_dataset(self):
        # Parse yaml and create tables
        self.data = tlc_check_det_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            project_name=self._settings.project_name,
        )
        return self.data["train"], self.data.get("val") or self.data.get("test")

    def build_dataset(self, table, mode="train", batch=None):
        # Dataset object for training / validation
        gs = max(int(de_parallel(self.model).stride.max() if self.model else 0), 32)
        return build_tlc_yolo_dataset(self.args, table, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_validator(self, dataloader=None):
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        if not dataloader:
            dataloader = self.test_loader

        return TLCDetectionValidator(
            dataloader,
            save_dir=self.save_dir,
            args=self.args,
            run=self._run,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            settings=self._settings,
        )

    def _process_metrics(self, metrics):
        return {
            metric.strip("(B)").replace("metrics", "val").replace("/", "_"): value
            for metric, value in metrics.items()}

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Construct and return dataloader."""
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)

        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("WARNING ⚠️ 'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        workers = self.args.workers if mode == "train" else self.args.workers * 2

        sampler = create_sampler(dataset.table, mode, self._settings, distributed=rank != -1)

        return build_dataloader(dataset, batch_size, workers, shuffle, rank, sampler)  # return dataloader
