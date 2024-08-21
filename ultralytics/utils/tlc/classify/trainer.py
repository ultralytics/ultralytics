# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from ultralytics.models import yolo

from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME, CLASSIFY_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.classify.dataset import TLCClassificationDataset
from ultralytics.utils.tlc.engine.trainer import TLCTrainerMixin
from ultralytics.utils.tlc.classify.validator import TLCClassificationValidator
from ultralytics.utils.tlc.classify.utils import tlc_check_cls_dataset


class TLCClassificationTrainer(TLCTrainerMixin, yolo.classify.ClassificationTrainer):
    _default_image_column_name = IMAGE_COLUMN_NAME
    _default_label_column_name = CLASSIFY_LABEL_COLUMN_NAME

    def get_dataset(self):
        """ Overrides the get_dataset method to get or create 3LC tables.
        
        """
        self.data = tlc_check_cls_dataset(
            self.args.data,
            self._tables,
            self._image_column_name,
            self._label_column_name,
            project_name=self._settings.project_name,
        )
        return self.data["train"], self.data.get("val") or self.data.get("test")

    def build_dataset(self, table, mode="train", batch=None):
        exclude_zero_weight = self._settings.exclude_zero_weight_training if mode=="train" else self._settings.exclude_zero_weight_collection
        sampling_weights = self._settings.sampling_weights if mode == "train" else False
        return TLCClassificationDataset(
            table,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            exclude_zero_weight=exclude_zero_weight,
            sampling_weights=sampling_weights
        )
    
    def get_validator(self, dataloader=None):
        self.loss_names = ["loss"]
        dataloader = dataloader or self.test_loader
        return TLCClassificationValidator(
            dataloader,
            self.save_dir,
            _callbacks=self.callbacks,
            run=self._run,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
            settings=self._settings,
        )
    
    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode, shuffle=mode=="train")
