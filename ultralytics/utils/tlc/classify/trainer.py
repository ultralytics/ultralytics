# Ultralytics YOLO 🚀, 3LC Integration, AGPL-3.0 license
from __future__ import annotations

from ultralytics.models import yolo

from ultralytics.data import build_dataloader
from ultralytics.utils.tlc.constants import IMAGE_COLUMN_NAME, CLASSIFY_LABEL_COLUMN_NAME
from ultralytics.utils.tlc.classify.dataset import TLCClassificationDataset
from ultralytics.utils.tlc.engine.trainer import TLCTrainerMixin
from ultralytics.utils.tlc.classify.validator import TLCClassificationValidator
from ultralytics.utils.tlc.classify.utils import tlc_check_cls_dataset
from ultralytics.utils.tlc.utils import create_sampler
from ultralytics.utils.torch_utils import is_parallel, torch_distributed_zero_first

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
        return TLCClassificationDataset(
            table,
            args=self.args,
            augment=mode == "train",
            prefix=mode,
            image_column_name=self._image_column_name,
            label_column_name=self._label_column_name,
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
        """Returns PyTorch DataLoader with transforms to preprocess images for inference."""
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode)

        sampler = create_sampler(dataset.table, mode=mode, settings=self._settings, distributed=is_parallel(self.model))
        loader = build_dataloader(dataset, batch_size, self.args.workers, rank=rank, shuffle=mode=="train", sampler=sampler)
        # Attach inference transforms
        if mode != "train":
            if is_parallel(self.model):
                self.model.module.transforms = loader.dataset.torch_transforms
            else:
                self.model.transforms = loader.dataset.torch_transforms
        return loader
