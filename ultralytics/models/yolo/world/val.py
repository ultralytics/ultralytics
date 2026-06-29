# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import torch

from ultralytics.data.utils import check_det_dataset
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.torch_utils import select_device


class WorldValidator(DetectionValidator):
    """A validator for YOLO-World models that sets dataset class names before validation.

    Open-vocabulary YOLO-World models default to 80 COCO classes, so validating on a dataset with different classes
    (e.g. LVIS) fails or yields zero metrics. This validator generates text embeddings for the dataset's class names so
    standalone `model.val()` works. During training, classes are set via the `on_pretrain_routine_end` callback instead.
    """

    def __call__(self, trainer=None, model=None):
        """Set dataset classes for standalone validation, then run validation."""
        if trainer is None:  # standalone val; training sets classes via on_pretrain_routine_end callback
            self.device = select_device(self.args.device, verbose=False)
            if not isinstance(model, torch.nn.Module):
                from ultralytics.nn.tasks import load_checkpoint

                model = load_checkpoint(model or self.args.model, device=self.device)[0]
            model.eval().to(self.device)
            names = [name.split("/", 1)[0] for name in check_det_dataset(self.args.data)["names"].values()]
            if list(model.names.values()) != names:  # regenerate prompts only if class order differs from dataset
                state = (model.names, model.txt_feats, model.model[-1].nc)  # restore after to avoid leak to caller
                model.set_classes(names, cache_clip_model=False)
                model.names = dict(enumerate(names))  # set_classes updates embeddings/nc but not names
                try:
                    return super().__call__(trainer, model)
                finally:
                    model.names, model.txt_feats, model.model[-1].nc = state
        return super().__call__(trainer, model)
