"""Apply explicit class aliases during detection head transfer."""

import json
import os

from ultralytics.models.yolo.detect import DetectionTrainer


class ClsMapTrainer(DetectionTrainer):
    """Apply class aliases before pretrained head-row transfer."""

    def set_model_names_for_load(self, model):
        """Alias dataset class names to their manually mapped source names."""
        cls_map = json.loads(os.environ["PHASE2_CLS_MAP"])
        model.names = {i: cls_map.get(str(name).strip().lower(), name) for i, name in self.data["names"].items()}
        return model
