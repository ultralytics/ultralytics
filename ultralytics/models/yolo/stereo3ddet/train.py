# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG
from ultralytics.models.yolo.stereo3ddet.visualize import plot_stereo_sample


class Stereo3DDetTrainer(yolo.detect.DetectionTrainer):
    """Stereo 3D Detection trainer.

    Initial scaffolding that reuses the standard DetectionTrainer while setting task to 'stereo3ddet'.
    This enables `yolo train task=stereo3ddet` end-to-end, using default detection behaviors until
    a dedicated stereo 3D head/loss and dataset pipe are added.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "stereo3ddet"
        super().__init__(cfg, overrides, _callbacks)

    def get_validator(self):
        """Return a Stereo3DDetValidator, currently extending DetectionValidator."""
        return yolo.stereo3ddet.Stereo3DDetValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot left-image training samples (default) plus stereo pairs using the existing dataset layout.

        This supplements the default detection plot with a stereo visualization by loading the matching right image
        and labels from `self.data['path']`.
        """
        # Keep default detection visualization
        super().plot_training_samples(batch, ni)

        try:
            root = Path(self.data.get("path", ".")).resolve()
            split = "train"
            im_files = batch.get("im_file", [])
            if not im_files:
                return

            # Prepare up to 4 stereo previews per batch
            previews = min(4, len(im_files))
            canvas_list = []

            for i in range(previews):
                left_path = Path(im_files[i])
                image_id = left_path.stem
                right_path = root / "images" / split / "right" / f"{image_id}.png"
                label_path = root / "labels" / split / f"{image_id}.txt"

                left_img = cv2.imread(str(left_path))
                right_img = cv2.imread(str(right_path)) if right_path.exists() else None
                if left_img is None or right_img is None:
                    continue

                # Parse minimal stereo label (class_id xl yl wl hl xr yr wr hr ...)
                labels = []
                if label_path.exists():
                    with open(label_path, "r") as f:
                        for line in f:
                            p = line.strip().split()
                            if len(p) >= 9:
                                try:
                                    cls = int(float(p[0]))
                                except Exception:
                                    continue
                                labels.append(
                                    {
                                        "class_id": cls,
                                        "left_box": {
                                            "center_x": float(p[1]),
                                            "center_y": float(p[2]),
                                            "width": float(p[3]),
                                            "height": float(p[4]),
                                        },
                                        "right_box": {
                                            "center_x": float(p[5]),
                                            "width": float(p[7]),
                                        },
                                    }
                                )

                L, R = plot_stereo_sample(left_img, right_img, labels, class_names=self.data.get("names"))
                h = max(L.shape[0], R.shape[0])
                # Resize to same height if needed
                if L.shape[0] != R.shape[0]:
                    scale_L = h / L.shape[0]
                    scale_R = h / R.shape[0]
                    if L.shape[0] != h:
                        L = cv2.resize(L, (int(L.shape[1] * scale_L), h))
                    if R.shape[0] != h:
                        R = cv2.resize(R, (int(R.shape[1] * scale_R), h))
                canvas = np.concatenate([L, R], axis=1)
                canvas_list.append(canvas)

            if canvas_list:
                grid = canvas_list[0]
                for c in canvas_list[1:]:
                    grid = np.concatenate([grid, c], axis=0)
                out = self.save_dir / f"stereo_train_batch{ni}.jpg"
                cv2.imwrite(str(out), grid)
        except Exception:
            # Stay non-intrusive in case of any optional plotting error
            pass
