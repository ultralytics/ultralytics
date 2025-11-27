# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG, YAML
from ultralytics.models.yolo.stereo3ddet.visualize import plot_stereo_sample
from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset
from ultralytics.data import build_dataloader


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

    def get_dataset(self) -> dict[str, Any]:
        """Parse stereo dataset YAML and return metadata for KITTIStereoDataset.

        This overrides the base implementation to avoid the default YOLO detection dataset checks
        and instead wire up paths/splits intended for the custom `KITTIStereoDataset` loader.

        Returns:
            dict: Dataset dictionary with fields used by the trainer and model.
        """
        # Load YAML if a path is provided; accept dicts directly
        data_cfg = self.args.data
        if isinstance(data_cfg, (str, Path)):
            data_cfg = YAML.load(str(data_cfg))

        if not isinstance(data_cfg, dict):
            raise RuntimeError("stereo3ddet: data must be a YAML path or dict")

        # Root path and splits
        root_path = data_cfg.get("path") or "."
        root = Path(str(root_path)).resolve()
        # Accept either directory-style train/val or txt; KITTIStereoDataset uses split names
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        # Names/nc fallback
        names = data_cfg.get("names") or {
            0: "Car",
            1: "Van",
            2: "Truck",
            3: "Pedestrian",
            4: "Person_sitting",
            5: "Cyclist",
            6: "Tram",
            7: "Misc",
        }
        nc = data_cfg.get("nc", len(names))

        # Return a dict compatible with BaseTrainer expectations, plus stereo descriptors
        return {
            "yaml_file": str(self.args.data) if isinstance(self.args.data, (str, Path)) else None,
            "path": str(root),
            # Signal to our get_dataloader/build_dataset that this is a stereo dataset
            "train": {"type": "kitti_stereo", "root": str(root), "split": train_split},
            "val": {"type": "kitti_stereo", "root": str(root), "split": val_split},
            "names": names,
            "nc": nc,
            # carry over optional stereo metadata if present
            "stereo": data_cfg.get("stereo", True),
            "image_size": data_cfg.get("image_size", [375, 1242]),
            "baseline": data_cfg.get("baseline"),
            "focal_length": data_cfg.get("focal_length"),
        }

    def build_dataset(self, img_path, mode: str = "train", batch: int | None = None):
        """Build Stereo3DDetAdapterDataset when given our descriptor; fallback to detection dataset otherwise."""
        # If img_path is a stereo descriptor dict created in get_dataset
        desc = img_path if isinstance(img_path, dict) else self.data.get(mode)
        if isinstance(desc, dict) and desc.get("type") == "kitti_stereo":
            imgsz = int(self.args.imgsz) if hasattr(self.args, "imgsz") else 640
            return Stereo3DDetAdapterDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", "train")),
                imgsz=imgsz,
                names=self.data.get("names"),
            )
        # Otherwise, use the default detection dataset builder
        return super().build_dataset(img_path, mode=mode, batch=batch)

    def get_dataloader(self, dataset_path, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader using the stereo adapter dataset if applicable."""
        # Build our dataset (handles both stereo descriptor dict and path strings)
        dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)

        # If using our adapter, build InfiniteDataLoader with its collate_fn via Ultralytics helper
        if isinstance(dataset, Stereo3DDetAdapterDataset):
            shuffle = mode == "train"
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
                pin_memory=True,
            )
        # Fallback to default detection dataloader
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

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
