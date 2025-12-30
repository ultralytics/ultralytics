# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.models import yolo
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, YAML
from ultralytics.models.yolo.stereo3ddet.visualize import plot_stereo_sample
from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetDataset
from ultralytics.models.yolo.stereo3ddet.model import Stereo3DDetModel
from ultralytics.data import build_dataloader
from ultralytics.data.stereo.target_improved import TargetGenerator as TargetGeneratorImproved

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
        # Disable plots and validation for custom stereo pipeline until a dedicated validator is implemented
        overrides.setdefault("plots", False)
        overrides.setdefault("val", False)
        super().__init__(cfg, overrides, _callbacks)
        # T204: Determine loss names after model is initialized (will be set in set_model_attributes or get_validator)
        # Don't set here as model may not be available yet

    def get_validator(self):
        """Return a Stereo3DDetValidator, currently extending DetectionValidator."""
        # T204: Determine loss names dynamically from model before creating validator
        self._determine_loss_names()
        return yolo.stereo3ddet.Stereo3DDetValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def _determine_loss_names(self):
        """Determine loss names dynamically from model's loss dictionary keys or loss_names attribute.
        
        Priority order:
        1. Check if model has loss_names attribute
        2. Check if model.core.criterion returns loss_dict with keys (StereoYOLOv11Wrapper)
        3. Fallback to hardcoded list if model structure unknown
        
        Sets self.loss_names as tuple matching DetectionTrainer pattern.
        """
        # Default loss names for stereo 3D detection (10 branches)
        # Order matches stereo_yolo_v11.py:677-688
        default_loss_names = (
            "heatmap_loss",
            "offset_loss",
            "bbox_size_loss",
            "lr_distance_loss",
            "right_width_loss",
            "dimensions_loss",
            "orientation_loss",
            "vertices_loss",
            "vertex_offset_loss",
            "vertex_dist_loss",
        )
        
        # Check if loss_names is already set to stereo 3D detection names (10 branches)
        if hasattr(self, "loss_names") and self.loss_names:
            if isinstance(self.loss_names, (tuple, list)) and len(self.loss_names) == 10:
                expected_set = set(default_loss_names)
                current_set = set(self.loss_names)
                if expected_set == current_set:
                    return  # Already set correctly to stereo loss names
        
        # Try to get from model
        if hasattr(self, "model") and self.model is not None:
            # Option 1: Check if model has loss_names attribute
            if hasattr(self.model, "loss_names") and self.model.loss_names:
                self.loss_names = tuple(self.model.loss_names) if isinstance(self.model.loss_names, (list, tuple)) else (self.model.loss_names,)
                return
            
            # Option 2: Check if model is StereoYOLOv11Wrapper and has core.criterion
            # The criterion returns loss_dict with keys matching default_loss_names
            if hasattr(self.model, "core") and hasattr(self.model.core, "criterion"):
                # StereoYOLOv11Wrapper uses StereoCenterNetLoss which returns loss_dict with 10 keys
                # We can use the default loss names directly
                self.loss_names = default_loss_names
                return
            
            # Option 3: Try to extract from model.loss() return value
            # Note: StereoYOLOv11Wrapper.loss() returns (total_loss, loss_items) not (total_loss, loss_dict)
            # So we need to check the model structure instead
            if hasattr(self.model, "core") and hasattr(self.model.core, "criterion"):
                # Already handled above
                pass
        
        # Option 4: Fallback to default loss names for stereo 3D detection
        self.loss_names = default_loss_names

    def progress_string(self):
        """T205: Return a formatted string showing training progress with dynamically determined loss branches.
        
        Follows DetectionTrainer pattern from detect/train.py:187-195.
        Format: ("\n" + "%11s" * (4 + len(self.loss_names))) % ("Epoch", "GPU_mem", *self.loss_names, "Instances", "Size")
        
        Returns:
            str: Formatted progress string with column headers.
        """
        # Ensure loss_names is determined
        if not hasattr(self, "loss_names") or not self.loss_names:
            self._determine_loss_names()
        
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def get_dataset(self) -> dict[str, Any]:
        """Parse stereo dataset YAML and return metadata for KITTIStereoDataset.

        This overrides the base implementation to avoid the default YOLO detection dataset checks
        and instead wire up paths/splits intended for the custom `KITTIStereoDataset` loader.

        Returns:
            dict: Dataset dictionary with fields used by the trainer and model.
        """
        # Handle None data for testing purposes
        if self.args.data is None:
            return {
                "names": {0: "Car", 1: "Pedestrian", 2: "Cyclist"},
                "nc": 3,
                "channels": 6,
            }
        
        # Load YAML if a path is provided; accept dicts directly
        data_cfg = self.args.data
        if isinstance(data_cfg, (str, Path)):
            data_cfg = YAML.load(str(data_cfg))

        if not isinstance(data_cfg, dict):
            raise RuntimeError("stereo3ddet: data must be a YAML path or dict")

        # Validate channels for stereo (must be 6 = left RGB + right RGB)
        channels = data_cfg.get("channels", 6)
        if channels != 6:
            raise ValueError(
                f"Stereo3DDet requires 6 input channels (left + right RGB), "
                f"but dataset config has channels={channels}. "
                f"Please set 'channels: 6' in your dataset YAML."
            )

        # Root path and splits
        root_path = data_cfg.get("path") or "."
        root = Path(str(root_path)).resolve()
        # Accept either directory-style train/val or txt; KITTIStereoDataset uses split names
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        # Names/nc fallback - use paper classes (3 classes: Car, Pedestrian, Cyclist)
        from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
        names = data_cfg.get("names") or get_paper_class_names()  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        nc = data_cfg.get("nc", len(names))

        # Extract mean dimensions if present in dataset config
        mean_dims = data_cfg.get("mean_dims")
        
        # Return a dict compatible with BaseTrainer expectations, plus stereo descriptors
        return {
            "yaml_file": str(self.args.data) if isinstance(self.args.data, (str, Path)) else None,
            "path": str(root),
            # Channels for model input (6 = left+right stacked)
            "channels": 6,
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
            "mean_dims": mean_dims,  # Mean dimensions per class [L, W, H] from dataset.yaml
        }

    def build_dataset(self, img_path, mode: str = "train", batch: int | None = None):
        """Build Stereo3DDetDataset when given our descriptor; fallback to detection dataset otherwise."""
        # If img_path is a stereo descriptor dict created in get_dataset
        desc = img_path if isinstance(img_path, dict) else self.data.get(mode)
        if isinstance(desc, dict) and desc.get("type") == "kitti_stereo":
            imgsz = getattr(self.args, "imgsz", 640)
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                imgsz_hw = (int(imgsz[0]), int(imgsz[1]))  # (H, W)
            else:
                imgsz_hw = (int(imgsz), int(imgsz))  # square fallback
            
            # Determine output_size from model if available, otherwise use default (8x downsampling)
            output_size = None
            if hasattr(self, "model") and self.model is not None:
                try:
                    with torch.no_grad():
                        dummy_img = torch.zeros(1, 6, imgsz_hw[0], imgsz_hw[1], device=self.device)
                        dummy_output = self.model(dummy_img)
                        if isinstance(dummy_output, dict):
                            sample_branch = dummy_output.get("heatmap", list(dummy_output.values())[0])
                            if sample_branch is not None:
                                _, _, output_h, output_w = sample_branch.shape
                                output_size = (output_h, output_w)
                except Exception:
                    # Fallback to default if model forward fails
                    pass
            
            # Get mean_dims from dataset config
            mean_dims = self.data.get("mean_dims")
            
            return Stereo3DDetDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", "train")),
                imgsz=imgsz_hw,
                names=self.data.get("names"),
                output_size=output_size,
                mean_dims=mean_dims,
            )
        # Otherwise, use the default detection dataset builder
        return super().build_dataset(img_path, mode=mode, batch=batch)

    def get_dataloader(self, dataset_path, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct dataloader using the stereo adapter dataset if applicable."""
        # Build our dataset (handles both stereo descriptor dict and path strings)
        dataset = self.build_dataset(dataset_path, mode=mode, batch=batch_size)

        # If using our adapter, build InfiniteDataLoader with its collate_fn via Ultralytics helper
        if isinstance(dataset, Stereo3DDetDataset):
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

    def get_model(
        self,
        cfg: str | Path | dict[str, Any] | None = None,
        weights: str | Path | None = None,
        verbose: bool = True,
    ) -> Stereo3DDetModel:
        """Build stereo 3D detection model from YAML config.

        Args:
            cfg (str | Path | dict, optional): Model configuration file path or dictionary.
            weights (str | Path, optional): Path to the model weights file.
            verbose (bool): Whether to display model information during initialization.

        Returns:
            (Stereo3DDetModel): Initialized stereo 3D detection model.
        """
        model = Stereo3DDetModel(
            cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1
        )
        if verbose and RANK == -1:
            LOGGER.info(
                f"Initialized Stereo3DDetModel with {self.data['nc']} classes and {self.data['channels']} input channels"
            )
        if weights:
            model.load(weights)
            if verbose and RANK == -1:
                LOGGER.info(f"Loaded weights from {weights}")

        return model

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        super().set_model_attributes()
        # T204: Determine loss names after model is set
        self._determine_loss_names()

    def preprocess_batch(self, batch):
        """Normalize 6-channel images to float [0,1] and move targets to device.
        
        Targets are now generated in the dataset's collate_fn, so we just need to
        move them to the device if they're not already there.
        """
        imgs = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = imgs.float() / 255.0
        
        # Move targets to device if present (generated by dataset)
        if "targets" in batch:
            batch["targets"] = {k: v.to(self.device, non_blocking=True) for k, v in batch["targets"].items()}
        
        return batch

    def _forward_train(self, batch):
        """Forward training pass for the stereo model, passing targets dict if present."""
        imgs = batch["img"]
        targets = batch.get("targets", None)
        out = self.model(imgs, targets=targets)
        if isinstance(out, dict) and "loss" in out:
            return out["loss"], out
        # Fallback for unexpected outputs
        import torch
        return out if isinstance(out, torch.Tensor) else torch.tensor(0.0, device=imgs.device), {"out": out}

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot left-image training samples (default) plus stereo pairs using the existing dataset layout.

        This supplements the default detection plot with a stereo visualization by loading the matching right image
        and labels from `self.data['path']`.
        """
        assert 'im_file' in batch, "im_file is required in batch"
        im_files = batch["im_file"]
        # Prepare up to 4 stereo previews per batch
        previews = min(4, len(im_files))
        canvas_list = []

        for i in range(previews):
            _6_channel_img = batch["img"][i]
            assert _6_channel_img.shape[0] == 6, "6-channel image is required"
            assert _6_channel_img.max() <= 1.0, "image is not normalized"
            assert _6_channel_img.min() >= 0.0, "image is not normalized"
            # convert to cpu and numpy
            _6_channel_img = _6_channel_img.cpu().numpy()
            # Batch images are stored as RGB; OpenCV drawing/saving expects BGR.
            left_img = (_6_channel_img[:3, :].transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1].copy()
            right_img = (_6_channel_img[3:, :].transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1].copy()
            labels = batch["labels"][i]
            L, R = plot_stereo_sample(left_img, right_img, labels, class_names=self.data["names"])
            panel = np.concatenate([L, R], axis=1)

            # Add filename to the top-left corner of each rendered stereo panel for easier debugging.
            filename = Path(str(im_files[i])).name
            if filename:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 1
                (tw, th), baseline = cv2.getTextSize(filename, font, font_scale, thickness)
                x = 6
                y = 6 + th
                pad = 3
                cv2.rectangle(
                    panel,
                    (x - pad, y - th - pad),
                    (x + tw + pad, y + baseline + pad),
                    (0, 0, 0),
                    thickness=-1,
                )
                cv2.putText(panel, filename, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

            canvas_list.append(panel)

        if canvas_list:
            grid = canvas_list[0]
            for c in canvas_list[1:]:
                grid = np.concatenate([grid, c], axis=0)
            out = self.save_dir / f"stereo_train_batch{ni}.jpg"
            cv2.imwrite(str(out), grid)

    def plot_training_labels(self) -> None:
        """Override default label-plotting which expects detection dataset internals.

        Our stereo adapter does not provide a global `dataset.labels` cache, so skip gracefully.
        """
        return