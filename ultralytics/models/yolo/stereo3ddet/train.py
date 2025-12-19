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
from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetAdapterDataset
from ultralytics.models.yolo.stereo3ddet.model import Stereo3DDetModel
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
        """T204: Determine loss names dynamically from model's loss dictionary keys or loss_names attribute.
        
        Priority order:
        1. Check if model has loss_names attribute
        2. Check if model.core.criterion returns loss_dict with keys (StereoYOLOv11Wrapper)
        3. Fallback to hardcoded list if model structure unknown
        
        Sets self.loss_names as tuple matching DetectionTrainer pattern.
        """
        # Default loss names for stereo 3D detection (10 branches)
        # Order matches stereo_yolo_v11.py:677-688
        default_loss_names = (
            "heatmap",
            "offset",
            "bbox_size",
            "lr_distance",
            "right_width",
            "dimensions",
            "orientation",
            "vertices",
            "vertex_offset",
            "vertex_dist",
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
        """Normalize 6-channel images to float [0,1] and generate targets."""
        imgs = batch["img"].to(self.device, non_blocking=True)
        batch["img"] = imgs.float() / 255.0
        
        # Generate targets from labels if present
        if "labels" in batch and batch["labels"]:
            from ultralytics.data.stereo.target import TargetGenerator
            from ultralytics.data.stereo.target_improved import TargetGenerator as TargetGeneratorImproved
            
            # Get image size (needed for both target generator initialization and target generation)
            # Use actual image dimensions if available, otherwise fall back to args.imgsz
            if imgs.shape[2] == imgs.shape[3]:  # Square image
                imgsz = imgs.shape[2]
            else:
                # For rectangular images, use the larger dimension
                imgsz = max(imgs.shape[2], imgs.shape[3])
            
            # Override with args.imgsz if available (for consistency with training config)
            if hasattr(self.args, "imgsz"):
                imgsz = int(self.args.imgsz)
            
            # Initialize target generator if not already done
            if not hasattr(self, "target_generator"):
                num_classes = self.data.get("nc", 3)
                
                # Get mean dimensions from dataset config if available
                mean_dims = self.data.get("mean_dims")
                # mean_dims from dataset.yaml is already in format {class_name: [L, W, H]}
                # which matches what TargetGeneratorImproved expects
                
                # Dynamically determine output size from model forward pass
                # This works for any architecture (P3, P4, P5, etc.) instead of hardcoding 32x
                # The model's actual output size depends on the YAML config (e.g., P3 = 8x downsampling)
                # We do a dummy forward pass to get the actual output shape, making it architecture-agnostic
                with torch.no_grad():
                    # Create dummy input with same shape as actual input
                    dummy_img = torch.zeros(1, 6, imgsz, imgsz, device=self.device)
                    # Forward pass to get actual output shape
                    dummy_output = self.model(dummy_img)
                    
                    # Extract output shape from predictions
                    # For stereo3ddet, output is a dict with 10 branches
                    if isinstance(dummy_output, dict):
                        # Get shape from any branch (all have same spatial dimensions)
                        sample_branch = dummy_output.get("heatmap", list(dummy_output.values())[0])
                        if sample_branch is not None:
                            _, _, output_h, output_w = sample_branch.shape
                        else:
                            # Fallback: try to get from model stride if available
                            if hasattr(self.model, "stride") and self.model.stride is not None:
                                stride = float(self.model.stride[0]) if isinstance(self.model.stride, torch.Tensor) else float(self.model.stride)
                                output_h = int(imgsz / stride)
                                output_w = int(imgsz / stride)
                            else:
                                # Last resort: assume 8x downsampling for P3 (common case)
                                output_h = imgsz // 8
                                output_w = imgsz // 8
                    else:
                        # If output is not a dict, try to infer from model structure
                        if hasattr(self.model, "stride") and self.model.stride is not None:
                            stride = float(self.model.stride[0]) if isinstance(self.model.stride, torch.Tensor) else float(self.model.stride)
                            output_h = int(imgsz / stride)
                            output_w = int(imgsz / stride)
                        else:
                            # Fallback: assume 8x downsampling for P3
                            output_h = imgsz // 8
                            output_w = imgsz // 8
                
                self.target_generator = TargetGeneratorImproved(
                    output_size=(output_h, output_w),
                    num_classes=num_classes,
                    mean_dims=mean_dims,  # Pass mean dimensions from dataset.yaml
                )
            
            # Convert labels to target format
            # The model expects targets to be a single dict (not a list)
            # We'll use the first sample's targets for now, or batch them properly
            # For proper batching, we need to stack targets across batch dimension
            targets_list = []
            for labels in batch["labels"]:
                target = self.target_generator.generate_targets(
                    labels, 
                    input_size=(imgsz, imgsz),  # Assuming square input
                    calib=batch.get("calib"),
                    original_size=batch.get("ori_shape"),
                )
                # Move to device
                target = {k: v.to(self.device) for k, v in target.items()}
                targets_list.append(target)
            
            # Stack targets across batch dimension
            # Each target is a dict with tensors of shape [C, H, W]
            # We need to stack to [B, C, H, W]
            batched_targets = {}
            for key in targets_list[0].keys():
                batched_targets[key] = torch.stack([t[key] for t in targets_list], dim=0)
            
            batch["targets"] = batched_targets
        
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

    def plot_training_labels(self) -> None:
        """Override default label-plotting which expects detection dataset internals.

        Our stereo adapter does not provide a global `dataset.labels` cache, so skip gracefully.
        """
        return

    # def final_eval(self):
    #     """Perform final evaluation and validation for stereo 3D detection model.
        
    #     Overrides BaseTrainer.final_eval() to convert Path to string for AutoBackend compatibility.
    #     This is required because AutoBackend expects a string path, not a Path object.
        
    #     Note: This override is constitution-compliant - we do not modify BaseTrainer.
    #     """
    #     model_path = self.best if self.best.exists() else None
    #     if isinstance(model_path, Path):
    #         # load the model from the path
    #         model = torch.load(model_path, weights_only=False)
    #         model_path = str(model_path)
    #     else:
    #         model_path = str(model_path)

    #     with torch_distributed_zero_first(LOCAL_RANK):
    #         if RANK in {-1, 0}:
    #             ckpt = strip_optimizer(self.last) if self.last.exists() else {}
    #             if model_path:
    #                 # update best.pt train_metrics from last.pt
    #                 strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
    #     if model_path:
    #         LOGGER.info(f"\nValidating {model_path}...")
    #         self.validator.args.plots = self.args.plots
    #         self.validator.args.compile = False  # disable final val compile as too slow
    #         # Convert Path to string for AutoBackend compatibility
    #         self.metrics = self.validator(model=model)
    #         self.metrics.pop("fitness", None)
    #         self.run_callbacks("on_fit_epoch_end")
