# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import re
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data import build_dataloader
from ultralytics.models import yolo
from ultralytics.models.yolo.stereo3ddet.dataset import Stereo3DDetDataset
from ultralytics.models.yolo.stereo3ddet.model import Stereo3DDetModel
from ultralytics.models.yolo.stereo3ddet.preprocess import preprocess_stereo_batch
from ultralytics.models.yolo.stereo3ddet.visualize import labels_to_box3d
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, ROOT, YAML
from ultralytics.utils.checks import check_yaml
from ultralytics.utils.plotting import Annotator, VisualizationConfig, colors, plot_labels, plot_stereo3d_boxes
from ultralytics.utils.torch_utils import intersect_dicts


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
        """Set loss names for stereo 3D detection (disparity + dimensions)."""
        self.loss_names = ("box", "cls", "dfl", "lr_dist", "depth", "dims", "orient")

    def progress_string(self):
        """Return a formatted string showing training progress with dynamically determined loss branches.
        
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

        This leverages check_det_dataset() for path resolution and automatic downloads,
        then transforms the result into stereo-specific format for our custom dataset loader.

        Returns:
            dict: Dataset dictionary with fields used by the trainer and model.
        """


        # Use check_det_dataset for path resolution, validation, and automatic download
        # This handles: finding default configs, executing download scripts, resolving paths
        from ultralytics.data.utils import check_det_dataset
        data_cfg = check_det_dataset(self.args.data, autodownload=True)

        # Validate channels for stereo (must be 6 = left RGB + right RGB)
        channels = data_cfg.get("channels", 6)
        if channels != 6:
            raise ValueError(
                f"Stereo3DDet requires 6 input channels (left + right RGB), "
                f"but dataset config has channels={channels}. "
                f"Please set 'channels: 6' in your dataset YAML."
            )

        # Root path and splits
        root = Path(data_cfg["path"])
        # Accept either directory-style train/val or txt; KITTIStereoDataset uses split names
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        # Names/nc fallback - use paper classes (3 classes: Car, Pedestrian, Cyclist)
        from ultralytics.models.yolo.stereo3ddet.utils import get_paper_class_names
        names = data_cfg.get("names") or get_paper_class_names()  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        nc = data_cfg.get("nc", len(names))

        # Extract mean dimensions if present in dataset config
        mean_dims = data_cfg.get("mean_dims")
        
        # Extract standard deviation of dimensions if present in dataset config
        std_dims = data_cfg.get("std_dims")
        
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
            "std_dims": std_dims,  # Standard deviation of dimensions per class [L, W, H] from dataset.yaml
        }

    def build_dataset(self, img_path, mode: str = "train", batch: int | None = None):
        """Build Stereo3DDetDataset when given our descriptor; fallback to detection dataset otherwise.

        TODO: Remove this method once the base trainer delegates val dataloader creation to the validator.
        """
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
                            # YOLO11 returns "det" (list) + aux branches (tensors)
                            # Use first aux branch or "det" to get output size
                            sample_branch = None
                            if "det" in dummy_output:
                                det_out = dummy_output["det"]
                                if isinstance(det_out, list) and len(det_out) > 0:
                                    sample_branch = det_out[0]
                            if sample_branch is None:
                                # Fallback to first aux branch
                                aux_branches = [k for k in dummy_output.keys() if k != "det"]
                                if aux_branches:
                                    sample_branch = dummy_output[aux_branches[0]]
                            if sample_branch is not None and len(sample_branch.shape) >= 2:
                                _, _, output_h, output_w = sample_branch.shape
                                output_size = (output_h, output_w)
                except Exception:
                    # Fallback to default if model forward fails
                    pass
            
            # Get mean_dims from dataset config
            mean_dims = self.data.get("mean_dims")
            std_dims = self.data.get("std_dims")
            return Stereo3DDetDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", "train")),
                imgsz=imgsz_hw,
                names=self.data.get("names"),
                output_size=output_size,
                mean_dims=mean_dims,
                std_dims=std_dims,
                augment=(mode == "train"),
                hyp=self.args
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

        Uses shared preprocessing from preprocess.py for consistency with validator.
        Training always uses full precision (half=False).
        """
        return preprocess_stereo_batch(batch, self.device, half=False)

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples as a 2-column grid for clarity.

        Layout (per row/sample):
        - Left column: LEFT image with 2D boxes
        - Right column: LEFT image with 3D wireframes (projected), not the right-camera image
        """
        assert 'im_file' in batch, "im_file is required in batch"
        im_files = batch["im_file"]
        calibs = batch.get("calib", None)
        # Prepare up to 4 stereo previews per batch
        previews = min(4, len(im_files))
        canvas_list = []

        def _add_title(img: np.ndarray, title: str) -> np.ndarray:
            """Add a small title banner to the top-left of an image (BGR)."""
            out = img.copy()
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), baseline = cv2.getTextSize(title, font, font_scale, thickness)
            pad = 6
            x, y = 8, 8 + th
            cv2.rectangle(out, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
            cv2.putText(out, title, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            return out

        for i in range(previews):
            _6_channel_img = batch["img"][i]
            assert _6_channel_img.shape[0] == 6, "6-channel image is required"
            assert _6_channel_img.max() <= 1.0, "image is not normalized"
            assert _6_channel_img.min() >= 0.0, "image is not normalized"
            # convert to cpu and numpy
            _6_channel_img = _6_channel_img.cpu().numpy()
            # Batch images are stored as RGB; OpenCV drawing/saving expects BGR.
            left_img = (_6_channel_img[:3, :].transpose(1, 2, 0) * 255).astype(np.uint8)[..., ::-1].copy()
            labels = batch["labels"][i]
            calib_i = None
            if isinstance(calibs, (list, tuple)) and i < len(calibs):
                calib_i = calibs[i]

            # ------------------------------------------------------------------
            # Left column: 2D boxes on LEFT image (using Annotator)
            # ------------------------------------------------------------------
            H, W = left_img.shape[:2]
            names = self.data["names"]
            annotator = Annotator(left_img.copy(), line_width=2, font_size=12, example=str(names))
            for lab in labels:
                lb = lab["left_box"]
                cls_id = int(lab["class_id"])
                cx_px = float(lb["center_x"]) * W
                cy_px = float(lb["center_y"]) * H
                bw = float(lb["width"]) * W
                bh = float(lb["height"]) * H
                box = [cx_px - bw / 2, cy_px - bh / 2, cx_px + bw / 2, cy_px + bh / 2]
                annotator.box_label(box, names.get(cls_id, str(cls_id)), color=colors(cls_id, True))
            L2 = _add_title(annotator.result(), "2D (left)")

            # ------------------------------------------------------------------
            # Right column: 3D wireframes on a separate LEFT-image view (cleaner)
            # ------------------------------------------------------------------
            L3 = left_img.copy()
            if isinstance(calib_i, dict):
                boxes3d = labels_to_box3d(
                    labels=labels,
                    calib=calib_i,
                    image_hw=L3.shape[:2],
                    class_names=self.data["names"],
                )
                if boxes3d:
                    class_ids = {int(b.class_id) for b in boxes3d}
                    magenta = (255, 0, 255)  # BGR
                    scheme = {cid: magenta for cid in class_ids}
                    cfg = VisualizationConfig(
                        line_width=2,
                        font_size=0.5,
                        show_labels=True,
                        show_conf=False,
                        gt_color_scheme=scheme,
                    )
                    L3, _, _ = plot_stereo3d_boxes(
                        left_img=L3,
                        right_img=L3.copy(),  # dummy
                        pred_boxes3d=None,
                        gt_boxes3d=boxes3d,
                        left_calib=calib_i,
                        right_calib=calib_i,
                        config=cfg,
                        letterbox_scale=None,
                        letterbox_pad_left=None,
                        letterbox_pad_top=None,
                    )
            L3 = _add_title(L3, "3D (proj)")

            panel = np.concatenate([L2, L3], axis=1)

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
        """Plot training label statistics for stereo3ddet.

        The default detection implementation expects a YOLODetectionDataset-style `dataset.labels` cache.
        Our stereo dataset does not provide that cache, so we build the arrays by scanning label files.

        Note: stereo datasets may include "negative" images (empty label files). We count those and overlay
        the summary onto the generated `labels.jpg`.
        """
        dataset = getattr(self.train_loader, "dataset", None)
        if dataset is None:
            return

        # Try to use the Stereo3DDetDataset API (label_dir + image_ids + _parse_labels).
        label_dir = getattr(dataset, "label_dir", None)
        image_ids = getattr(dataset, "image_ids", None)
        parse_labels = getattr(dataset, "_parse_labels", None)
        if label_dir is None or image_ids is None or parse_labels is None:
            LOGGER.warning("stereo3ddet: plot_training_labels() skipped (dataset missing label_dir/image_ids/_parse_labels).")
            return

        boxes_list: list[list[float]] = []
        cls_list: list[int] = []
        neg_images = 0
        total_images = 0

        for image_id in image_ids:
            label_file = label_dir / f"{image_id}.txt"
            try:
                labels = parse_labels(label_file)
            except FileNotFoundError:
                LOGGER.warning(f"stereo3ddet: missing label file, skipping: {label_file}")
                continue

            total_images += 1
            if not labels:
                neg_images += 1
                continue

            for lab in labels:
                cls_list.append(int(lab["class_id"]))
                lb = lab["left_box"]
                boxes_list.append(
                    [float(lb["center_x"]), float(lb["center_y"]), float(lb["width"]), float(lb["height"])]
                )

        out = self.save_dir / "labels.jpg"
        if not boxes_list:
            # All-negative or no-label dataset: create a small placeholder image instead of crashing.
            panel = np.full((480, 960, 3), 255, dtype=np.uint8)
            msg1 = "No object labels found to plot."
            msg2 = f"Negative images (empty label files): {neg_images}/{total_images}"
            cv2.putText(panel, msg1, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(panel, msg2, (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.imwrite(str(out), panel)
            if self.on_plot:
                self.on_plot(out)
            return

        boxes = np.asarray(boxes_list, dtype=np.float32)
        cls = np.asarray(cls_list, dtype=np.int64)

        names = self.data.get("names", {})
        if isinstance(names, (list, tuple)):
            names = {i: n for i, n in enumerate(names)}

        plot_labels(boxes, cls, names=names, save_dir=self.save_dir, on_plot=self.on_plot)

        # Overlay negative-image summary onto the generated plot for quick sanity-checking.
        if out.exists():
            im = cv2.imread(str(out))
            if im is not None:
                summary = f"neg images: {neg_images}/{total_images}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                thickness = 2
                (tw, th), baseline = cv2.getTextSize(summary, font, font_scale, thickness)
                x, y = 10, 10 + th
                pad = 6
                cv2.rectangle(im, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), -1)
                cv2.putText(im, summary, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                cv2.imwrite(str(out), im)