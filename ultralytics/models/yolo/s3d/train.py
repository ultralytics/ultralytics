# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import json
import math
from collections import Counter
from copy import copy
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from ultralytics.data import build_dataloader
from ultralytics.data.stereo.box3d import Box3D
from ultralytics.models import yolo
from ultralytics.models.yolo.s3d.dataset import Stereo3DDetDataset
from ultralytics.models.yolo.s3d.head import DEPTH_MAX, DEPTH_MIN
from ultralytics.models.yolo.s3d.loss import LOSS_NAMES
from ultralytics.models.yolo.s3d.model import Stereo3DDetModel
from ultralytics.models.yolo.s3d.preprocess import preprocess_stereo_batch
from ultralytics.nn.modules.block import StereoCostVolume
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.plotting import Annotator, VisualizationConfig, colors, plot_labels, plot_stereo3d_boxes


class DriveBalancedSampler(torch.utils.data.WeightedRandomSampler):
    """A `WeightedRandomSampler` that satisfies the sampler contract DDP training expects.

    `BaseTrainer` calls `self.train_loader.sampler.set_epoch(epoch)` unconditionally whenever `RANK != -1`
    (`ultralytics/engine/trainer.py`), because the default distributed sampler needs it to reshuffle. Plain
    `WeightedRandomSampler` has no such method, so every rank raised `AttributeError` on the first epoch and
    a multi-GPU run died before its first step while single-process runs were unaffected.

    Reseeding per epoch is also better than letting the generator drift: the draw becomes a pure function of
    (seed, epoch), so a resumed run reproduces the same sequence instead of depending on how far the
    generator happened to advance before the interruption. The epoch is mixed in with a large stride so that
    rank r at epoch e never collides with rank r' at epoch e'.
    """

    def __init__(self, weights: torch.Tensor, num_samples: int, seed: int = 0) -> None:
        """Initialize with sampling weights, per-epoch draw count, and a base seed (pass the rank)."""
        super().__init__(
            weights, num_samples=num_samples, replacement=True, generator=torch.Generator().manual_seed(seed)
        )
        self.seed = seed

    def set_epoch(self, epoch: int) -> None:
        """Reseed so each epoch draws a different, reproducible sample.

        Args:
            epoch (int): Current epoch index.
        """
        self.generator.manual_seed(self.seed + epoch * 1_000_003)


def drive_balanced_sampler(
    image_ids: list[str],
    drives: dict[str, str],
    balance: float = 1.0,
    num_samples: int | None = None,
    seed: int = 0,
) -> DriveBalancedSampler:
    """Sample frames so that no recording drive dominates an epoch.

    KITTI-style stereo splits are extremely drive-concentrated: on the drive-disjoint 3712-frame split the
    71 drives have a median of 10 frames but a maximum of 321, so the top five drives supply 37.8% of every
    epoch's gradient while the smallest half of the drives supply 3.9%. Consecutive frames of one drive are
    near-duplicates, so proportional sampling counts redundant scenes many times over.

    The correction caps each drive's contribution instead of inverting its frequency. A frame in a drive of
    `n` frames gets weight `min(1, cap / n)` with `cap = balance * len(image_ids) / n_drives`, i.e. `balance`
    multiples of the mean frames-per-drive. Only over-represented drives are down-weighted; every frame in
    a drive at or below the cap keeps the uniform weight it has today, so relative probabilities inside the
    long tail are untouched. Full inverse-frequency balancing (weight `1/n`) was rejected: it makes a
    one-frame drive as influential as the 321-frame drive, drawing that single frame ~52 times per epoch.

    Args:
        image_ids (list[str]): Frame stems in dataset order (index i of the returned sampler indexes here).
        drives (dict[str, str]): Map of frame stem to drive identifier. Unmapped frames form singleton drives.
        balance (float): Cap in multiples of the mean frames-per-drive. Smaller is flatter; large values
            recover uniform sampling exactly.
        num_samples (int, optional): Draws per epoch. Defaults to `len(image_ids)`, keeping epoch length —
            and therefore the LR schedule and total step count — identical to uniform sampling.
        seed (int): Generator seed; pass the process rank so DDP ranks draw independent samples.

    Returns:
        (torch.utils.data.WeightedRandomSampler): Sampler over `range(len(image_ids))`, with replacement.
    """

    # Unmapped frames become their own singleton drive; the tuple key cannot collide with a drive string.
    # KITTI frame ids appear zero-padded on disk ("000003") but split files conventionally key them as
    # plain integers ("3"), so accept both. Without this every lookup misses, every frame becomes its own
    # drive, and the sampler silently degrades to uniform — the exact failure this function exists to fix.
    def _drive_of(stem: str):
        if stem in drives:
            return drives[stem]
        if stem.isdigit() and (plain := str(int(stem))) in drives:
            return drives[plain]
        return (None, stem)  # singleton; the tuple key cannot collide with a drive string

    groups = [_drive_of(str(i)) for i in image_ids]
    if unmapped := sum(1 for g in groups if isinstance(g, tuple)):
        LOGGER.warning(
            "s3d: %d of %d train frames are absent from the drives map and are treated as single-frame "
            "drives. Check that the drives file covers this split.",
            unmapped,
            len(groups),
        )
    counts = Counter(groups)
    cap = max(balance, 1e-9) * len(groups) / len(counts)
    weights = torch.tensor([min(1.0, cap / counts[g]) for g in groups], dtype=torch.double)

    total = float(weights.sum())
    shares = sorted((counts[g] * min(1.0, cap / counts[g]) / total for g in counts), reverse=True)
    LOGGER.info(
        "s3d: drive-balanced sampling over %d drives (balance=%.2f, cap=%.1f frames/drive) — top-5 drive "
        "share %.1f%% -> %.1f%%, largest drive %.1f%% -> %.1f%%",
        len(counts),
        balance,
        cap,
        100 * sum(sorted((c / len(groups) for c in counts.values()), reverse=True)[:5]),
        100 * sum(shares[:5]),
        100 * max(counts.values()) / len(groups),
        100 * shares[0],
    )
    return DriveBalancedSampler(weights, num_samples=num_samples or len(groups), seed=seed)


class Stereo3DDetTrainer(yolo.detect.DetectionTrainer):
    """Stereo 3D Detection trainer extending DetectionTrainer with stereo-specific dataset, loss, and validation."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides["task"] = "s3d"
        super().__init__(cfg, overrides, _callbacks)

    def get_validator(self):
        """Return a Stereo3DDetValidator, currently extending DetectionValidator."""
        # T204: Determine loss names dynamically from model before creating validator
        self.loss_names = LOSS_NAMES
        val = yolo.s3d.Stereo3DDetValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )
        # Set names early so CSV header includes per-class/difficulty R40 AP keys
        names = getattr(self.model, "names", None)
        if names:
            val.metrics.names = names
            val.metrics.nc = len(names)
        return val

    def progress_string(self):
        """Return formatted training progress string with loss names."""
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

        # Root path and splits
        root = Path(data_cfg["path"])
        # Accept either directory-style train/val or txt; KITTIStereoDataset uses split names
        train_split = data_cfg.get("train_split", "train")
        val_split = data_cfg.get("val_split", "val")

        names = data_cfg.get("names")  # {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
        nc = data_cfg.get("nc", len(names))

        # Extract mean dimensions if present in dataset config
        mean_dims = data_cfg.get("mean_dims")
        std_dims = data_cfg.get("std_dims")

        # Scan up to 200 label files for the width-normalized disparities they contain — fields 1 and 5
        # are the left/right box centres, so their difference IS disparity/width, the grid
        # StereoCostVolume needs to sample. Frame ids are grouped by recording drive, so the files are
        # strided rather than truncated: taking the first 200 would measure the grid on two or three
        # scenes and miss whatever depth range the rest of the split covers.
        disparities: list[float] = []
        label_files = sorted((root / "labels" / train_split).glob("*.txt"))
        for f in label_files[:: max(1, len(label_files) // 200)][:200]:
            with open(f) as fh:
                for line in fh:
                    parts = line.strip().split()
                    if len(parts) > 5:
                        d = float(parts[1]) - float(parts[5])
                        if d > 0:
                            disparities.append(d)

        # Return a dict compatible with BaseTrainer expectations, plus stereo descriptors
        return {
            "yaml_file": str(self.args.data) if isinstance(self.args.data, (str, Path)) else None,
            "path": str(root),
            "channels": 6,
            # Signal to our get_dataloader/build_dataset that this is a stereo dataset
            "train": {"type": "kitti_stereo", "root": str(root), "split": train_split},
            "val": {"type": "kitti_stereo", "root": str(root), "split": val_split},
            "names": names,
            "nc": nc,
            # carry over optional stereo metadata if present
            "stereo": data_cfg.get("stereo", True),
            "baseline": data_cfg.get("baseline"),
            "depth_min": data_cfg.get("depth_min", DEPTH_MIN),
            "depth_max": data_cfg.get("depth_max", DEPTH_MAX),
            "disparity_range": self._disparity_range(disparities, data_cfg),
            "mean_dims": mean_dims,
            "std_dims": std_dims,
            # Optional: JSON file with a {"drives": {frame_id: drive}} map enabling drive-balanced
            # sampling. Absent for datasets that do not record which recording each frame came from.
            "drives": data_cfg.get("drives"),
            "drive_balance": float(data_cfg.get("drive_balance", 1.0)),
        }

    @staticmethod
    def _disparity_range(disparities: list[float], data_cfg: dict) -> tuple[float, float] | None:
        """Width-normalized disparity range for the cost volume grid, from the scanned labels.

        An explicit `disparity_range: [min, max]` in the dataset YAML wins. Otherwise the range is the
        [p1, p99] of observed disparities widened by 20%, so the grid covers the tail without spending
        levels on disparities the dataset never contains. Returns None when nothing can be inferred,
        leaving the module's KITTI-like default in place.
        """
        explicit = data_cfg.get("disparity_range")
        if explicit is not None:
            lo, hi = (float(v) for v in explicit)
            return lo, hi
        if len(disparities) < 32:
            LOGGER.warning(
                "s3d: only %d labelled disparities found — keeping the default cost-volume disparity grid. "
                "Set `disparity_range: [min, max]` (width-normalized) in the dataset YAML to target it.",
                len(disparities),
            )
            return None
        lo, hi = (float(v) for v in np.percentile(disparities, (1, 99)))
        pad = 0.2 * (hi - lo)
        return max(lo - pad, 0.0), hi + pad

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

            # Get mean_dims from dataset config
            mean_dims = self.data.get("mean_dims")
            std_dims = self.data.get("std_dims")
            return Stereo3DDetDataset(
                root=str(desc.get("root", ".")),
                split=str(desc.get("split", "train")),
                imgsz=imgsz_hw,
                names=self.data.get("names"),
                mean_dims=mean_dims,
                std_dims=std_dims,
                augment=(mode == "train"),
                hyp=self.args,
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
                sampler=self._drive_sampler(dataset, rank) if mode == "train" else None,
            )
        # Fallback to default detection dataloader
        return super().get_dataloader(dataset_path, batch_size=batch_size, rank=rank, mode=mode)

    def _drive_sampler(self, dataset: Stereo3DDetDataset, rank: int):
        """Build the drive-balanced train sampler, or None when the dataset declares no drives map.

        Each DDP rank draws its own `len(dataset) / world_size` frames from the shared weight vector. The
        draw is with replacement, so an exact partition is neither possible nor meaningful; matching
        DistributedSampler's per-rank count is what keeps the batch count equal across ranks.

        Args:
            dataset (Stereo3DDetDataset): Train dataset, whose `im_file` order defines the sampler indices.
            rank (int): Process rank, -1 for single-process training.

        Returns:
            (torch.utils.data.WeightedRandomSampler | None): Sampler, or None to keep the default behaviour.
        """
        drives_file = self.data.get("drives")
        if not drives_file:
            return None
        path = Path(drives_file)
        if not path.is_absolute():
            path = Path(self.data["path"]) / path
        drives = json.loads(path.read_text())
        drives = drives.get("drives", drives)  # accept either a bare map or a split file carrying one
        world_size = max(getattr(self, "world_size", 1) or 1, 1)
        return drive_balanced_sampler(
            [Path(f).stem for f in dataset.im_files],
            drives,
            balance=float(self.data.get("drive_balance", 1.0)),
            num_samples=math.ceil(len(dataset) / world_size) if rank != -1 else len(dataset),
            # Mix the run seed with the rank. Seeding from the rank alone made every single-process run
            # (rank -1 -> 0) draw the identical frame sequence, so a multi-seed noise-floor measurement
            # would have varied initialisation and augmentation but NOT the sampling order, understating
            # the true run-to-run spread. The 10_000 stride keeps distinct (seed, rank) pairs distinct.
            seed=int(getattr(self.args, "seed", 0) or 0) * 10_000 + max(rank, 0),
        )

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
        model = Stereo3DDetModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if verbose and RANK == -1:
            LOGGER.info(
                f"Initialized Stereo3DDetModel with {self.data['nc']} classes and {self.data['channels']} input channels"
            )
        if weights:
            model.load(weights)
            if verbose and RANK == -1:
                LOGGER.info(f"Loaded weights from {weights}")
        model.model[-1].depth_dfl._set_range(float(self.data["depth_min"]), float(self.data["depth_max"]))

        disp_range = self.data.get("disparity_range")
        if disp_range is not None:
            for m in model.model:
                if isinstance(m, StereoCostVolume):
                    m.set_disparity_range(*disp_range)
                    if verbose and RANK == -1:
                        LOGGER.info(
                            "s3d: cost-volume disparity grid set to %.5f..%.5f of image width "
                            "(%d levels, %d correlation groups)",
                            *disp_range,
                            m.num_levels,
                            m.groups,
                        )

        return model

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        super().set_model_attributes()
        # Store mean/std dims on model so predictor can read them without the data YAML.
        # Reorder from YAML [L,W,H] to decode format {int_id: (H,W,L)}.
        self.model.mean_dims = self._reorder_dims(self.data.get("mean_dims"))
        self.model.std_dims = self._reorder_dims(self.data.get("std_dims"))

    @staticmethod
    def _reorder_dims(raw_dims):
        """Convert YAML dims {key: [L,W,H]} to decode format {int_id: (H,W,L)}."""
        if raw_dims is None:
            return None
        result = {}
        for i, (key, dims) in enumerate(raw_dims.items()):
            if isinstance(dims, (list, tuple)) and len(dims) == 3:
                l, w, h = dims
                cid = key if isinstance(key, int) else i
                result[cid] = (h, w, l)
        return result if result else None

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
        assert "im_file" in batch, "im_file is required in batch"
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
            assert _6_channel_img.shape[0] == 6, f"6 channel image required, got {_6_channel_img.shape[0]}"
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
            if isinstance(calib_i, dict) and len(labels) and calib_i is not None:
                boxes3d = [
                    b
                    for lab in labels
                    if (b := Box3D.from_label(lab, calib_i, self.data["names"], L3.shape[:2])) is not None
                ]
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
            out = self.save_dir / f"train_batch{ni}.jpg"
            cv2.imwrite(str(out), grid)

    def plot_training_labels(self) -> None:
        """Plot training label statistics for s3d.

        The default detection implementation expects a YOLODetectionDataset-style `dataset.labels` cache.
        Our stereo dataset does not provide that cache, so we build the arrays by scanning label files.

        Note: stereo datasets may include "negative" images (empty label files). We count those and overlay
        the summary onto the generated `labels.jpg`.
        """
        boxes_list: list[list[float]] = []
        cls_list: list[int] = []
        neg_images = 0
        total_images = 0
        for label in self.train_loader.dataset.labels:
            label = label["labels"]
            total_images += 1
            if not label:
                neg_images += 1
                continue

            for lab in label:
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
