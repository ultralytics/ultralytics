from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.data.kitti_stereo import KITTIStereoDataset
from ultralytics.models.yolo.stereo3ddet.augment import StereoAugmentationPipeline, StereoCalibration, PhotometricAugmentor


def _letterbox(image: np.ndarray, new_shape: int, color=(114, 114, 114)) -> Tuple[np.ndarray, float, int, int]:
    h, w = image.shape[:2]
    scale = min(new_shape / h, new_shape / w)
    new_unpad = (int(round(w * scale)), int(round(h * scale)))
    resized = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    dw, dh = new_shape - new_unpad[0], new_shape - new_unpad[1]
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return padded, scale, left, top


class Stereo3DDetAdapterDataset(Dataset):
    """Wraps KITTIStereoDataset to emit YOLO-style batches for training.

    - Returns keys: 'img' (6-channel), 'targets' (dict), 'im_file', 'ori_shape'.
    - Builds a single 6-channel tensor by stacking left and right RGB after identical letterbox.
    - Converts normalized left 2D boxes to resized+letterboxed normalized xywh.
    """

    def __init__(self, root: str | Path, split: str, imgsz: int, names: Dict[int, str] | List[str] | None = None):
        self.root = Path(root)
        self.split = split
        self.imgsz = int(imgsz)
        self.names = names or {}

        self.base = KITTIStereoDataset(root=self.root, split=self.split)
        self.left_dir = self.root / "images" / split / "left"
        # Full stereo augmentation pipeline (photometric + geometric)
        self._aug = StereoAugmentationPipeline(
            photometric=PhotometricAugmentor(p_apply=0.9)
        )

    def __len__(self) -> int:
        return len(self.base)

    def _labels_to_tensors(
        self, labels: List[Dict[str, Any]], scale: float, pad_left: int, pad_top: int, ori_w: int, ori_h: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cls_list: List[float] = []
        bboxes_list: List[List[float]] = []

        for obj in labels:
            try:
                cid = int(obj.get("class_id", -1))
                if cid < 0:
                    continue
                lb = obj.get("left_box", {})
                cx, cy, bw, bh = float(lb.get("center_x", 0)), float(lb.get("center_y", 0)), float(
                    lb.get("width", 0)
                ), float(lb.get("height", 0))
                # denormalize to original pixels
                x = cx * ori_w
                y = cy * ori_h
                w = bw * ori_w
                h = bh * ori_h
                # apply resize + pad
                x = x * scale + pad_left
                y = y * scale + pad_top
                w = w * scale
                h = h * scale
                # normalize to new square size (imgsz)
                cxn = x / self.imgsz
                cyn = y / self.imgsz
                wn = w / self.imgsz
                hn = h / self.imgsz
                # clamp
                cxn = float(min(max(cxn, 0.0), 1.0))
                cyn = float(min(max(cyn, 0.0), 1.0))
                wn = float(min(max(wn, 0.0), 1.0))
                hn = float(min(max(hn, 0.0), 1.0))

                cls_list.append(float(cid))
                bboxes_list.append([cxn, cyn, wn, hn])
            except Exception:
                continue

        if len(cls_list) == 0:
            cls_t = torch.zeros((0,), dtype=torch.float32)
            bboxes_t = torch.zeros((0, 4), dtype=torch.float32)
        else:
            cls_t = torch.tensor(cls_list, dtype=torch.float32)
            bboxes_t = torch.tensor(bboxes_list, dtype=torch.float32)
        return cls_t, bboxes_t

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.base[idx]
        left_img: np.ndarray = sample["left_img"]
        right_img: np.ndarray = sample["right_img"]
        h0, w0 = left_img.shape[:2]

        # Optional stereo augmentation (train split only)
        if self.split == "train":
            calib = sample.get("calib", {})
            calib_obj = StereoCalibration(
                fx=float(calib.get("fx", 0.0)),
                fy=float(calib.get("fy", 0.0)),
                cx=float(calib.get("cx", 0.0)),
                cy=float(calib.get("cy", 0.0)),
                baseline=float(calib.get("baseline", 0.0)),
                height=h0,
                width=w0,
            )
            left_img, right_img, labels_aug, _ = self._aug.augment(left_img, right_img, sample.get("labels", []), calib_obj)
        else:
            labels_aug = sample.get("labels", [])

        # BGR -> RGB for both
        left_rgb = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_rgb = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        # Letterbox both identically using left's parameters, then reapply to right by computing separately
        left_resized, scale_l, pad_left_l, pad_top_l = _letterbox(left_rgb, self.imgsz)
        right_resized, _, _, _ = _letterbox(right_rgb, self.imgsz)
        # Stack to 6 channels
        left_t = torch.from_numpy(left_resized).permute(2, 0, 1).contiguous()
        right_t = torch.from_numpy(right_resized).permute(2, 0, 1).contiguous()
        img6 = torch.cat([left_t, right_t], dim=0)
        if img6.dtype != torch.uint8:
            img6 = img6.to(torch.uint8)

        # Pass labels through for target generation in trainer
        # The trainer will convert these to proper target format
        labels = labels_aug

        image_id = sample.get("image_id")
        im_file = str(self.left_dir / f"{image_id}.png")

        return {
            "img": img6,  # uint8 [0,255], shape (6,H,W)
            "labels": labels,  # Pass labels for target generation
            "im_file": im_file,
            "ori_shape": (h0, w0),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        imgs = torch.stack([b["img"] for b in batch], 0)  # (B,6,H,W)
        labels_list = [b.get("labels", []) for b in batch]
        # Extract class IDs for compatibility with trainer
        # Trainer expects cls to be a tensor with shape [batch_size]
        # We'll use the number of objects as a proxy
        cls_tensor = torch.tensor([len(labels) for labels in labels_list], dtype=torch.long)
        return {
            "img": imgs,
            "labels": labels_list,  # Pass labels for target generation
            "cls": cls_tensor,  # Add cls for trainer compatibility (number of objects per image)
            "im_file": [b["im_file"] for b in batch],
            "ori_shape": [b["ori_shape"] for b in batch],
        }
