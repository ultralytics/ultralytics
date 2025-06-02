"""
@Description: Manitou dataset class.
@Author: Sijie Hu
@Date: 2025-05-06
"""
import glob
import os
import math
from typing import Optional, List, Dict, Tuple
from copy import deepcopy
from collections import defaultdict
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset

from ultralytics.utils.patches import imread
from ultralytics.utils import LOCAL_RANK, LOGGER, NUM_THREADS, TQDM, colorstr
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments
from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds
from ultralytics.utils.torch_utils import TORCHVISION_0_18

from .converter import merge_multi_segment
from .manitou_api import ManitouAPI
from .augmentV1 import (
    Compose,
    FormatManitou,
    LetterBox,
    ManitouResizeCrop,
    RandomLoadText,
    classify_augmentations,
    classify_transforms,
    v8_transformsV1,
)
from .base import BaseDataset
from .converter import merge_multi_segment

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class ManitouDataset(Dataset):
    """
    Manitou Dataset class for loading and processing the Manitou dataset.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        data (dict): Dataset configuration dictionary.
    
    Methods:

    """
    
    def __init__(self, *args, data=None, use_segments=False, stride, imgsz, **kwargs):
        """
        Initialize the ManitouDataset class.

        Args:
            *args: Variable length argument list.
            data (dict): Dataset configuration dictionary.
            use_segments (bool): Indicates if segmentation masks should be used.
            **kwargs: Additional keyword arguments.
        """
        self.use_segments = use_segments
        self.use_keypoints = False  # to be compatible with the original code  TODO: delete
        self.use_obb = False  # to be compatible with the original code  TODO: delete
        self.data = data
        self.stride = stride
        self.imgsz = imgsz
        self.ori_imgsz = imgsz
        self.pre_crop_cfg = {"is_crop": False, "scale": 1, "target_size": imgsz, "original_size": imgsz}
        # ============================================
        # Image size should be divisible by stride
        # Strategically crop the image size to be divisible by stride
        #     1. resize the width to be divisible by stride
        #     2. crop the heigh to be divisible by stride
        # ============================================
        h = imgsz[0] // stride * stride
        w = math.ceil(imgsz[1] / stride) * stride
        if self.imgsz != (h, w):
            self.pre_crop_cfg["is_crop"] = True
            self.pre_crop_cfg["scale"] = w / imgsz[1]
            self.pre_crop_cfg["target_size"] = (h, w)
            self.imgsz = (h, w)
            LOGGER.warning(
                f"Image size {self.imgsz} is not divisible by stride {stride}, resizing and cropping to {(h, w)}"
            )
        
        # In Manitou dataset, the image size is fixed, so rect is not needed.
        kwargs.pop("rect", False)
        kwargs.pop("pad", 0.0)  # for rect, not used in Manitou dataset
        self.full_init(*args, channels=data["channels"], rect=False, pad=0.0, **kwargs)
        
    def full_init(
        self,
        ann_path,
        cache=False,
        augment=True,
        hyp=dict(),
        prefix="",
        rect=False,
        batch_size=16,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        channels=3,
    ):
        """
        Initialize BaseDataset with given configuration and options.

        Args:
            ann_path (str): Path to the json annotation file.
            imgsz (int, optional): Image size for resizing.
            cache (bool | str, optional): Cache images to RAM or disk during training.
            augment (bool, optional): If True, data augmentation is applied.
            hyp (dict, optional): Hyperparameters to apply data augmentation.
            prefix (str, optional): Prefix to print in log messages.
            rect (bool, optional): If True, rectangular training is used.
            batch_size (int, optional): Size of batches.
            stride (int, optional): Stride used in the model.
            pad (float, optional): Padding value.
            single_cls (bool, optional): If True, single class training is used.
            classes (list, optional): List of included classes.
            fraction (float, optional): Fraction of dataset to utilize.
            channels (int, optional): Number of channels in the images (1 for grayscale, 3 for RGB).
        """
        super().__init__()
        self.ann_path = ann_path
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files, self.labels = self.get_img_files_labels(self.ann_path)
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.pad = pad
        if self.rect:  # only for datasets with different sizes of images
            assert self.batch_size is not None
            self.set_rectangle()

        # Buffer thread for mosaic images
        self.buffer = []  # buffer size = batch size
        self.max_buffer_length = min((self.ni, self.batch_size * 8, 1000)) if self.augment else 0

        # Cache images (options are cache = True, False, None, "ram", "disk")
        self.ims, self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni, [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        self.cache = cache.lower() if isinstance(cache, str) else "ram" if cache is True else None
        if self.cache == "ram" and self.check_cache_ram():
            if hyp.deterministic:
                LOGGER.warning(
                    "cache='ram' may produce non-deterministic training results. "
                    "Consider cache='disk' as a deterministic alternative if your disk space allows."
                )
            self.cache_images()
        elif self.cache == "disk" and self.check_cache_disk():
            self.cache_images()

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)
                   
    def parse_label_info(self, raw_label_info):
        """
        Parse label information to target format.
        """
        img_info = raw_label_info["raw_img_info"]
        radar_info = raw_label_info["raw_radar_info"]
        ann_info = raw_label_info["raw_ann_info"]
        
        img_path = str(Path(self.data["path"]) / self.data["img_prefix"] / img_info["file_name"])
        img_shape = (img_info["height"], img_info["width"])  # (h, w)
        img_timestamp = img_info["time_stamp"]
        radar_path = str(Path(self.data["path"]) / self.data["radar_prefix"] / radar_info["file_name"])
        radar_timestamp = radar_info["time_stamp"]
        _prev = None  # will be updated in the 'label_update' function
        _next = None
        img_frame_id = img_info["frame_id"]
        img_frame_name = img_info["frame_name"]
        radar_frame_name = radar_info["frame_name"]
        is_start = img_info["is_start"]
        is_end = img_info["is_end"]
        
        boxes = []
        segments = []
        classes = []
        ins_ids = []
        mot_confs = []
        visibilities = []
        
        for i, ann in enumerate(ann_info):
            if ann.get("is_ego", False):
                continue
            
            if ann.get('ignore', False):
                continue
            
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            
            # if ann['area'] <= 400 or w < 10 or h < 20:
            #     continue
            
            if ann['category_id'] not in self.data["cat_ids"]:
                continue
            
            # The coco box format is [top_left_x, top_left_y, width, height]
            cx, cy = x1 + w / 2, y1 + h / 2
            
            boxes.append([cx, cy, w, h])
            
            if len(ann['segmentation']) == 0:
                s = []
            elif len(ann['segmentation']) > 1:
                s = merge_multi_segment(ann['segmentation'])
                s = np.concatenate(s, axis=0)
            else:
                s = [j for i in ann["segmentation"] for j in i]
                s = np.array(s).reshape(-1, 2)
            segments.append(s)
            
            classes.append(self.data["cat2label"][ann['category_id']])
            ins_ids.append(ann['instance_id'])
            mot_confs.append(ann['mot_conf'])
            visibilities.append(ann['visibility'])
        
        if len(boxes) > 0:
            boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
            classes = np.array(classes, dtype=np.float32).reshape(-1, 1)
            ins_ids = np.array(ins_ids, dtype=np.float32).reshape(-1, 1)
            mot_confs = np.array(mot_confs, dtype=np.float32).reshape(-1, 1)
            visibilities = np.array(visibilities, dtype=np.float32).reshape(-1, 1)
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0, 1), dtype=np.float32)
            ins_ids = np.zeros((0, 1), dtype=np.float32)
            mot_confs = np.zeros((0, 1), dtype=np.float32)
            visibilities = np.zeros((0, 1), dtype=np.float32)
            
        return {
            "im_file": img_path,
            "shape": img_shape,
            "img_timestamp": img_timestamp,
            "img_frame_id": img_frame_id,
            "im_frame_name": img_frame_name,
            "is_start": is_start,
            "is_end": is_end,
            "radar_file": radar_path,
            "radar_timestamp": radar_timestamp,
            "radar_frame_name": radar_frame_name,
            "prev": _prev,
            "next": _next,
            "cls": classes,
            "bboxes": boxes,
            "segments": segments,
            "ins_ids": ins_ids,
            "mot_confs": mot_confs,
            "visibilities": visibilities,
            "normalized": False,
            "bbox_format": "xywh",
        }
    
    def get_img_files_labels(self, ann_path):
        
        def update_labels(label_list):
            """Update 'prev' and 'next' fields in the labels."""
            for i, _lb in enumerate(label_list):
                is_start = _lb["is_start"]
                is_end = _lb["is_end"]
                if not is_start and i > 0:
                    _lb["prev"] = label_list[i - 1]
                if not is_end and i < len(label_list) - 1:
                    _lb["next"] = label_list[i + 1]
            
            return label_list
        
        manitou = ManitouAPI(ann_path)
        cat_ids = self.data["cat_ids"]
        
        img_list = []
        label_list = []
        
        ne = 0  # number of empty images
        vid_ids = manitou.get_vid_ids()
        for vid_id in vid_ids:
            img_ids = manitou.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                # load img info
                raw_img_info = manitou.load_imgs([img_id])[0]
                raw_img_info["img_id"] = img_id
                raw_img_info["video_length"] = len(img_ids)
                
                raw_radar_info = manitou.load_radars([img_id])[0]
                raw_radar_info["radar_id"] = img_id
                
                # load ann info
                ann_ids = manitou.get_ann_ids(imgIds=[img_id], catIds=cat_ids)
                raw_ann_info = manitou.load_anns(ann_ids)
                # get label info
                parsed_label_info = self.parse_label_info(
                    dict(raw_img_info=raw_img_info, raw_radar_info=raw_radar_info, raw_ann_info=raw_ann_info))
                
                if parsed_label_info["bboxes"].shape[0] >0:
                    label_list.append(parsed_label_info)
                    img_list.append(parsed_label_info["im_file"])
                else:
                    ne += 1
                    
        # label_list = update_labels(label_list)  # TODO: update labels with prev and next
    
        return img_list, label_list

    def build_transforms(self, hyp=None):
        """
        Builds and appends transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            hyp.mosaic = hyp.mosaic if self.augment and not self.rect else 0.0
            hyp.mixup = hyp.mixup if self.augment and not self.rect else 0.0
            hyp.cutmix = hyp.cutmix if self.augment and not self.rect else 0.0
            transforms = v8_transformsV1(self, self.imgsz, hyp, self.pre_crop_cfg)
            
        else:
            transforms = Compose([ManitouResizeCrop(self.pre_crop_cfg["scale"],
                                                   self.pre_crop_cfg["target_size"],
                                                   self.pre_crop_cfg["original_size"],
                                                   1.0 if self.pre_crop_cfg["is_crop"] else 0.0),
                                #   LetterBox(new_shape=self.imgsz, scaleup=False)  # no need to use LetterBox
                                  ])
        transforms.append(
            FormatManitou(
                bbox_format="xywh",
                normalize=True,
                return_mask=self.use_segments,
                batch_idx=True,
                mask_ratio=hyp.mask_ratio,
                mask_overlap=hyp.overlap_mask,
                bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
            )
        )
        return transforms

    def close_mosaic(self, hyp):
        """
        Disable mosaic, copy_paste, mixup and cutmix augmentations by setting their probabilities to 0.0.

        Args:
            hyp (dict): Hyperparameters for transforms.
        """
        hyp.mosaic = 0.0
        hyp.copy_paste = 0.0
        hyp.mixup = 0.0
        hyp.cutmix = 0.0
        self.transforms = self.build_transforms(hyp)

    def update_labels_info(self, label):  # TODO: support tracking
        """
        Custom your label format here.

        Args:
            label (dict): Label dictionary containing bboxes, segments, keypoints, etc.

        Returns:
            (dict): Updated label dictionary with instances.

        Note:
            cls is not with bboxes now, classification and semantic segmentation need an independent cls label
            Can also support classification and semantic segmentation by adding or removing dict keys there.
        """
        bboxes = label.pop("bboxes")
        segments = label.pop("segments", [])
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")

        # NOTE: do NOT resample oriented boxes
        segment_resamples = 100 if self.use_obb else 1000
        if len(segments) > 0:
            # make sure segments interpolate correctly if original length is greater than segment_resamples
            max_len = max(len(s) for s in segments)
            segment_resamples = (max_len + 1) if segment_resamples < max_len else segment_resamples
            # list[np.array(segment_resamples, 2)] * num_samples
            segments = np.stack(resample_segments(segments, n=segment_resamples), axis=0)
        else:
            segments = np.zeros((0, segment_resamples, 2), dtype=np.float32)
        label["instances"] = Instances(bboxes, segments, None, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        """
        Collates data samples into batches.

        Args:
            batch (List[dict]): List of dictionaries containing sample data.

        Returns:
            (dict): Collated batch with stacked tensors.
        """
        new_batch = {}
        batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k in {"img", "text_feats"}:
                value = torch.stack(value, 0)
            elif k == "visuals":
                value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
            if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
    
        return new_batch
    
    def __getitem__(self, index):
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index):
        """
        Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (dict): Label dictionary with image and metadata.
        """
        label = deepcopy(self.labels[index])  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(index)
        label["ratio_pad"] = (
            (label["resized_shape"][0] / label["ori_shape"][0], label["resized_shape"][1] / label["ori_shape"][1]),
            (0, 0),  # padding (to compatible with the evaluation, cause we don't use LetterBox data augmentation for Manitou)
            ) # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Return the length of the labels list for the dataset."""
        return len(self.labels)
        
    def load_image(self, i, rect_mode=True):
        """
        Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool, optional): Whether to use rectangular resizing.

        Returns:
            (np.ndarray): Loaded image as a NumPy array.
            (Tuple[int, int]): Original image dimensions in (height, width) format.
            (Tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            # im, _, _, _, _ = self.resize_and_crop(im, bbox=None, classes=None, seg=None)
            if im.ndim == 2:
                im = im[..., None]
            h0, w0 = im.shape[:2]  # orig hw

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    if self.cache != "ram":
                        self.ims[j], self.im_hw0[j], self.im_hw[j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]
                