"""
@Description: Manitou video dataset class.
@Author: Sijie Hu
@Date: 2025-05-19.
"""

import math
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ultralytics.utils import LOGGER
from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import resample_segments
from ultralytics.utils.patches import imread

from .augmentV2 import (
    Compose,
    FormatManitou_MultiImg,
    ManitouResizeCrop_MultiImg,
    RandomFlip_MultiImg,
    RandomHSV_MultiImg,
)
from .converter import merge_multi_segment
from .manitou_api import ManitouAPI, get_manitou_calibrations

# Ultralytics dataset *.cache version, >= 1.0.0 for YOLOv8
DATASET_CACHE_VERSION = "1.0.3"


class ManitouVideoDataset(Dataset):
    """
    Manitou Video Dataset class for loading and processing the Manitou dataset.

    Attributes:
        use_segments (bool): Indicates if segmentation masks should be used.
        data (dict): Dataset configuration dictionary.

    Methods:
    """

    def __init__(self, *args, data=None, use_segments=False, use_radar=False, stride, imgsz, ref_img_sampler, **kwargs):
        """
        Initialize the ManitouDataset class.

        Args:
            *args: Variable length argument list.
            data (dict): Dataset configuration dictionary.
            use_segments (bool): Indicates if segmentation masks should be used.
            **kwargs: Additional keyword arguments.
        """
        super().__init__()
        self.use_segments = use_segments
        self.use_keypoints = False  # to be compatible with the original code  TODO: delete
        self.use_obb = False  # to be compatible with the original code  TODO: delete
        self.data = data
        self.stride = stride
        self.ref_img_sampler = ref_img_sampler
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

        self.use_radar = use_radar  # if True, prepare radar data
        if self.use_radar:
            self.calib_params = get_manitou_calibrations(self.data)
            self.accumulation = self.data.get("accumulation", 1)  # accumulation for radar data

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
        self.ann_path = ann_path
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        # img_files: list of image files, labels : list of annotations, id2idx: dict of video id and frame id to index
        self.im_files, self.radar_files, self.labels, self.id2idx = self.get_img_files_labels(self.ann_path)
        self.ni = len(self.labels[1])  # number of images for one camera
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
        self.ims = {}  # images in RAM
        self.im_hw0 = {}  # original image size
        self.im_hw = {}  # resized image size
        for i in range(1, 5):
            self.ims[i] = [None] * self.ni
            self.im_hw0[i] = [None] * self.ni
            self.im_hw[i] = [None] * self.ni

        # Transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def parse_label_info(self, raw_label_info):
        """Parse label information to target format."""
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
        vid_id = img_info["video_id"]
        bag_id = img_info["bag_id"]
        video_length = img_info["video_length"]
        cam_idx = img_info["cam_idx"]

        boxes = []
        segments = []
        classes = []
        ins_ids = []
        mot_confs = []
        visibilities = []

        for i, ann in enumerate(ann_info):
            if ann.get("is_ego", False):
                continue

            if ann.get("ignore", False):
                continue

            x1, y1, w, h = ann["bbox"]
            inter_w = max(0, min(x1 + w, img_info["width"]) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info["height"]) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue

            # if ann['area'] <= 400 or w < 10 or h < 20:
            #     continue

            if ann["category_id"] not in self.data["cat_ids"]:
                continue

            # The coco box format is [top_left_x, top_left_y, width, height]
            cx, cy = x1 + w / 2, y1 + h / 2

            boxes.append([cx, cy, w, h])

            if len(ann["segmentation"]) == 0:
                s = []
            elif len(ann["segmentation"]) > 1:
                s = merge_multi_segment(ann["segmentation"])
                s = np.concatenate(s, axis=0)
            else:
                s = [j for i in ann["segmentation"] for j in i]
                s = np.array(s).reshape(-1, 2)
            segments.append(s)

            classes.append(self.data["cat2label"][ann["category_id"]])
            ins_ids.append(ann["mot_instance_id"])
            mot_confs.append(ann["mot_conf"])
            visibilities.append(ann["visibility"])

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
            "vid_id": vid_id,
            "bag_id": bag_id,
            "video_length": video_length,
            "cam_idx": cam_idx,
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

        # Initialize lists to store image and label information for each camera
        img_list = {1: [], 2: [], 3: [], 4: []}
        label_list = {1: [], 2: [], 3: [], 4: []}
        radar_list = {1: [], 2: [], 3: [], 4: []}
        idx_dict = {}

        bag_ids = manitou.get_bag_ids()
        for bag_id in bag_ids:
            vid_ids = manitou.get_vid_ids(bagIds=[bag_id])
            assert len(vid_ids) == 4, (
                f"Only 4 cameras are supported, but {len(vid_ids)} cameras found in bag {bag_id}. \n"
                f"\t {manitou.bags[bag_id]['cam_idx']}"
            )
            idx_dict[bag_id] = {1: {}, 2: {}, 3: {}, 4: {}}

            # To address the length mismatch issue, we make the length
            # of each camera's label and image lists to the smallest length of the four cameras.
            cam_len = [len(manitou.get_img_ids_from_vid(vid_id)) for vid_id in vid_ids]
            min_length = min(cam_len)
            if np.any(np.array(cam_len) != min_length):
                LOGGER.warning(
                    f"Camera lengths are not equal in bag {bag_id}, name: {manitou.bags[bag_id]['name']}. \n"
                    f"\t Camera lengths: {cam_len}. \n"
                    f"\t The minimum length is {min_length}, the other cameras will be truncated to this length."
                )

            for vid_id in vid_ids:
                img_ids = manitou.get_img_ids_from_vid(vid_id)
                vid_name = manitou.vids[vid_id]["name"]
                cam_idx = int(vid_name[-1])

                for img_id in img_ids:
                    # load img info
                    raw_img_info = manitou.load_imgs([img_id])[0]
                    raw_img_info["img_id"] = img_id
                    raw_img_info["video_length"] = min_length
                    raw_img_info["cam_idx"] = cam_idx

                    raw_radar_info = manitou.load_radars([img_id])[0]
                    raw_radar_info["radar_id"] = img_id

                    # load ann info
                    ann_ids = manitou.get_ann_ids(imgIds=[img_id], catIds=cat_ids)
                    raw_ann_info = manitou.load_anns(ann_ids)
                    # get label info
                    parsed_label_info = self.parse_label_info(
                        dict(raw_img_info=raw_img_info, raw_radar_info=raw_radar_info, raw_ann_info=raw_ann_info)
                    )

                    global_idx = len(img_list[cam_idx])
                    img_frame_id = parsed_label_info["img_frame_id"]

                    if img_frame_id >= min_length:
                        LOGGER.warning(
                            f"Image frame id {img_frame_id} is greater than the minimum length {min_length} for bag {bag_id}, cam {cam_idx}. \n"
                            f"\t This image will be skipped."
                        )
                        continue
                    label_list[cam_idx].append(parsed_label_info)
                    img_list[cam_idx].append(parsed_label_info["im_file"])
                    radar_list[cam_idx].append(parsed_label_info["radar_file"])
                    idx_dict[bag_id][cam_idx][img_frame_id] = global_idx

        assert (
            len(label_list[1])
            == len(label_list[2])
            == len(label_list[3])
            == len(label_list[4])
            == len(img_list[1])
            == len(img_list[2])
            == len(img_list[3])
            == len(img_list[4])
            == len(radar_list[1])
            == len(radar_list[2])
            == len(radar_list[3])
            == len(radar_list[4])
        ), (
            f"Number of images and labels do not match. \n"
            f"\t number of labels (cam1, cam2, cam3, cam4): {len(label_list[1])}, {len(label_list[2])}, {len(label_list[3])}, {len(label_list[4])} \n"
            f"\t number of images (cam1, cam2, cam3, cam4): {len(img_list[1])}, {len(img_list[2])}, {len(img_list[3])}, {len(img_list[4])} \n"
            f"\t number of radars (radar1, radar2, radar3, radar4): {len(radar_list[1])}, {len(radar_list[2])}, {len(radar_list[3])}, {len(radar_list[4])} \n"
        )

        # label_list = update_labels(label_list)  # TODO: update labels with prev and next

        return img_list, radar_list, label_list, idx_dict

    def build_transforms(self, hyp=None):
        """
        Builds and appends transforms to the list.

        Args:
            hyp (dict, optional): Hyperparameters for transforms.

        Returns:
            (Compose): Composed transforms.
        """
        if self.augment:
            transforms = Compose(
                [
                    ManitouResizeCrop_MultiImg(
                        self.pre_crop_cfg["scale"],
                        self.pre_crop_cfg["target_size"],
                        self.pre_crop_cfg["original_size"],
                        1.0 if self.pre_crop_cfg["is_crop"] else 0.0,
                    ),
                    RandomHSV_MultiImg(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
                    RandomFlip_MultiImg(direction="horizontal", p=hyp.fliplr),
                ]
            )

        else:
            transforms = Compose(
                [
                    ManitouResizeCrop_MultiImg(
                        self.pre_crop_cfg["scale"],
                        self.pre_crop_cfg["target_size"],
                        self.pre_crop_cfg["original_size"],
                        1.0 if self.pre_crop_cfg["is_crop"] else 0.0,
                    )
                ]
            )

        transforms.append(
            FormatManitou_MultiImg(
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

    def update_labels_info(self, label):
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

        # collect key frames and reference frames
        def to_list(batch):
            """Convert batch to a list of dictionaries."""
            b_list = []
            for b in batch:
                if isinstance(b, list):
                    b_list.extend(to_list(b))
                elif isinstance(b, dict):
                    b_list.extend([b])
                else:
                    raise TypeError(f"Unsupported collate type: {type(b)}")
            return b_list

        def _collate_cameras(batch):
            # to list
            new_batch = {}
            batch = [dict(sorted(b.items())) for b in batch]  # make sure the keys are in the same order
            keys = batch[0].keys()
            values = list(zip(*[list(b.values()) for b in batch]))
            for i, k in enumerate(keys):
                value = values[i]
                if k in {
                    "img",
                }:  # image shape is (N, C, H, W) N is the number of cameras
                    value = torch.cat(value, 0)  # b (N, C, H, W) -> (b * N, C, H, W)
                elif k == "visuals":
                    value = torch.nn.utils.rnn.pad_sequence(value, batch_first=True)
                if k in {"masks", "keypoints", "bboxes", "cls", "segments", "obb"}:
                    value = torch.cat(value, 0)

                # if k not in {"batch_idx", "visuals", "masks", "keypoints", "bboxes", "cls", "segments", "obb", "radar"}:
                if isinstance(value, (list, tuple)):
                    if isinstance(value[0], (list, tuple)):
                        # if value is a list of lists, we need to extend it into a single list
                        _value = []
                        for v in value:
                            _value.extend(v)
                        value = _value

                new_batch[k] = value
            new_batch["batch_idx"] = list(new_batch["batch_idx"])
            for i in range(len(new_batch["batch_idx"])):
                if len(new_batch["batch_idx"][i]) == 0:
                    continue
                # get the maximum index of the batch_idx for each batch
                offset = max(new_batch["batch_idx"][i]) + 1
                if offset > 4:
                    LOGGER.warning(
                        f"Batch index {new_batch['batch_idx'][i]} exceeds 3, which is not expected. "
                        f"Please check the batch size and the number of cameras."
                    )
                new_batch["batch_idx"][i] += i * offset  # add target image index for build_targets()
            new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)

            return new_batch

        key_batch = []
        ref_batch = []
        for i, b in enumerate(batch):
            key_batch.append(b["labels"])
            ref_batch.append(b["ref_labels"])

        # collate key frames
        key_batch = _collate_cameras(key_batch)
        ref_batch = _collate_cameras(ref_batch) if ref_batch[0] is not None else None
        # key_radars = _collate_radars(key_radars)
        # ref_radars = _collate_radars(ref_radars) if len(ref_radars[1]) > 0 and ref_radars[1][0] is not None else None

        return {"key_frames": key_batch, "ref_frames": ref_batch, "key_radars": None, "ref_radars": None}

    def check_name(self, labels):
        """Check if the labels have the same name."""
        frame_names = [label["im_frame_name"] for label in labels]
        if not len(set(frame_names)) == 1:
            LOGGER.error(f"Frame names of different cameras are not the same: {frame_names}.")
            raise ValueError(f"Frame names of different cameras are not the same: {frame_names}.")

    def __getitem__(self, index):
        """Return transformed label information for given index."""
        labels = [self.get_image_and_label(i, index) for i in range(1, 5)]

        # check name
        self.check_name(labels)

        if self.ref_img_sampler is not None and self.ref_img_sampler.get("num_ref_imgs", 0) > 0:
            # ref_labels = [self.ref_img_sampling(label, **self.ref_img_sampler) for label in labels]
            ref_labels = self.ref_img_sampling_4cam(labels[0], **self.ref_img_sampler)
        else:
            ref_labels = [None] * 4

        new_labels = {"labels": labels, "ref_labels": ref_labels}

        new_labels = self.transforms(new_labels)

        return new_labels

    def ref_img_sampling_4cam(self, key_img_label, scope, num_ref_imgs=1, method="uniform"):
        """
        Sample reference images for the given key image label across 4 cameras.
        Note: To avoid memory increase, we sample the same reference images for all cameras. This will decrease the diversity of reference images.

        Return:
            (list): List of reference image labels for each camera.
        """
        if num_ref_imgs <= 0 or scope <= 0:
            return [None] * 4

        if method != "uniform":
            raise NotImplementedError(f"Ref_img_sampler method {method} is not implemented.")

        key_vid_id = key_img_label["vid_id"]
        key_bag_id = key_img_label["bag_id"]
        cam_idx = key_img_label["cam_idx"]
        key_frame_id = key_img_label["img_frame_id"]

        if method == "uniform":
            # sample uniformly from the video
            left = max(0, key_frame_id - scope)
            right = min(key_frame_id + scope, key_img_label["video_length"] - 1)
            # remove the key frame id
            valid_ref_frame_ids = [i for i in range(left, right + 1) if i != key_frame_id]
            if len(valid_ref_frame_ids) < num_ref_imgs:
                raise ValueError(
                    f"Not enough reference frames in the video {key_vid_id} for frame {key_frame_id}. "
                    f"Only {len(valid_ref_frame_ids)} available, but {num_ref_imgs} requested."
                    f" Please check the scope and num_ref_imgs parameters."
                )
            ref_frame_ids = np.random.choice(valid_ref_frame_ids, num_ref_imgs, replace=False)
            if isinstance(ref_frame_ids, int):
                ref_frame_ids = [ref_frame_ids]
            ref_frame_ids = sorted(ref_frame_ids)
            # return the reference image labels
            indexs = [self.id2idx[key_bag_id][cam_idx][int(i)] for i in ref_frame_ids]
            ref_img_labels = []
            for j in range(1, 5):
                ref_img_labels.append([self.get_image_and_label(j, i) for i in indexs])

            return ref_img_labels

    def ref_img_sampling(self, key_img_label, scope, num_ref_imgs=1, method="uniform"):
        """
        Sample reference images for the given key image label.

        Return:
            (list): List of reference image labels.
        """
        if num_ref_imgs <= 0 or scope <= 0:
            return None

        if method != "uniform":
            raise NotImplementedError(f"Ref_img_sampler method {method} is not implemented.")

        key_vid_id = key_img_label["vid_id"]
        key_bag_id = key_img_label["bag_id"]
        cam_idx = key_img_label["cam_idx"]
        key_frame_id = key_img_label["img_frame_id"]

        if method == "uniform":
            # sample uniformly from the video
            left = max(0, key_frame_id - scope)
            right = min(key_frame_id + scope, key_img_label["video_length"] - 1)
            # remove the key frame id
            valid_ref_frame_ids = [i for i in range(left, right + 1) if i != key_frame_id]
            if len(valid_ref_frame_ids) < num_ref_imgs:
                raise ValueError(
                    f"Not enough reference frames in the video {key_vid_id} for frame {key_frame_id}. "
                    f"Only {len(valid_ref_frame_ids)} available, but {num_ref_imgs} requested."
                    f" Please check the scope and num_ref_imgs parameters."
                )
            ref_frame_ids = np.random.choice(valid_ref_frame_ids, num_ref_imgs, replace=False)
            if isinstance(ref_frame_ids, int):
                ref_frame_ids = [ref_frame_ids]
            ref_frame_ids = sorted(ref_frame_ids)
            # return the reference image labels
            indexs = [self.id2idx[key_bag_id][cam_idx][int(i)] for i in ref_frame_ids]
            ref_img_labels = [self.get_image_and_label(cam_idx, i) for i in indexs]

            return ref_img_labels

    def get_image_and_label(self, cam_idx, index):
        """
        Get and return label information from the dataset.

        Args:
            cam_idx (int): Camera index (1-4).
            index (int): Index of the image to retrieve.

        Returns:
            (dict): Label dictionary with image and metadata.
        """
        label = deepcopy(
            self.labels[cam_idx][index]
        )  # requires deepcopy() https://github.com/ultralytics/ultralytics/pull/1948
        label.pop("shape", None)  # shape is for rect, remove it
        label["img"], label["ori_shape"], label["resized_shape"] = self.load_image(cam_idx, index)
        label["radar"] = self.load_radar(cam_idx, index) if self.use_radar else None
        label["ratio_pad"] = (
            (label["resized_shape"][0] / label["ori_shape"][0], label["resized_shape"][1] / label["ori_shape"][1]),
            (
                0,
                0,
            ),  # padding (to compatible with the evaluation, cause we don't use LetterBox data augmentation for Manitou)
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        # update label with intrinsic camera matrix if available
        if self.use_radar:
            label["intrinsic_K"] = deepcopy(self.calib_params[f"camera{cam_idx}_K"])
        return self.update_labels_info(label)

    def __len__(self):
        """Return the length of the labels list for the dataset."""
        return len(
            self.labels[1]
        )  # Note: each rosbag contains 4 cameras, we only need to return the number of images in one camera.

    def load_image(self, cam_idx, i):
        """
        Load an image from camera index and dataset index.

        Args:
            cam_idx (int): Camera index (1-4).
            i (int): Index of the image to load.

        Returns:
            (np.ndarray): Loaded image as a NumPy array.
            (Tuple[int, int]): Original image dimensions in (height, width) format.
            (Tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        im, f = self.ims[cam_idx][i], self.im_files[cam_idx][i]
        if im is None:  # not cached in RAM
            im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            # im, _, _, _, _ = self.resize_and_crop(im, bbox=None, classes=None, seg=None)
            if im.ndim == 2:
                im = im[..., None]
            h0, w0 = im.shape[:2]  # orig hw

            # Add to buffer if training with augmentations
            if self.augment:
                self.ims[cam_idx][i], self.im_hw0[cam_idx][i], self.im_hw[cam_idx][i] = (
                    im,
                    (h0, w0),
                    im.shape[:2],
                )  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    self.ims[cam_idx][j], self.im_hw0[cam_idx][j], self.im_hw[cam_idx][j] = None, None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[cam_idx][i], self.im_hw0[cam_idx][i], self.im_hw[cam_idx][i]

    def load_radar(self, radar_idx, index):
        path = self.radar_files[radar_idx][index]
        radar_pc = self._load_radar(path)
        if self.accumulation > 1:
            for _ in range(self.accumulation - 1):
                frame_name = int(self.labels[radar_idx][index]["radar_frame_name"])
                if frame_name - 1 < 0:
                    break
                prev_name = f"{frame_name - 1:06d}.{path.split('.')[-1]}"
                path = str(Path(path).parent / f"{prev_name}")
                radar_pc = np.concatenate((radar_pc, self._load_radar(path)), axis=0)

        return radar_pc

    def _load_radar(self, path):
        if path.endswith(".txt"):
            return np.loadtxt(path, delimiter=" ", dtype=np.float32)
        elif path.endswith(".npy"):
            return np.load(path, allow_pickle=True)
        else:
            raise ValueError(f"❌ Unsupported radar file format: {path}. Supported formats are .txt and .npy.")
