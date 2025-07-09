import functools
import random

import cv2
import numpy as np

from ultralytics.utils.instance import Instances
from ultralytics.utils.ops import segment2box

from .augment import *


class MosaicV1(Mosaic):
    """Mosaic adapted to support rectangular images, i.e., width != height."""

    def _merge_into_first_label(fn):
        """
        Decorator to update the labels instead of returning a new dict.

        This is used for the `Mosaic` transformation.
        """

        @functools.wraps(fn)
        def wrapper(self, mosaic_labels, *args, **kwargs):
            new_labels = fn(self, mosaic_labels, *args, **kwargs)
            if len(mosaic_labels) > 0 and isinstance(new_labels, dict):
                original = mosaic_labels[0]
                original.update(new_labels)
                return original
            return new_labels

        return wrapper

    def __init__(self, dataset, imgsz=640, p=1.0, n=4, pre_transform=None):
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4}, "grid must be equal to 4 (combine 4 images)."
        if isinstance(imgsz, int):
            imgsz = [imgsz, imgsz]
        assert isinstance(imgsz, (list, tuple)) and len(imgsz) == 2, (
            f"imgsz should be a list or tuple of length 2 (h, w), but got {imgsz}."
        )
        self.imgsz = imgsz  # (h, w)
        self.border = (-imgsz[0] // 2, -imgsz[1] // 2)  # border size for the mosaic
        self.n = n  # number of images to combine
        self.p = p  # probability of applying mosaic
        self.pre_transform = pre_transform  # pre-transform to apply to each image
        self.dataset = dataset  # dataset to use for mosaic

    def _mosaic4(self, labels):
        mosaic_labels = []
        im_h, im_w = self.imgsz
        new_h, new_w = im_h * 2, im_w * 2
        border_y, border_x = self.border

        # randomly select a center point for the mosaic
        xc = int(random.uniform(-border_x, new_w + border_x))  # mosaic center x
        yc = int(random.uniform(-border_y, new_h + border_y))  # mosaic center y

        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full((new_h, new_w, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, new_w), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(new_h, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, new_w), min(new_h, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    @_merge_into_first_label
    def _cat_labels(self, mosaic_labels):
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        im_h, im_w = self.imgsz
        new_h, new_w = im_h * 2, im_w * 2  # mosaic image size
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        # Final labels
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (new_h, new_w),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(new_w, new_h)
        good = final_labels["instances"].remove_zero_area_boxes()
        final_labels["cls"] = final_labels["cls"][good]
        if "texts" in mosaic_labels[0]:
            final_labels["texts"] = mosaic_labels[0]["texts"]
        return final_labels


class ManitouResizeCrop:
    """
    Resize and crop (top-left, (h, w)) the image , bounding box, and segmentations based on the given configuration.

    Step 1: Resize the image based on the width
    Step 2: Crop the image to the specified height
    In this way, the image only loses the upper part, which is less important for the model.
    """

    def __init__(self, scale, target_size, original_size, p):
        self.scale = scale
        self.target_size = target_size
        self.original_size = original_size
        self.p = p

        self.rescaled_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        self.y_offset = self.rescaled_size[0] - target_size[0]  # y offset for cropping
        self.x_offset = 0  # x offset for cropping, always 0 in this case

    def __call__(self, labels=None, image=None):
        """
        Args:
            labels (Dict, None): Dictionary containing image data and associated labels, or enpty Dict if None.
            image (np.ndarray, None): The input image as a numpy array. If None, the image is taken from `labels`.

        Returns:
            (Dict | np.ndarray): If `labels` is provided, returns an updated dictionary with the resized and cropped image,
                updated labels, and additional metadata. If `labels` is empty, returns a tuple containing the resized
                and cropped image.
        """
        if labels is None:
            labels = {}

        if random.random() < self.p:
            img = labels.get("img") if image is None else image
            h, w = img.shape[:2]
            crop_h, crop_w = self.target_size
            img = cv2.resize(img, (int(w * self.scale), int(h * self.scale)), interpolation=cv2.INTER_LINEAR)
            new_h, new_w = img.shape[:2]
            img = img[new_h - crop_h : new_h, :crop_w]

            instances = labels.pop("instances", None)
            if instances is not None:
                if instances.normalized:
                    instances.denormalize(w, h)

                cls = labels.pop("cls", [])
                segments = instances.segments
                ins_ids = labels.pop("ins_ids", [])

                y_offset = new_h - crop_h
                x_offset = 0

                new_bboxes_xyxy, new_segments, mask = self.apply_segments(segments, x_offset, y_offset)
                new_cls = cls[mask]
                new_ins_ids = ins_ids[mask]

                assert len(new_cls) == len(new_bboxes_xyxy) == len(new_segments), (
                    f"cls: {len(new_cls)}, bboxes: {len(new_bboxes_xyxy)}, segments: {len(new_segments)}"
                )

                new_instances = Instances(new_bboxes_xyxy, new_segments, bbox_format="xyxy", normalized=False)

                labels["cls"] = new_cls
                labels["ins_ids"] = new_ins_ids
                labels["instances"] = new_instances
                labels["img"] = img
                labels["manitou_resize_crop"] = {"scale": self.scale, "target_size": self.target_size}
                labels["resized_shape"] = self.target_size

            # update the camera intrinsics if available
            intrinsic_K = labels.get("intrinsic_K", None)
            if intrinsic_K is not None:
                labels["intrinsic_K"] = self.update_camera_intrinsics(intrinsic_K)

        if len(labels) > 0:
            return labels
        else:
            return img

    def apply_segments(self, segments, x_offset, y_offset):
        n, num = segments.shape[:2]
        if n == 0:
            return np.zeros((0, 4), dtype=np.float32), segments, np.ones(n, dtype=bool)

        segments = segments * self.scale
        # crop the segments
        segments[:, :, 0] = np.clip(segments[:, :, 0] - x_offset, 0, self.target_size[1])
        segments[:, :, 1] = np.clip(segments[:, :, 1] - y_offset, 0, self.target_size[0])

        bboxes_xyxy = np.stack([segment2box(seg, *self.target_size[:2][::-1]) for seg in segments], axis=0)
        # remove bboxes
        _w = bboxes_xyxy[:, 2] - bboxes_xyxy[:, 0]
        _h = bboxes_xyxy[:, 3] - bboxes_xyxy[:, 1]
        mask = np.logical_and(_w > 5, _h > 5)
        segments = segments[mask]
        bboxes_xyxy = bboxes_xyxy[mask]

        return bboxes_xyxy, segments, mask

    def update_camera_intrinsics(self, intrinsic):
        """Update the camera intrinsics."""
        M = np.array([[self.scale, 0, 0], [0, self.scale, -self.y_offset], [0, 0, 1]], dtype=intrinsic.dtype)
        return M @ intrinsic


def v8_transformsV1(dataset, imgsz, hyp, pre_crop_cfg, stretch=False):
    """Adapted from v8_transforms to support rectangular images."""
    if isinstance(imgsz, int):
        imgsz = [imgsz, imgsz]
    assert isinstance(imgsz, (list, tuple)) and len(imgsz) == 2, (
        f"imgsz should be a list or tuple of length 2 (h, w), but got {imgsz}."
    )

    resize_crop = ManitouResizeCrop(
        pre_crop_cfg["scale"],
        pre_crop_cfg["target_size"],
        pre_crop_cfg["original_size"],
        1.0 if pre_crop_cfg["is_crop"] else 0.0,
    )
    mosaic = MosaicV1(dataset, imgsz=imgsz, p=hyp.mosaic, pre_transform=Compose([resize_crop]))
    affine = RandomPerspective(
        degrees=hyp.degrees,
        translate=hyp.translate,
        scale=hyp.scale,
        shear=hyp.shear,
        perspective=hyp.perspective,
        pre_transform=None if stretch else resize_crop,
    )

    pre_transform = Compose([resize_crop, mosaic, affine])
    if hyp.copy_paste_mode == "flip":
        pre_transform.insert(
            1 + 1, CopyPaste(p=hyp.copy_paste, mode=hyp.copy_paste_mode)
        )  # +1 because of insert `resize_crop` at the beginning
    else:
        pre_transform.append(
            CopyPaste(
                dataset,
                pre_transform=Compose(
                    [
                        resize_crop,
                        MosaicV1(dataset, imgsz=imgsz, p=hyp.mosaic, pre_transform=Compose([resize_crop])),
                        affine,
                    ]
                ),
                p=hyp.copy_paste,
                mode=hyp.copy_paste_mode,
            )
        )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            CutMix(dataset, pre_transform=pre_transform, p=hyp.cutmix),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms


class FormatManitou(Format):
    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = False  # TODO: delete
        self.return_obb = False  # TODO: delete
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels):
        labels.pop("prev", None)  # in case recursive call when using pin_memory=True
        labels.pop("next", None)  # in case recursive call when using pin_memory=True
        return super().__call__(labels)
