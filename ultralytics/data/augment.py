# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import deepcopy
from typing import Tuple, Union

import cv2
import numpy as np
import torch

from ultralytics.data.utils import polygons2masks, polygons2masks_overlap
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.checks import check_version
from ultralytics.utils.instance import Instances
from ultralytics.utils.metrics import bbox_ioa
from ultralytics.utils.ops import segment2box, xyxyxyxy2xywhr
from ultralytics.utils.torch_utils import TORCHVISION_0_10, TORCHVISION_0_11, TORCHVISION_0_13

DEFAULT_MEAN = (0.0, 0.0, 0.0)
DEFAULT_STD = (1.0, 1.0, 1.0)
DEFAULT_CROP_FRACTION = 1.0


class BaseTransform:
    """
    Base class for image transformations in the Ultralytics library.

    This class serves as a foundation for implementing various image processing operations, designed to be
    compatible with both classification and semantic segmentation tasks.
    """

    def __init__(self) -> None:
        """Initializes the BaseTransform object."""
        pass

    def apply_image(self, labels):
        """Applies image transformations to labels."""
        pass

    def apply_instances(self, labels):
        """Applies transformations to object instances in labels."""
        pass

    def apply_semantic(self, labels):
        """Applies semantic segmentation transformations to labels."""
        pass

    def __call__(self, labels):
        """Applies all label transformations to an image, instances, and semantic masks."""
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:
    """
    A class for composing multiple image transformations.
    """

    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms if isinstance(transforms, list) else [transforms]

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def insert(self, index, transform):
        """Inserts a new transform at a specified index in the list of transforms."""
        self.transforms.insert(index, transform)

    def __getitem__(self, index: Union[list, int]) -> "Compose":
        """Retrieves a specific transform or a set of transforms using indexing."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        index = [index] if isinstance(index, int) else index
        return Compose([self.transforms[i] for i in index])

    def __setitem__(self, index: Union[list, int], value: Union[list, int]) -> None:
        """Sets one or more transforms in the composition using indexing."""
        assert isinstance(index, (int, list)), f"The indices should be either list or int type but got {type(index)}"
        if isinstance(index, list):
            assert isinstance(
                value, list
            ), f"The indices should be the same type as values, but got {type(index)} and {type(value)}"
        if isinstance(index, int):
            index, value = [index], [value]
        for i, v in zip(index, value):
            assert i < len(self.transforms), f"list index {i} out of range {len(self.transforms)}."
            self.transforms[i] = v

    def tolist(self):
        """Converts the list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Returns a string representation of the Compose object."""
        return f"{self.__class__.__name__}({', '.join([f'{t}' for t in self.transforms])})"


class BaseMixTransform:
    """
    Base class for mix transformations like MixUp and Mosaic.

    This class provides a foundation for implementing mix transformations on datasets. It handles the
    probability-based application of transforms and manages the mixing of multiple images and labels.
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initializes the BaseMixTransform object for mix transformations like MixUp and Mosaic."""
        self.dataset = dataset
        self.pre_transform = pre_transform
        self.p = p

    def __call__(self, labels):
        """Applies pre-processing transforms and mixup/mosaic transforms to labels data."""
        if random.uniform(0, 1) > self.p:
            return labels

        # Get index of one or three other images
        indexes = self.get_indexes()
        if isinstance(indexes, int):
            indexes = [indexes]

        # Get images information will be used for Mosaic or MixUp
        mix_labels = [self.dataset.get_image_and_label(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

        # Update cls and texts
        labels = self._update_label_text(labels)
        # Mosaic or MixUp
        labels = self._mix_transform(labels)
        labels.pop("mix_labels", None)
        return labels

    def _mix_transform(self, labels):
        """Applies MixUp or Mosaic augmentation to the label dictionary."""
        raise NotImplementedError

    def get_indexes(self):
        """Gets a list of shuffled indexes for mosaic augmentation."""
        raise NotImplementedError

    def _update_label_text(self, labels):
        """Updates label text and class IDs for mixed labels in image augmentation."""
        if "texts" not in labels:
            return labels

        mix_texts = sum([labels["texts"]] + [x["texts"] for x in labels["mix_labels"]], [])
        mix_texts = list({tuple(x) for x in mix_texts})
        text2id = {text: i for i, text in enumerate(mix_texts)}

        for label in [labels] + labels["mix_labels"]:
            for i, cls in enumerate(label["cls"].squeeze(-1).tolist()):
                text = label["texts"][int(cls)]
                label["cls"][i] = text2id[tuple(text)]
            label["texts"] = mix_texts
        return labels


class Mosaic(BaseMixTransform):
    """
    Mosaic augmentation for image datasets.

    This class performs mosaic augmentation by combining multiple (4 or 9) images into a single mosaic image.
    The augmentation is applied to a dataset with a given probability.
    """

    def __init__(self, dataset, imgsz=640, p=1.0, n=4):
        """Initializes the Mosaic augmentation object."""
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."
        assert n in {4, 9}, "grid must be equal to 4 or 9."
        super().__init__(dataset=dataset, p=p)
        self.imgsz = imgsz
        self.border = (-imgsz // 2, -imgsz // 2)  # width, height
        self.n = n

    def get_indexes(self, buffer=True):
        """Returns a list of random indexes from the dataset for mosaic augmentation."""
        if buffer:  # select images from buffer
            return random.choices(list(self.dataset.buffer), k=self.n - 1)
        else:  # select any images
            return [random.randint(0, len(self.dataset) - 1) for _ in range(self.n - 1)]

    def _mix_transform(self, labels):
        """Applies mosaic augmentation to the input image and labels."""
        assert labels.get("rect_shape", None) is None, "rect and mosaic are mutually exclusive."
        assert len(labels.get("mix_labels", [])), "There are no other images for mosaic augment."
        return (
            self._mosaic3(labels) if self.n == 3 else self._mosaic4(labels) if self.n == 4 else self._mosaic9(labels)
        )  # This code is modified for mosaic3 method.

    # The implementations of _mosaic3, _mosaic4, _mosaic9, _update_labels, and _cat_labels remain the same.


class MixUp(BaseMixTransform):
    """
    Applies MixUp augmentation to image datasets.

    This class implements the MixUp augmentation technique as described in the paper "mixup: Beyond Empirical Risk
    Minimization" (https://arxiv.org/abs/1710.09412). MixUp combines two images and their labels using a random weight.
    """

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        """Initializes the MixUp augmentation object."""
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """Get a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies MixUp augmentation to the input labels."""
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        # Change: Ensure correct data type during mixup
        img_dtype = labels["img"].dtype
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(img_dtype)
        labels["instances"] = Instances.concatenate([labels["instances"], labels2["instances"]], axis=0)
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class RandomPerspective:
    """
    Implements random perspective and affine transformations on images and corresponding annotations.

    This class applies random rotations, translations, scaling, shearing, and perspective transformations
    to images and their associated bounding boxes, segments, and keypoints.
    """

    def __init__(
        self, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, border=(0, 0), pre_transform=None
    ):
        """Initializes RandomPerspective object with transformation parameters."""
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        self.border = border  # mosaic border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """Applies a sequence of affine transformations centered around the image center."""
        # Center
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(-self.perspective, self.perspective)  # x perspective (about y)
        P[2, 1] = random.uniform(-self.perspective, self.perspective)  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        s = random.uniform(1 - self.scale, 1 + self.scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-self.shear, self.shear) * math.pi / 180)  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]  # x translation (pixels)
        T[1, 2] = random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any():  # image changed
            if self.perspective:
                img = cv2.warpPerspective(img, M, dsize=self.size, borderValue=(114, 114, 114))
            else:  # affine
                img = cv2.warpAffine(img, M[:2], dsize=self.size, borderValue=(114, 114, 114))
        return img, M, s

    # The implementations of apply_bboxes, apply_segments, apply_keypoints, __call__, and box_candidates remain the same.


class RandomHSV:
    """
    Randomly adjusts the Hue, Saturation, and Value (HSV) channels of an image.

    This class applies random HSV augmentation to images within predefined limits set by hgain, sgain, and vgain.
    """

    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        """Initializes the RandomHSV object for random HSV augmentation."""
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """Applies random HSV augmentation to an image within predefined limits."""
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            # Determine the data type range
            dtype = img.dtype
            if dtype == np.uint8:
                max_value = 255
                hsv_max = np.array([180, 255, 255], dtype=np.float32)
            elif dtype == np.uint16:
                max_value = 65535
                hsv_max = np.array([180, 65535, 65535], dtype=np.float32)
            else:
                LOGGER.warning(f"RandomHSV: Unsupported image dtype {dtype}, skipping HSV augmentation.")
                return labels

            img = img.astype(np.float32)
            img /= max_value  # Normalize to [0, 1]

            # Convert to HSV
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)
            r = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1  # random gains

            h = ((h * r[0]) % 1.0).clip(0, 1)
            s = (s * r[1]).clip(0, 1)
            v = (v * r[2]).clip(0, 1)

            img_hsv = cv2.merge((h, s, v))
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            img = (img * max_value).astype(dtype)
            labels["img"] = img
        return labels


class RandomFlip:
    """
    Applies a random horizontal or vertical flip to an image with a given probability.

    This class performs random image flipping and updates corresponding instance annotations such as
    bounding boxes and keypoints.
    """

    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        """Initializes the RandomFlip class with probability and direction."""
        assert direction in {"horizontal", "vertical"}, f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0, f"The probability should be in range [0, 1], but got {p}."

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        """Applies random flip to an image and updates any instances like bounding boxes or keypoints accordingly."""
        img = labels["img"]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xywh")
        h, w = img.shape[:2]
        h = 1 if instances.normalized else h
        w = 1 if instances.normalized else w

        # Flip up-down
        if self.direction == "vertical" and random.random() < self.p:
            img = np.flipud(img)
            instances.flipud(h)
        if self.direction == "horizontal" and random.random() < self.p:
            img = np.fliplr(img)
            instances.fliplr(w)
            # For keypoints
            if self.flip_idx is not None and instances.keypoints is not None:
                instances.keypoints = np.ascontiguousarray(instances.keypoints[:, self.flip_idx, :])
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:
    """
    Resize image and padding for detection, instance segmentation, pose.

    This class resizes and pads images to a specified shape while preserving aspect ratio. It also updates
    corresponding labels and bounding boxes.
    """

    def __init__(self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, center=True, stride=32):
        """Initialize LetterBox object for resizing and padding images."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride
        self.center = center  # Put the image in the middle or top-left

    def __call__(self, labels=None, image=None):
        """Resizes and pads an image for object detection, instance segmentation, or pose estimation tasks."""
        if labels is None:
            labels = {}
        img = labels.get("img") if image is None else image
        shape = img.shape[:2]  # current shape [height, width]
        new_shape = labels.pop("rect_shape", self.new_shape)
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not self.scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if self.auto:  # minimum rectangle
            dw, dh = np.mod(dw, self.stride), np.mod(dh, self.stride)  # wh padding
        elif self.scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        if self.center:
            dw /= 2  # divide padding into 2 sides
            dh /= 2

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)) if self.center else 0, int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)) if self.center else 0, int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (left, top))  # for evaluation

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Updates labels after applying letterboxing to an image."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste:
    """
    Implements Copy-Paste augmentation as described in https://arxiv.org/abs/2012.07177.

    This class applies Copy-Paste augmentation on images and their corresponding instances.
    """

    def __init__(self, p=0.5) -> None:
        """Initializes the CopyPaste augmentation object."""
        self.p = p

    def __call__(self, labels):
        """Applies Copy-Paste augmentation to an image and its instances."""
        im = labels["img"]
        cls = labels["cls"]
        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            _, w, _ = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, np.uint8)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(ins_flip.bboxes, instances.bboxes)  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(im_new, instances.segments[[j]].astype(np.int32), -1, (1, 1, 1), cv2.FILLED)

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]

        labels["img"] = im
        labels["cls"] = cls
        labels["instances"] = instances
        return labels


class Albumentations:
    """
    Albumentations transformations for image augmentation.

    This class applies various image transformations using the Albumentations library.
    """

    def __init__(self, p=1.0):
        """Initialize the Albumentations transform object for YOLO bbox formatted parameters."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")

        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            # List of possible spatial transforms
            spatial_transforms = {
                "Affine",
                "BBoxSafeRandomCrop",
                "CenterCrop",
                "CoarseDropout",
                "Crop",
                "CropAndPad",
                "CropNonEmptyMaskIfExists",
                "D4",
                "ElasticTransform",
                "Flip",
                "GridDistortion",
                "GridDropout",
                "HorizontalFlip",
                "Lambda",
                "LongestMaxSize",
                "MaskDropout",
                "MixUp",
                "Morphological",
                "NoOp",
                "OpticalDistortion",
                "PadIfNeeded",
                "Perspective",
                "PiecewiseAffine",
                "PixelDropout",
                "RandomCrop",
                "RandomCropFromBorders",
                "RandomGridShuffle",
                "RandomResizedCrop",
                "RandomRotate90",
                "RandomScale",
                "RandomSizedBBoxSafeCrop",
                "RandomSizedCrop",
                "Resize",
                "Rotate",
                "SafeRotate",
                "ShiftScaleRotate",
                "SmallestMaxSize",
                "Transpose",
                "VerticalFlip",
                "XYMasking",
            }  # from https://albumentations.ai/docs/getting_started/transforms_and_targets/#spatial-level-transforms

            # Transforms
            T = [
                A.Blur(p=0.01),
                A.MedianBlur(p=0.01),
                A.ToGray(p=0.01),
                A.CLAHE(p=0.01),
                A.RandomBrightnessContrast(p=0.0),
                A.RandomGamma(p=0.0),
                A.ImageCompression(quality_lower=75, p=0.0),
            ]

            # Compose transforms
            self.contains_spatial = any(transform.__class__.__name__ in spatial_transforms for transform in T)
            self.transform = (
                A.Compose(T, bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]))
                if self.contains_spatial
                else A.Compose(T)
            )
            LOGGER.info(prefix + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p))
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        """Applies Albumentations transformations to input labels."""
        if self.transform is None or random.random() > self.p:
            return labels

        if self.contains_spatial:
            cls = labels["cls"]
            if len(cls):
                im = labels["img"]
                labels["instances"].convert_bbox("xywh")
                labels["instances"].normalize(*im.shape[:2][::-1])
                bboxes = labels["instances"].bboxes
                # TODO: add supports of segments and keypoints
                new = self.transform(image=im, bboxes=bboxes, class_labels=cls)  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"], dtype=np.float32)
                labels["instances"].update(bboxes=bboxes)
        else:
            labels["img"] = self.transform(image=labels["img"])["image"]  # transformed

        return labels


class Format:
    """
    A class for formatting image annotations for object detection, instance segmentation, and pose estimation tasks.

    This class standardizes image and instance annotations to be used by the `collate_fn` in PyTorch DataLoader.
    """

    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        return_obb=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
        bgr=0.0,
    ):
        """Initializes the Format class with given parameters for image and instance annotation formatting."""
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.return_obb = return_obb
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes
        self.bgr = bgr

    def __call__(self, labels):
        """Formats image annotations for object detection, instance segmentation, and pose estimation tasks."""
        img = labels.pop("img")
        h, w = img.shape[:2]
        cls = labels.pop("cls")
        instances = labels.pop("instances")
        instances.convert_bbox(format=self.bbox_format)
        instances.denormalize(w, h)
        nl = len(instances)

        if self.return_mask:
            if nl:
                masks, instances, cls = self._format_segments(instances, cls, w, h)
                masks = torch.from_numpy(masks)
            else:
                masks = torch.zeros(
                    1 if self.mask_overlap else nl, img.shape[0] // self.mask_ratio, img.shape[1] // self.mask_ratio
                )
            labels["masks"] = masks
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
            if self.normalize:
                labels["keypoints"][..., 0] /= w
                labels["keypoints"][..., 1] /= h
        if self.return_obb:
            labels["bboxes"] = (
                xyxyxyxy2xywhr(torch.from_numpy(instances.segments)) if len(instances.segments) else torch.zeros((0, 5))
            )
        # NOTE: need to normalize obb in xywhr format for width-height consistency
        if self.normalize:
            labels["bboxes"][:, [0, 2]] /= w
            labels["bboxes"][:, [1, 3]] /= h
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """Formats an image for YOLO from a Numpy array to a PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img[::-1] if random.uniform(0, 1) > self.bgr else img)
        img = torch.from_numpy(img)
        # Change: Normalize image depending on its data type
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.dtype == torch.uint16:
            img = img.float() / 65535.0
        else:
            img = img.float()
        return img

    def _format_segments(self, instances, cls, w, h):
        """Converts polygon segments to bitmap masks."""
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap((h, w), segments, downsample_ratio=self.mask_ratio)
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks((h, w), segments, color=1, downsample_ratio=self.mask_ratio)

        return masks, instances, cls


class RandomLoadText:
    """
    Randomly samples positive and negative texts and updates class indices accordingly.

    This class is responsible for sampling texts from a given set of class texts, including both positive
    (present in the image) and negative (not present in the image) samples. It updates the class indices
    to reflect the sampled texts and can optionally pad the text list to a fixed length.
    """

    def __init__(
        self,
        prompt_format: str = "{}",
        neg_samples: Tuple[int, int] = (80, 80),
        max_samples: int = 80,
        padding: bool = False,
        padding_value: str = "",
    ) -> None:
        """Initializes the RandomLoadText class for randomly sampling positive and negative texts."""
        self.prompt_format = prompt_format
        self.neg_samples = neg_samples
        self.max_samples = max_samples
        self.padding = padding
        self.padding_value = padding_value

    def __call__(self, labels: dict) -> dict:
        """Randomly samples positive and negative texts and updates class indices accordingly."""
        assert "texts" in labels, "No texts found in labels."
        class_texts = labels["texts"]
        num_classes = len(class_texts)
        cls = np.asarray(labels.pop("cls"), dtype=int)
        pos_labels = np.unique(cls).tolist()

        if len(pos_labels) > self.max_samples:
            pos_labels = random.sample(pos_labels, k=self.max_samples)

        neg_samples = min(min(num_classes, self.max_samples) - len(pos_labels), random.randint(*self.neg_samples))
        neg_labels = [i for i in range(num_classes) if i not in pos_labels]
        neg_labels = random.sample(neg_labels, k=neg_samples)

        sampled_labels = pos_labels + neg_labels
        random.shuffle(sampled_labels)

        label2ids = {label: i for i, label in enumerate(sampled_labels)}
        valid_idx = np.zeros(len(labels["instances"]), dtype=bool)
        new_cls = []
        for i, label in enumerate(cls.squeeze(-1).tolist()):
            if label not in label2ids:
                continue
            valid_idx[i] = True
            new_cls.append([label2ids[label]])
        labels["instances"] = labels["instances"][valid_idx]
        labels["cls"] = np.array(new_cls)

        # Randomly select one prompt when there's more than one prompts
        texts = []
        for label in sampled_labels:
            prompts = class_texts[label]
            assert len(prompts) > 0
            prompt = self.prompt_format.format(prompts[random.randrange(len(prompts))])
            texts.append(prompt)

        if self.padding:
            valid_labels = len(pos_labels) + len(neg_labels)
            num_padding = self.max_samples - valid_labels
            if num_padding > 0:
                texts += [self.padding_value] * num_padding

        labels["texts"] = texts
        return labels


def v8_transforms(dataset, imgsz, hyp, stretch=False):
    """
    Applies a series of image transformations for YOLOv8 training.

    This function creates a composition of image augmentation techniques to prepare images for YOLOv8 training.
    It includes operations such as mosaic, copy-paste, random perspective, mixup, and various color adjustments.
    """
    pre_transform = Compose(
        [
            Mosaic(dataset, imgsz=imgsz, p=hyp.mosaic),
            CopyPaste(p=hyp.copy_paste),
            RandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=None if stretch else LetterBox(new_shape=(imgsz, imgsz)),
            ),
        ]
    )
    flip_idx = dataset.data.get("flip_idx", [])  # for keypoints augmentation
    if dataset.use_keypoints:
        kpt_shape = dataset.data.get("kpt_shape", None)
        if len(flip_idx) == 0 and hyp.fliplr > 0.0:
            hyp.fliplr = 0.0
            LOGGER.warning("WARNING ⚠️ No 'flip_idx' array defined in data.yaml, setting augmentation 'fliplr=0.0'")
        elif flip_idx and (len(flip_idx) != kpt_shape[0]):
            raise ValueError(f"data.yaml flip_idx={flip_idx} length must be equal to kpt_shape[0]={kpt_shape[0]}")

    return Compose(
        [
            pre_transform,
            MixUp(dataset, pre_transform=pre_transform, p=hyp.mixup),
            Albumentations(p=1.0),
            RandomHSV(hgain=hyp.hsv_h, sgain=hyp.hsv_s, vgain=hyp.hsv_v),
            RandomFlip(direction="vertical", p=hyp.flipud),
            RandomFlip(direction="horizontal", p=hyp.fliplr, flip_idx=flip_idx),
        ]
    )  # transforms


# Classification augmentations -----------------------------------------------------------------------------------------
def classify_transforms(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    interpolation=cv2.INTER_LINEAR,  # Change: Use OpenCV interpolation
    crop_fraction: float = DEFAULT_CROP_FRACTION,
):
    """
    Creates a composition of image transforms for classification tasks.

    This function generates a sequence of transforms suitable for preprocessing images
    for classification models during evaluation or inference. The transforms include resizing,
    center cropping, conversion to tensor, and normalization.

    Args:
        size (int | tuple): The target size for the transformed image. If an int, it defines the shortest edge. If a
            tuple, it defines (height, width).
        mean (tuple): Mean values for each RGB channel used in normalization.
        std (tuple): Standard deviation values for each RGB channel used in normalization.
        interpolation (int): Interpolation method, e.g., cv2.INTER_LINEAR.
        crop_fraction (float): Fraction of the image to be cropped.

    Returns:
        (callable): A function that applies the transformations to an image.

    Examples:
        >>> transforms = classify_transforms(size=224)
        >>> img = cv2.imread("path/to/image.jpg")
        >>> transformed_img = transforms(img)
    """

    def transform(img):
        if isinstance(size, (tuple, list)):
            assert len(size) == 2, f"'size' tuples must be length 2, not length {len(size)}"
            scale_size = tuple(int(math.floor(x / crop_fraction)) for x in size)
            target_size = size
        else:
            scale_size = int(math.floor(size / crop_fraction))
            scale_size = (scale_size, scale_size)
            target_size = (size, size)

        # Resize the image
        im_h, im_w = img.shape[:2]
        # Determine new size preserving aspect ratio
        if (im_w <= im_h and im_w == scale_size[0]) or (im_h <= im_w and im_h == scale_size[1]):
            resized_img = img
        else:
            # Resize preserving aspect ratio
            r = max(scale_size[0] / im_w, scale_size[1] / im_h)
            new_size = (int(im_w * r), int(im_h * r))
            resized_img = cv2.resize(img, new_size, interpolation=interpolation)

        # Center crop
        im_h, im_w = resized_img.shape[:2]
        crop_h, crop_w = target_size
        start_x = (im_w - crop_w) // 2
        start_y = (im_h - crop_h) // 2
        cropped_img = resized_img[start_y:start_y + crop_h, start_x:start_x + crop_w]

        # Convert to tensor and normalize
        img = cropped_img.astype(np.float32)
        # Change: Normalize image depending on its data type
        if img.dtype == np.uint8:
            img /= 255.0
        elif img.dtype == np.uint16:
            img /= 65535.0

        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img[[2, 1, 0], :, :]  # BGR to RGB
        img = torch.from_numpy(img)
        img = (img - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)
        return img

    return transform


# Classification training augmentations --------------------------------------------------------------------------------
def classify_augmentations(
    size=224,
    mean=DEFAULT_MEAN,
    std=DEFAULT_STD,
    scale=(0.08, 1.0),
    ratio=(3.0 / 4.0, 4.0 / 3.0),
    hflip=0.5,
    vflip=0.0,
    auto_augment=None,
    hsv_h=0.015,  # image HSV-Hue augmentation (fraction)
    hsv_s=0.4,  # image HSV-Saturation augmentation (fraction)
    hsv_v=0.4,  # image HSV-Value augmentation (fraction)
    force_color_jitter=False,
    erasing=0.0,
    interpolation=cv2.INTER_LINEAR,  # Change: Use OpenCV interpolation
):
    """
    Creates a composition of image augmentation transforms for classification tasks.

    This function generates a set of image transformations suitable for training classification models.
    """

    def transform(img):
        # Random Resized Crop
        im_h, im_w = img.shape[:2]
        area = im_h * im_w
        for attempt in range(10):
            target_area = random.uniform(scale[0], scale[1]) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= im_w and 0 < h <= im_h:
                x1 = random.randint(0, im_w - w)
                y1 = random.randint(0, im_h - h)

                img_cropped = img[y1:y1 + h, x1:x1 + w]
                img_resized = cv2.resize(img_cropped, (size, size), interpolation=interpolation)
                break
        else:
            # Fallback
            img_resized = cv2.resize(img, (size, size), interpolation=interpolation)

        # Horizontal Flip
        if hflip > 0 and random.random() < hflip:
            img_resized = cv2.flip(img_resized, 1)

        # Vertical Flip
        if vflip > 0 and random.random() < vflip:
            img_resized = cv2.flip(img_resized, 0)

        # Color Jitter
        if force_color_jitter or not auto_augment:
            if hsv_h or hsv_s or hsv_v:
                hsv_transform = RandomHSV(hgain=hsv_h, sgain=hsv_s, vgain=hsv_v)
                img_resized = hsv_transform({"img": img_resized})["img"]

        # Additional augmentations like auto_augment can be added here if re-implemented without PIL

        # Convert to tensor and normalize
        img = img_resized.astype(np.float32)
        # Change: Normalize image depending on its data type
        if img.dtype == np.uint8:
            img /= 255.0
        elif img.dtype == np.uint16:
            img /= 65535.0

        img = img.transpose(2, 0, 1)  # HWC to CHW
        img = img[[2, 1, 0], :, :]  # BGR to RGB
        img = torch.from_numpy(img)
        img = (img - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)

        # Random Erasing
        if erasing > 0 and random.random() < erasing:
            img = random_erasing(img)

        return img

    return transform


def random_erasing(img, sl=0.02, sh=0.4, r1=0.3):
    """Applies Random Erasing to the input tensor."""
    c, h, w = img.shape
    area = h * w

    for attempt in range(100):
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1 / r1)

        erasing_h = int(round(math.sqrt(target_area * aspect_ratio)))
        erasing_w = int(round(math.sqrt(target_area / aspect_ratio)))

        if erasing_h < h and erasing_w < w:
            x1 = random.randint(0, h - erasing_h)
            y1 = random.randint(0, w - erasing_w)
            img[:, x1:x1 + erasing_h, y1:y1 + erasing_w] = torch.rand(c, erasing_h, erasing_w)
            return img
    return img


class CenterCrop:
    """
    Applies center cropping to images for classification tasks.

    This class performs center cropping on input images, resizing them to a specified size while maintaining the aspect
    ratio.
    """

    def __init__(self, size=640):
        """Initializes the CenterCrop object for image preprocessing."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):
        """Applies center cropping to an input image."""
        # Ensure that the image is a NumPy array
        if not isinstance(im, np.ndarray):
            raise TypeError(f"Expected image as NumPy array, got {type(im)}")
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        im_cropped = im[top:top + m, left:left + m]
        im_resized = cv2.resize(im_cropped, (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return im_resized


# NOTE: The ClassifyLetterBox and ToTensor classes are no longer needed as we have integrated their functionality
# directly into the classify_transforms and classify_augmentations functions.

