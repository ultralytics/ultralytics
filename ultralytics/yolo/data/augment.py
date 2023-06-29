# Ultralytics YOLO 🚀, AGPL-3.0 license

import math
import random
from copy import deepcopy
from typing import Optional, Iterable, Union, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

import pandas as pd
import pkg_resources
import pickle

from ..utils import LOGGER, colorstr
from ..utils.checks import check_version
from ..utils.instance import Instances
from ..utils.metrics import bbox_ioa
from ..utils.ops import segment2box
from .utils import polygons2masks, polygons2masks_overlap

POSE_FLIPLR_INDEX = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


# TODO: we might need a BaseTransform to make all these augments be compatible with both classification and semantic
class BaseTransform:
    def __init__(self) -> None:
        pass

    def apply_image(self, labels):
        """Applies image transformation to labels."""
        pass

    def apply_instances(self, labels):
        """Applies transformations to input 'labels' and returns object instances."""
        pass

    def apply_semantic(self, labels):
        """Applies semantic segmentation to an image."""
        pass

    def __call__(self, labels):
        """Applies label transformations to an image, instances and semantic masks."""
        self.apply_image(labels)
        self.apply_instances(labels)
        self.apply_semantic(labels)


class Compose:
    def __init__(self, transforms):
        """Initializes the Compose object with a list of transforms."""
        self.transforms = transforms

    def __call__(self, data):
        """Applies a series of transformations to input data."""
        for t in self.transforms:
            data = t(data)
        return data

    def append(self, transform):
        """Appends a new transform to the existing list of transforms."""
        self.transforms.append(transform)

    def tolist(self):
        """Converts list of transforms to a standard Python list."""
        return self.transforms

    def __repr__(self):
        """Return string representation of object."""
        format_string = f"{self.__class__.__name__}("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string


class BaseMixTransform:
    """This implementation is from mmyolo."""

    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
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
        mix_labels = [self.dataset.get_label_info(i) for i in indexes]

        if self.pre_transform is not None:
            for i, data in enumerate(mix_labels):
                mix_labels[i] = self.pre_transform(data)
        labels["mix_labels"] = mix_labels

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


class Mosaic(BaseMixTransform):
    """Mosaic augmentation.
    Args:
        imgsz (Sequence[int]): Image size after mosaic pipeline of single
            image. The shape order should be (height, width).
            Default to (640, 640).
    """

    def __init__(self, dataset, imgsz=640, p=1.0, border=(0, 0)):
        """Initializes the object with a dataset, image size, probability, and border."""
        assert 0 <= p <= 1.0, "The probability should be in range [0, 1]. " f"got {p}."
        super().__init__(dataset=dataset, p=p)
        self.dataset = dataset
        self.imgsz = imgsz
        self.border = border

    def get_indexes(self):
        """Return a list of 3 random indexes from the dataset."""
        return [random.randint(0, len(self.dataset) - 1) for _ in range(3)]

    def _mix_transform(self, labels):
        """Apply mixup transformation to the input image and labels."""
        mosaic_labels = []
        assert labels.get("rect_shape", None) is None, "rect and mosaic is exclusive."
        assert (
            len(labels.get("mix_labels", [])) > 0
        ), "There are no other images for mosaic augment."
        s = self.imgsz
        yc, xc = (
            int(random.uniform(-x, 2 * s + x)) for x in self.border
        )  # mosaic center x, y
        for i in range(4):
            labels_patch = labels if i == 0 else labels["mix_labels"][i - 1]
            # Load image
            img = labels_patch["img"]
            h, w = labels_patch.pop("resized_shape")

            # Place img in img4
            if i == 0:  # top left
                img4 = np.full(
                    (s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8
                )  # base image with 4 tiles
                x1a, y1a, x2a, y2a = (
                    max(xc - w, 0),
                    max(yc - h, 0),
                    xc,
                    yc,
                )  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = (
                    w - (x2a - x1a),
                    h - (y2a - y1a),
                    w,
                    h,
                )  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            labels_patch = self._update_labels(labels_patch, padw, padh)
            mosaic_labels.append(labels_patch)
        final_labels = self._cat_labels(mosaic_labels)
        final_labels["img"] = img4
        return final_labels

    def _update_labels(self, labels, padw, padh):
        """Update labels."""
        nh, nw = labels["img"].shape[:2]
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(nw, nh)
        labels["instances"].add_padding(padw, padh)
        return labels

    def _cat_labels(self, mosaic_labels):
        """Return labels with mosaic border instances clipped."""
        if len(mosaic_labels) == 0:
            return {}
        cls = []
        instances = []
        for labels in mosaic_labels:
            cls.append(labels["cls"])
            instances.append(labels["instances"])
        final_labels = {
            "im_file": mosaic_labels[0]["im_file"],
            "ori_shape": mosaic_labels[0]["ori_shape"],
            "resized_shape": (self.imgsz * 2, self.imgsz * 2),
            "cls": np.concatenate(cls, 0),
            "instances": Instances.concatenate(instances, axis=0),
            "mosaic_border": self.border,
        }
        final_labels["instances"].clip(self.imgsz * 2, self.imgsz * 2)
        return final_labels


class MixUp(BaseMixTransform):
    def __init__(self, dataset, pre_transform=None, p=0.0) -> None:
        super().__init__(dataset=dataset, pre_transform=pre_transform, p=p)

    def get_indexes(self):
        """Get a random index from the dataset."""
        return random.randint(0, len(self.dataset) - 1)

    def _mix_transform(self, labels):
        """Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf."""
        r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
        labels2 = labels["mix_labels"][0]
        labels["img"] = (labels["img"] * r + labels2["img"] * (1 - r)).astype(np.uint8)
        labels["instances"] = Instances.concatenate(
            [labels["instances"], labels2["instances"]], axis=0
        )
        labels["cls"] = np.concatenate([labels["cls"], labels2["cls"]], 0)
        return labels


class ThetaGuesser:
    def __init__(self):
        CARD_W = 64.2
        CARD_H = 89.5
        phi = np.arctan((CARD_H / CARD_W))

        def bbox_ratio(theta):
            return np.sin(theta + phi) / np.cos(theta - phi)

        def radian(degree):
            return degree / 360.0 * 2 * np.pi

        self.thetas = np.array([radian(d) for d in range(90)])
        self.ratios = np.array([bbox_ratio(t) for t in thetas])

    def guess_theta_from_ratio(self, ratio):
        return np.argmin(np.abs(self.ratios - ratio))


class CustomRandomPerspective:
    def __init__(
        self,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        shear=0.0,
        perspective=0.0,
        border=(0, 0),
        pre_transform=None,
    ):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        # Mosaic border
        self.border = border
        self.pre_transform = pre_transform
        self.initialize_theta_guessing()
        self.setup_models()

    def initialize_theta_guessing(self):
        CARD_W = 64.2
        CARD_H = 89.5
        phi = np.arctan((CARD_H / CARD_W))

        def bbox_ratio(theta):
            return np.sin(theta + phi) / np.cos(theta - phi)

        def radian(degree):
            return degree / 360.0 * 2 * np.pi

        thetas = np.array([radian(d) for d in range(90)])
        self.ratio_at_theta = np.array([bbox_ratio(t) for t in thetas])

    def setup_models(self):
        reg_path = pkg_resources.resource_filename(
            __name__, "models/abs_regressor.pickle"
        )
        with open(reg_path, "rb") as file:
            self.abs_regressor = pickle.load(file)

        cls_path = pkg_resources.resource_filename(
            __name__, "models/sign_classifier.pickle"
        )
        with open(cls_path, "rb") as file:
            self.sign_classifier = pickle.load(file)

    def affine_transform(self, img, border):
        """Center."""
        C = np.eye(3, dtype=np.float32)

        C[0, 2] = -img.shape[1] / 2  # x translation (pixels)
        C[1, 2] = -img.shape[0] / 2  # y translation (pixels)

        # Perspective
        P = np.eye(3, dtype=np.float32)
        P[2, 0] = random.uniform(
            -self.perspective, self.perspective
        )  # x perspective (about y)
        P[2, 1] = random.uniform(
            -self.perspective, self.perspective
        )  # y perspective (about x)

        # Rotation and Scale
        R = np.eye(3, dtype=np.float32)
        a = random.uniform(-self.degrees, self.degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

        # Shear
        S = np.eye(3, dtype=np.float32)
        S[0, 1] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # x shear (deg)
        S[1, 0] = math.tan(
            random.uniform(-self.shear, self.shear) * math.pi / 180
        )  # y shear (deg)

        # Translation
        T = np.eye(3, dtype=np.float32)
        T[0, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[0]
        )  # x translation (pixels)
        T[1, 2] = (
            random.uniform(0.5 - self.translate, 0.5 + self.translate) * self.size[1]
        )  # y translation (pixels)

        # Combined rotation matrix
        M = T @ S @ R @ P @ C  # order of operations (right to left) is IMPORTANT
        # Affine image
        if (
            (border[0] != 0) or (border[1] != 0) or (M != np.eye(3)).any()
        ):  # image changed
            if self.perspective:
                img = cv2.warpPerspective(
                    img, M, dsize=self.size, borderValue=(114, 114, 114)
                )
            else:  # affine
                img = cv2.warpAffine(
                    img, M[:2], dsize=self.size, borderValue=(114, 114, 114)
                )
        return img, M, a, s

    def guess_theta_abs(self, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[1] - bbox[3]
        ratio = h / w * 1080.0 / 1920.0

    # fn rgb_to_gray(rgb: ArrayView1<u8>) -> f32 {
    #     (0.2126 * rgb[0] as f32 + 0.7152 * rgb[1] as f32 + 0.0722 * rgb[2] as f32) / 255.0
    # }

    def extract_all_pixels(self, img, x, y, w, h, theta_abs):
        # first left rotation, i.e. theta is negative
        # to find actual card width height
        # TODO here outliers should be handled
        img_h, img_w = img.shape[:2]
        n = theta_abs.shape[0]

        # Denormalized xywh
        x = x * img_w
        y = y * img_h
        w = w * img_w
        h = h * img_h
        
        
        aspect_ratio = 89.5 / 64.2
        h_card = h / (
            np.abs(np.cos(theta_abs / 180.0 * np.pi))
            + np.abs(np.sin(theta_abs / 180.0 * np.pi)) / aspect_ratio
        )
        w_card = h_card / aspect_ratio

        # to corners left and convert to int
        corners_left = self.obbs_to_corners(x,y,w_card, h_card, -theta_abs)
        corners_left = np.concatenate(corners_left, axis=0).astype(int)
        # index img by corners
        x_pos = np.maximum(np.minimum(corners_left[:,0], img_w - 1), 0)
        y_pos = np.maximum(np.minimum(corners_left[:,1], img_h - 1), 0)
        pixels_left = img[y_pos, x_pos]
        # print("pixels_left")
        # print(pixels_left.shape)
        grayscale_left = (
            0.2126 * pixels_left[:, 0] + 0.7152 * pixels_left[:, 1] + 0.0722 * pixels_left[:, 2]
        ) / 255.0
        normalized_left = grayscale_left - 0.5

        # to corners right
        corners_right = self.obbs_to_corners(x,y,w_card, h_card, -theta_abs)
        corners_right = np.concatenate(corners_right, axis=0).astype(int)
        # index img by corners
        x_pos = np.maximum(np.minimum(corners_right[:,0], img_w - 1), 0)
        y_pos = np.maximum(np.minimum(corners_right[:,1], img_h - 1), 0)
        pixels_right = img[y_pos, x_pos]
        grayscale_right = (
            0.2126 * pixels_right[:, 0] + 0.7152 * pixels_right[:, 1] + 0.0722 * pixels_right[:, 2]
        ) / 255.0
        normalized_right = grayscale_right - 0.5

        pixels = np.concatenate([normalized_left.reshape(n,4), normalized_right.reshape(n,4)], axis=1)
        return pixels

    def extract_pixels(self, img, x_center, y_center, w, h, theta):
        img_w = img.shape[1]
        img_h = img.shape[0]
        # print(img_w, img_h)
        # print(f"Rotating by {theta}")
        img = cv2.circle(
            img,
            (int(x_center * img_w), int(y_center * img_h)),
            radius=2,
            thickness=1,
            color=(0, 255, 0),
        )
        # print(f"center {x_center, y_center}")

        # TODO extract to global variable, or to self
        inset = 0.1  # taken the same as in rust code which generates the dataset
        aspect_ratio = 89.5 / 64.2
        # print(f"bbox_w {w}")
        # print(f"bbox_h {h}")
        card_h = h / (
            np.cos(theta / 180.0 * np.pi) + np.sin(theta / 180.0 * np.pi) / aspect_ratio
        )
        card_w = card_h / aspect_ratio
        card_w_norm = card_w * (1 - inset)
        card_h_norm = card_h * (1 - inset)
        # print(f"card_w {card_w_norm}")
        # print(f"card_h {card_h_norm}")
        x, y, w2, h2 = x_center, y_center, card_w_norm / 2, card_h_norm / 2
        no_rot = np.array(
            [
                [-w2, -h2],
                [-w2, +h2],
                [+w2, -h2],
                [+w2, +h2],
            ]
        )  # tl, bl, tr, br
        th = theta / 180.0 * np.pi
        L = np.array([[np.cos(th), np.sin(th)], [-np.sin(th), np.cos(th)]])
        R = np.array([[np.cos(-th), np.sin(-th)], [-np.sin(-th), np.cos(-th)]])
        # print(L)
        # print((L @ no_rot.T).T)
        left = (L @ no_rot.T).T + np.array([x, y])
        right = (R @ no_rot.T).T + np.array([x, y])
        # print(left)
        values = []

        img_gray = (
            0.2126 * img[:, :, 0] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 2]
        ) / 255.0
        img_mean = np.mean(img_gray)
        img_var = np.std(img_gray)

        # debug_img = img.copy()
        for corner in list(left) + list(right):
            x_pos = int(corner[0] * img_w)
            x_pos = max(min(x_pos, img_w - 1), 0)
            y_pos = int(corner[1] * img_h)
            y_pos = max(min(y_pos, img_h - 1), 0)
            pixel = img[y_pos, x_pos, :]
            # print(x_pos, y_pos)
            grayscale = (
                0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2]
            ) / 255.0
            # normalized = (grayscale - img_mean) / img_var
            normalized = grayscale - 0.5
            values.append(normalized)
            # debug_img = cv2.circle(
            #     debug_img, (x_pos, y_pos), radius=2, thickness=1, color=(0, 0, 255)
            # )
        # print(values)
        # cv2.imshow("debug", debug_img)
        # cv2.waitKey(0)
        return values

    def to_oriented(self, bboxes, img):
        # print("bboxes", bboxes)
        scale_w = img.shape[1]
        scale_h = img.shape[0]
        # First guess based on inverse of function
        norm_bboxes = bboxes * np.array(
            [1 / scale_w, 1 / scale_h, 1 / scale_w, 1 / scale_h]
        )
        # norm_bboxes = bboxes
        # print("norm_bboxes", norm_bboxes)
        width = np.maximum(np.abs(norm_bboxes[:, 2] - norm_bboxes[:, 0]), 0.01)
        height = np.abs(norm_bboxes[:, 3] - norm_bboxes[:, 1])
        # print(w)
        # print(h)
        ratio = (
            height / width
        )  # w and h are normalized, but the img size is square! This only works with square img size
        ratio[ratio == np.inf] = 89.5 / 64.2
        # ratio[ratio == -np.inf] = 89.5 / 64.2
        # print(ratio)
        guessed_plus = np.argmin(
            np.abs(self.ratio_at_theta - ratio[:, np.newaxis]), axis=1
        )
        # print(guessed_plus)

        # Setup features for model
        x_center = (norm_bboxes[:, 0] + norm_bboxes[:, 2]) / 2.0
        # print("x_center")
        # print(x_center)
        y_center = (norm_bboxes[:, 1] + norm_bboxes[:, 3]) / 2.0
        # print(x_center)
        x_offset = x_center - 0.5
        x_dist2 = x_offset**2
        y_dist2 = (y_center - 1.0) ** 2
        # features from corners of guessed theta obb left and right
        pixels = self.extract_all_pixels(img, x_center, y_center, width, height, guessed_plus)
        # pixels = np.array(
        #      [
        #          self.extract_pixels(img, x_c, y_c, w, h, th)
        #          for x_c, y_c, w, h, th in zip(
        #              x_center, y_center, width, height, guessed_plus
        #          )
        #      ]
        #  )

        # n = x_center.shape[0]
        # pixels = np.zeros((n, 8))

        features = pd.DataFrame(
            {
                "x_offset": x_offset,
                "ratio": ratio,
                "guessed_plus": guessed_plus,
                "x_dist2": x_dist2,
                "y_dist2": y_dist2,
                "rot_left_tl": pixels[:, 0],
                "rot_left_bl": pixels[:, 1],
                "rot_left_tr": pixels[:, 2],
                "rot_left_br": pixels[:, 3],
                "rot_right_tl": pixels[:, 4],
                "rot_right_bl": pixels[:, 5],
                "rot_right_tr": pixels[:, 6],
                "rot_right_br": pixels[:, 6],
            }
        )

        # features = np.stack([
        #     x_offset,
        #     ratio,
        #     guessed_plus,
        #     x_dist2,
        #     y_dist2,
        # ], axis=1)
        # print(features.shape)
        # features = np.concatenate([
        #     features,
        #     pixels
        # ], axis=1)
        # print(features.shape)

        # Then apply model on data
        abs = self.abs_regressor.predict(features)
        is_pos = self.sign_classifier.predict(features)
        sign = np.where(is_pos, 1, -1)
        theta = abs * sign
        # print("theta")
        # print(theta)

        aspect_ratio = 89.5 / 64.2
        # print("height")
        # print(height)
        height_card = height / (
            np.abs(np.cos(theta / 180.0 * np.pi))
            + np.abs(np.sin(theta / 180.0 * np.pi)) / aspect_ratio
        )
        width_card = height_card / aspect_ratio
        # print("height_card")
        # print(height_card)
        # print("x_center")
        # print(x_center)

        return x_center, y_center, width_card, height_card, theta

    def obbs_to_corners(self, x, y, w, h, theta):
        tl = np.stack([-w / 2, h / 2], axis=1)
        bl = np.stack([-w / 2, -h / 2], axis=1)
        tr = np.stack([w / 2, h / 2], axis=1)
        br = np.stack([w / 2, -h / 2], axis=1)

        xy = np.stack([x, y], axis=1)
        # print("xy")
        # print(xy)

        n = theta.shape[0]
        th = theta / 180.0 * np.pi
        # print("Rotation matrices", ROT)
        # print(tl)
        # tl = (ROT @ tl.T).T
        # print("th")
        # print(th)
        tl_rot = tl.copy()
        # print("tl")
        # print(tl)
        sin_th = np.sin(th)
        cos_th = np.cos(th)
        tl_rot[:, 0] = cos_th * tl[:, 0] - sin_th * tl[:, 1]
        tl_rot[:, 1] = sin_th * tl[:, 0] + cos_th * tl[:, 1]
        # print("tl_rot")
        # print(tl_rot)
        bl_rot = bl.copy()
        bl_rot[:, 0] = cos_th * bl[:, 0] - sin_th * bl[:, 1]
        bl_rot[:, 1] = sin_th * bl[:, 0] + cos_th * bl[:, 1]
        tr_rot = tr.copy()
        tr_rot[:, 0] = cos_th * tr[:, 0] - sin_th * tr[:, 1]
        tr_rot[:, 1] = sin_th * tr[:, 0] + cos_th * tr[:, 1]
        br_rot = br.copy()
        br_rot[:, 0] = cos_th * br[:, 0] - sin_th * br[:, 1]
        br_rot[:, 1] = sin_th * br[:, 0] + cos_th * br[:, 1]
        return tl_rot + xy, bl_rot + xy, tr_rot + xy, br_rot + xy

    def obb_corners_to_xyxy(self, tl, bl, tr, br):
        # x = xy[:, [0, 2, 4, 6]]
        # y = xy[:, [1, 3, 5, 7]]
        n = tl.shape[0]
        x = np.stack([tl[:, 0], bl[:, 0], tr[:, 0], br[:, 0]], axis=1)
        y = np.stack([tl[:, 1], bl[:, 1], tr[:, 1], br[:, 1]], axis=1)
        # print("x shape", x.shape)
        # print("x min", x.min(1))
        xyxy = np.stack([x.min(1), y.min(1), x.max(1), y.max(1)]).T
        # print("xyxy", xyxy.shape)
        return xyxy

    def apply_bboxes(self, bboxes, img_rotated, img, phi, M):
        """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        # Here we need to compute the oriented bboxes as an array n x 2
        x, y, w, h, theta = self.to_oriented(bboxes, img)

        # Then rotate them
        ## To corners
        tl, bl, tr, br = self.obbs_to_corners(x, y, w, h, theta)
        corners = np.concatenate([tl, bl, tr, br], axis=0)
        ## Unnormalize
        img_h, img_w = img.shape[:2]
        corners = corners * np.array([img_w, img_h])

        ## Rotate, translate, scale, etc with M
        corners_rot = np.ones((n * 4, 3))
        corners_rot[:, :2] = corners
        corners_rot = corners_rot @ M.T
        n = tl.shape[0]
        tl, bl, tr, br = (
            corners_rot[0:n, :2],
            corners_rot[n : 2 * n, :2],
            corners_rot[2 * n : 3 * n, :2],
            corners_rot[3 * n : 4 * n, :2],
        )
        xyxy = self.obb_corners_to_xyxy(tl, bl, tr, br)

        # theta = theta + phi
        # x_rot = np.cos(phi / 180.0 * np.pi) * x - np.sin(phi / 180.0 * np.pi) * y
        # y_rot = np.sin(phi / 180.0 * np.pi) * x + np.cos(phi / 180.0 * np.pi) * y

        # # To corners
        # corners = self.obbs_to_corners(x_rot, y_rot, w, h, theta)
        # print("corners")
        # print(corners)

        # # To normal bbox xyxy format
        # xyxy = self.obb_corners_to_xyxy(*corners)
        # print("xyxy")
        # print(xyxy)

        # Denormalize
        # img_h, img_w = img.shape[:2]
        # xyxy *= np.array([img_w, img_h, img_w, img_h])

        # debug_img = img_rotated.copy()
        # for thing in xyxy:
        #     print(thing)
        #     pos_0 = thing[0:2]
        #     print(pos_0)
        #     debug_img = cv2.circle(
        #         debug_img,
        #         (int(pos_0[0]), int(pos_0[1])),
        #         radius=4,
        #         thickness=2,
        #         color=(255, 0, 255),
        #     )

        # cv2.imshow("debug", debug_img)
        # cv2.waitKey(0)
        # cv2.imshow("debug", img_rotated)
        # cv2.waitKey(0)

        return xyxy

    def apply_segments(self, segments, M):
        """
        Apply affine to segments and generate new bboxes from segments.

        Args:
            segments (ndarray): list of segments, [num_samples, 500, 2].
            M (ndarray): affine matrix.

        Returns:
            new_segments (ndarray): list of segments after affine, [num_samples, 500, 2].
            new_bboxes (ndarray): bboxes after affine, [N, 4].
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack(
            [segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0
        )
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        """
        Apply affine to keypoints.

        Args:
            keypoints (ndarray): keypoints, [N, 17, 3].
            M (ndarray): affine matrix.

        Return:
            new_keypoints (ndarray): keypoints after affine, [N, 17, 3].
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (
            (xy[:, 0] < 0)
            | (xy[:, 1] < 0)
            | (xy[:, 0] > self.size[0])
            | (xy[:, 1] > self.size[1])
        )
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and "mosaic_border" not in labels:
            labels = self.pre_transform(labels)
            labels.pop("ratio_pad")  # do not need ratio pad

        img_orig = labels["img"]
        cls = labels["cls"]
        instances = labels.pop("instances")
        border = labels.pop("mosaic_border", self.border)

        self.size = (
            img_orig.shape[1] + border[1] * 2,
            img_orig.shape[0] + border[0] * 2,
        )  # w, h
        # M is affine matrix
        # scale for func:`box_candidates`
        img, M, a, scale = self.affine_transform(img_orig, border)

        # Make sure the coord formats are right
        instances.convert_bbox(format="xyxy")
        # instances.denormalize(*img_orig.shape[:2][::-1])
        bboxes = self.apply_bboxes(instances.bboxes, img, img_orig, a, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # # Update bboxes if there are segments.
        # if len(segments):
        #     bboxes, segments = self.apply_segments(segments, M)

        # if keypoints is not None:
        #     keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(
            bboxes, segments, keypoints, bbox_format="xyxy", normalized=False
        )
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(
            box1=instances.bboxes.T,
            box2=new_instances.bboxes.T,
            area_thr=0.01 if len(segments) else 0.10,
        )
        labels["instances"] = new_instances[i]
        labels["cls"] = cls[i]
        labels["img"] = img
        labels["resized_shape"] = img.shape[:2]
        return labels

    def box_candidates(
        self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16
    ):  # box1(4,n), box2(4,n)
        # Compute box candidates: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (
            (w2 > wh_thr)
            & (h2 > wh_thr)
            & (w2 * h2 / (w1 * h1 + eps) > area_thr)
            & (ar < ar_thr)
        )  # candidates


class RandomPerspective:

    def __init__(self,
                 degrees=0.0,
                 translate=0.1,
                 scale=0.5,
                 shear=0.0,
                 perspective=0.0,
                 border=(0, 0),
                 pre_transform=None):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.perspective = perspective
        # Mosaic border
        self.border = border
        self.pre_transform = pre_transform

    def affine_transform(self, img, border):
        """Center."""
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
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - self.scale, 1 + self.scale)
        # s = 2 ** random.uniform(-scale, scale)
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

    def apply_bboxes(self, bboxes, M):
        """
        Apply affine to bboxes only.

        Args:
            bboxes (ndarray): list of bboxes, xyxy format, with shape (num_bboxes, 4).
            M (ndarray): affine matrix.

        Returns:
            new_bboxes (ndarray): bboxes after affine, [num_bboxes, 4].
        """
        n = len(bboxes)
        if n == 0:
            return bboxes

        xy = np.ones((n * 4, 3), dtype=bboxes.dtype)
        xy[:, :2] = bboxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3] if self.perspective else xy[:, :2]).reshape(n, 8)  # perspective rescale or affine

        # Create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        return np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1)), dtype=bboxes.dtype).reshape(4, n).T

    def apply_segments(self, segments, M):
        """
        Apply affine to segments and generate new bboxes from segments.

        Args:
            segments (ndarray): list of segments, [num_samples, 500, 2].
            M (ndarray): affine matrix.

        Returns:
            new_segments (ndarray): list of segments after affine, [num_samples, 500, 2].
            new_bboxes (ndarray): bboxes after affine, [N, 4].
        """
        n, num = segments.shape[:2]
        if n == 0:
            return [], segments

        xy = np.ones((n * num, 3), dtype=segments.dtype)
        segments = segments.reshape(-1, 2)
        xy[:, :2] = segments
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]
        segments = xy.reshape(n, -1, 2)
        bboxes = np.stack([segment2box(xy, self.size[0], self.size[1]) for xy in segments], 0)
        return bboxes, segments

    def apply_keypoints(self, keypoints, M):
        """
        Apply affine to keypoints.

        Args:
            keypoints (ndarray): keypoints, [N, 17, 3].
            M (ndarray): affine matrix.

        Return:
            new_keypoints (ndarray): keypoints after affine, [N, 17, 3].
        """
        n, nkpt = keypoints.shape[:2]
        if n == 0:
            return keypoints
        xy = np.ones((n * nkpt, 3), dtype=keypoints.dtype)
        visible = keypoints[..., 2].reshape(n * nkpt, 1)
        xy[:, :2] = keypoints[..., :2].reshape(n * nkpt, 2)
        xy = xy @ M.T  # transform
        xy = xy[:, :2] / xy[:, 2:3]  # perspective rescale or affine
        out_mask = (xy[:, 0] < 0) | (xy[:, 1] < 0) | (xy[:, 0] > self.size[0]) | (xy[:, 1] > self.size[1])
        visible[out_mask] = 0
        return np.concatenate([xy, visible], axis=-1).reshape(n, nkpt, 3)

    def __call__(self, labels):
        """
        Affine images and targets.

        Args:
            labels (dict): a dict of `bboxes`, `segments`, `keypoints`.
        """
        if self.pre_transform and 'mosaic_border' not in labels:
            labels = self.pre_transform(labels)
            labels.pop('ratio_pad')  # do not need ratio pad

        img = labels['img']
        cls = labels['cls']
        instances = labels.pop('instances')
        # Make sure the coord formats are right
        instances.convert_bbox(format='xyxy')
        instances.denormalize(*img.shape[:2][::-1])

        border = labels.pop('mosaic_border', self.border)
        self.size = img.shape[1] + border[1] * 2, img.shape[0] + border[0] * 2  # w, h
        # M is affine matrix
        # scale for func:`box_candidates`
        img, M, scale = self.affine_transform(img, border)

        bboxes = self.apply_bboxes(instances.bboxes, M)

        segments = instances.segments
        keypoints = instances.keypoints
        # Update bboxes if there are segments.
        if len(segments):
            bboxes, segments = self.apply_segments(segments, M)

        if keypoints is not None:
            keypoints = self.apply_keypoints(keypoints, M)
        new_instances = Instances(bboxes, segments, keypoints, bbox_format='xyxy', normalized=False)
        # Clip
        new_instances.clip(*self.size)

        # Filter instances
        instances.scale(scale_w=scale, scale_h=scale, bbox_only=True)
        # Make the bboxes have the same scale with new_bboxes
        i = self.box_candidates(box1=instances.bboxes.T,
                                box2=new_instances.bboxes.T,
                                area_thr=0.01 if len(segments) else 0.10)
        labels['instances'] = new_instances[i]
        labels['cls'] = cls[i]
        labels['img'] = img
        labels['resized_shape'] = img.shape[:2]
        return labels

    def box_candidates(self, box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
        # Compute box candidates: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
        w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
        w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
        ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
        return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates



class RandomHSV:
    def __init__(self, hgain=0.5, sgain=0.5, vgain=0.5) -> None:
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain

    def __call__(self, labels):
        """Applies random horizontal or vertical flip to an image with a given probability."""
        img = labels["img"]
        if self.hgain or self.sgain or self.vgain:
            r = (
                np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            )  # random gains
            hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
            dtype = img.dtype  # uint8

            x = np.arange(0, 256, dtype=r.dtype)
            lut_hue = ((x * r[0]) % 180).astype(dtype)
            lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
            lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

            im_hsv = cv2.merge(
                (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
            )
            cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed
        return labels


class RandomFlip:
    def __init__(self, p=0.5, direction="horizontal", flip_idx=None) -> None:
        assert direction in [
            "horizontal",
            "vertical",
        ], f"Support direction `horizontal` or `vertical`, got {direction}"
        assert 0 <= p <= 1.0

        self.p = p
        self.direction = direction
        self.flip_idx = flip_idx

    def __call__(self, labels):
        """Resize image and padding for detection, instance segmentation, pose."""
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
                instances.keypoints = np.ascontiguousarray(
                    instances.keypoints[:, self.flip_idx, :]
                )
        labels["img"] = np.ascontiguousarray(img)
        labels["instances"] = instances
        return labels


class LetterBox:
    """Resize image and padding for detection, instance segmentation, pose."""

    def __init__(
        self, new_shape=(640, 640), auto=False, scaleFill=False, scaleup=True, stride=32
    ):
        """Initialize LetterBox object with specific parameters."""
        self.new_shape = new_shape
        self.auto = auto
        self.scaleFill = scaleFill
        self.scaleup = scaleup
        self.stride = stride

    def __call__(self, labels=None, image=None):
        """Return updated labels and image with added border."""
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
            ratio = (
                new_shape[1] / shape[1],
                new_shape[0] / shape[0],
            )  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2
        if labels.get("ratio_pad"):
            labels["ratio_pad"] = (labels["ratio_pad"], (dw, dh))  # for evaluation

        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )  # add border

        if len(labels):
            labels = self._update_labels(labels, ratio, dw, dh)
            labels["img"] = img
            labels["resized_shape"] = new_shape
            return labels
        else:
            return img

    def _update_labels(self, labels, ratio, padw, padh):
        """Update labels."""
        labels["instances"].convert_bbox(format="xyxy")
        labels["instances"].denormalize(*labels["img"].shape[:2][::-1])
        labels["instances"].scale(*ratio)
        labels["instances"].add_padding(padw, padh)
        return labels


class CopyPaste:
    def __init__(self, p=0.5) -> None:
        self.p = p

    def __call__(self, labels):
        """Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)."""
        im = labels["img"]
        cls = labels["cls"]
        h, w = im.shape[:2]
        instances = labels.pop("instances")
        instances.convert_bbox(format="xyxy")
        instances.denormalize(w, h)
        if self.p and len(instances.segments):
            n = len(instances)
            _, w, _ = im.shape  # height, width, channels
            im_new = np.zeros(im.shape, np.uint8)

            # Calculate ioa first then select indexes randomly
            ins_flip = deepcopy(instances)
            ins_flip.fliplr(w)

            ioa = bbox_ioa(
                ins_flip.bboxes, instances.bboxes
            )  # intersection over area, (N, M)
            indexes = np.nonzero((ioa < 0.30).all(1))[0]  # (N, )
            n = len(indexes)
            for j in random.sample(list(indexes), k=round(self.p * n)):
                cls = np.concatenate((cls, cls[[j]]), axis=0)
                instances = Instances.concatenate((instances, ins_flip[[j]]), axis=0)
                cv2.drawContours(
                    im_new,
                    instances.segments[[j]].astype(np.int32),
                    -1,
                    (1, 1, 1),
                    cv2.FILLED,
                )

            result = cv2.flip(im, 1)  # augment segments (flip left-right)
            i = cv2.flip(im_new, 1).astype(bool)
            im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

        labels["img"] = im
        labels["cls"] = cls
        labels["instances"] = instances
        return labels


from albumentations.core.transforms_interface import DualTransform
from albumentations.augmentations.dropout.cutout import cutout


class CoarseDropout(DualTransform):
    """CoarseDropout of the rectangular regions in the image.

    Args:
        max_holes (int): Maximum number of regions to zero out.
        max_height (int, float): Maximum height of the hole.
        If float, it is calculated as a fraction of the image height.
        max_width (int, float): Maximum width of the hole.
        If float, it is calculated as a fraction of the image width.
        min_holes (int): Minimum number of regions to zero out. If `None`,
            `min_holes` is be set to `max_holes`. Default: `None`.
        min_height (int, float): Minimum height of the hole. Default: None. If `None`,
            `min_height` is set to `max_height`. Default: `None`.
            If float, it is calculated as a fraction of the image height.
        min_width (int, float): Minimum width of the hole. If `None`, `min_height` is
            set to `max_width`. Default: `None`.
            If float, it is calculated as a fraction of the image width.

        fill_value (int, float, list of int, list of float): value for dropped pixels.
        mask_fill_value (int, float, list of int, list of float): fill value for dropped pixels
            in mask. If `None` - mask is not affected. Default: `None`.

    Targets:
        image, mask, keypoints

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """

    def __init__(
        self,
        max_holes: int = 8,
        max_height: int = 8,
        max_width: int = 8,
        min_holes: Optional[int] = None,
        min_height: Optional[int] = None,
        min_width: Optional[int] = None,
        fill_value: int = 0,
        mask_fill_value: Optional[int] = None,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super(CoarseDropout, self).__init__(always_apply, p)
        self.max_holes = max_holes
        self.max_height = max_height
        self.max_width = max_width
        self.min_holes = min_holes if min_holes is not None else max_holes
        self.min_height = min_height if min_height is not None else max_height
        self.min_width = min_width if min_width is not None else max_width
        self.fill_value = fill_value
        self.mask_fill_value = mask_fill_value
        if not 0 < self.min_holes <= self.max_holes:
            raise ValueError(
                "Invalid combination of min_holes and max_holes. Got: {}".format(
                    [min_holes, max_holes]
                )
            )

        self.check_range(self.max_height)
        self.check_range(self.min_height)
        self.check_range(self.max_width)
        self.check_range(self.min_width)

        if not 0 < self.min_height <= self.max_height:
            raise ValueError(
                "Invalid combination of min_height and max_height. Got: {}".format(
                    [min_height, max_height]
                )
            )
        if not 0 < self.min_width <= self.max_width:
            raise ValueError(
                "Invalid combination of min_width and max_width. Got: {}".format(
                    [min_width, max_width]
                )
            )

    def check_range(self, dimension):
        if isinstance(dimension, float) and not 0 <= dimension < 1.0:
            raise ValueError(
                "Invalid value {}. If using floats, the value should be in the range [0.0, 1.0)".format(
                    dimension
                )
            )

    def apply(
        self,
        img: np.ndarray,
        fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params,
    ) -> np.ndarray:
        return cutout(img, holes, fill_value)

    def apply_to_mask(
        self,
        img: np.ndarray,
        mask_fill_value: Union[int, float] = 0,
        holes: Iterable[Tuple[int, int, int, int]] = (),
        **params,
    ) -> np.ndarray:
        if mask_fill_value is None:
            return img
        return cutout(img, holes, mask_fill_value)

    def apply_to_bbox(self, bbox, **params):
        return bbox

    def get_params_dependent_on_targets(self, params):
        img = params["image"]
        height, width = img.shape[:2]

        holes = []
        for _n in range(random.randint(self.min_holes, self.max_holes)):
            if all(
                [
                    isinstance(self.min_height, int),
                    isinstance(self.min_width, int),
                    isinstance(self.max_height, int),
                    isinstance(self.max_width, int),
                ]
            ):
                hole_height = random.randint(self.min_height, self.max_height)
                hole_width = random.randint(self.min_width, self.max_width)
            elif all(
                [
                    isinstance(self.min_height, float),
                    isinstance(self.min_width, float),
                    isinstance(self.max_height, float),
                    isinstance(self.max_width, float),
                ]
            ):
                hole_height = int(
                    height * random.uniform(self.min_height, self.max_height)
                )
                hole_width = int(width * random.uniform(self.min_width, self.max_width))
            else:
                raise ValueError(
                    "Min width, max width, \
                    min height and max height \
                    should all either be ints or floats. \
                    Got: {} respectively".format(
                        [
                            type(self.min_width),
                            type(self.max_width),
                            type(self.min_height),
                            type(self.max_height),
                        ]
                    )
                )

            y1 = random.randint(0, height - hole_height)
            x1 = random.randint(0, width - hole_width)
            y2 = y1 + hole_height
            x2 = x1 + hole_width
            holes.append((x1, y1, x2, y2))

        return {"holes": holes}

    @property
    def targets_as_params(self):
        return ["image"]

    def get_transform_init_args_names(self):
        return (
            "max_holes",
            "max_height",
            "max_width",
            "min_holes",
            "min_height",
            "min_width",
            "fill_value",
            "mask_fill_value",
        )


class Albumentations:
    # YOLOv8 Albumentations class (optional, only used if package is installed)
    def __init__(self, p=1.0):
        """Initialize the transform object for YOLO bbox formatted params."""
        self.p = p
        self.transform = None
        prefix = colorstr("albumentations: ")
        try:
            import albumentations as A

            check_version(A.__version__, "1.0.3", hard=True)  # version requirement

            T = [
                A.RandomBrightnessContrast(
                    brightness_limit=(-0.2, 0.3), contrast_limit=0.3, p=0.8
                ),
                A.Blur(blur_limit=(1, 3), p=0.3),
                A.GaussNoise(var_limit=(20, 200), p=0.6),
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.4
                ),
                A.ISONoise(intensity=(0.1, 0.5), color_shift=(0.01, 0.05), p=0.25),
                CoarseDropout(
                    max_holes=100,
                    max_height=40,
                    max_width=40,
                    min_holes=50,
                    min_height=5,
                    min_width=5,
                    fill_value=0,
                    p=0.5,
                ),
            ]
            self.transform = A.Compose(
                T,
                bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
            )

            LOGGER.info(
                prefix
                + ", ".join(
                    f"{x}".replace("always_apply=False, ", "") for x in T if x.p
                )
            )
        except ImportError:  # package not installed, skip
            pass
        except Exception as e:
            print(e)
            LOGGER.info(f"{prefix}{e}")

    def __call__(self, labels):
        """Generates object detections and returns a dictionary with detection results."""
        im = labels["img"]
        cls = labels["cls"]
        if len(cls):
            labels["instances"].convert_bbox("xywh")
            labels["instances"].normalize(*im.shape[:2][::-1])
            bboxes = labels["instances"].bboxes
            # TODO: add supports of segments and keypoints
            if self.transform and random.random() < self.p:
                new = self.transform(
                    image=im, bboxes=bboxes, class_labels=cls
                )  # transformed
                if len(new["class_labels"]) > 0:  # skip update if no bbox in new im
                    labels["img"] = new["image"]
                    labels["cls"] = np.array(new["class_labels"])
                    bboxes = np.array(new["bboxes"])
            labels["instances"].update(bboxes=bboxes)
        return labels


# TODO: technically this is not an augmentation, maybe we should put this to another files
class Format:
    def __init__(
        self,
        bbox_format="xywh",
        normalize=True,
        return_mask=False,
        return_keypoint=False,
        mask_ratio=4,
        mask_overlap=True,
        batch_idx=True,
    ):
        self.bbox_format = bbox_format
        self.normalize = normalize
        self.return_mask = return_mask  # set False when training detection only
        self.return_keypoint = return_keypoint
        self.mask_ratio = mask_ratio
        self.mask_overlap = mask_overlap
        self.batch_idx = batch_idx  # keep the batch indexes

    def __call__(self, labels):
        """Return formatted image, classes, bounding boxes & keypoints to be used by 'collate_fn'."""
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
                    1 if self.mask_overlap else nl,
                    img.shape[0] // self.mask_ratio,
                    img.shape[1] // self.mask_ratio,
                )
            labels["masks"] = masks
        if self.normalize:
            instances.normalize(w, h)
        labels["img"] = self._format_img(img)
        labels["cls"] = torch.from_numpy(cls) if nl else torch.zeros(nl)
        labels["bboxes"] = (
            torch.from_numpy(instances.bboxes) if nl else torch.zeros((nl, 4))
        )
        if self.return_keypoint:
            labels["keypoints"] = torch.from_numpy(instances.keypoints)
        # Then we can use collate_fn
        if self.batch_idx:
            labels["batch_idx"] = torch.zeros(nl)
        return labels

    def _format_img(self, img):
        """Format the image for YOLOv5 from Numpy array to PyTorch tensor."""
        if len(img.shape) < 3:
            img = np.expand_dims(img, -1)
        img = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])
        img = torch.from_numpy(img)
        return img

    def _format_segments(self, instances, cls, w, h):
        """convert polygon points to bitmap."""
        segments = instances.segments
        if self.mask_overlap:
            masks, sorted_idx = polygons2masks_overlap(
                (h, w), segments, downsample_ratio=self.mask_ratio
            )
            masks = masks[None]  # (640, 640) -> (1, 640, 640)
            instances = instances[sorted_idx]
            cls = cls[sorted_idx]
        else:
            masks = polygons2masks(
                (h, w), segments, color=1, downsample_ratio=self.mask_ratio
            )

        return masks, instances, cls


def v8_transforms(dataset, imgsz, hyp):
    """Convert images to a size suitable for YOLOv8 training."""
    pre_transform = Compose(
        [
            Mosaic(
                dataset, imgsz=imgsz, p=hyp.mosaic, border=[-imgsz // 2, -imgsz // 2]
            ),
            CopyPaste(p=hyp.copy_paste),
            # RandomRotateCustom(degrees=45.0),
            CustomRandomPerspective(
                degrees=hyp.degrees,
                translate=hyp.translate,
                scale=hyp.scale,
                shear=hyp.shear,
                perspective=hyp.perspective,
                pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
            ),
            # RandomPerspective(
            #     degrees=hyp.degrees,
            #     translate=hyp.translate,
            #     scale=hyp.scale,
            #     shear=hyp.shear,
            #     perspective=hyp.perspective,
            #     pre_transform=LetterBox(new_shape=(imgsz, imgsz)),
            # ),
            CopyPaste(p=hyp.copy_paste),
        ]
    )
    flip_idx = dataset.data.get("flip_idx", None)  # for keypoints augmentation
    if dataset.use_keypoints and flip_idx is None and hyp.fliplr > 0.0:
        hyp.fliplr = 0.0
        LOGGER.warning(
            "WARNING ⚠️ No `flip_idx` provided while training keypoints, setting augmentation 'fliplr=0.0'"
        )
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
    size=224, mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
):  # IMAGENET_MEAN, IMAGENET_STD
    # Transforms to apply if albumentations not installed
    if not isinstance(size, int):
        raise TypeError(
            f"classify_transforms() size {size} must be integer, not (list, tuple)"
        )
    if any(mean) or any(std):
        return T.Compose(
            [CenterCrop(size), ToTensor(), T.Normalize(mean, std, inplace=True)]
        )
    else:
        return T.Compose([CenterCrop(size), ToTensor()])


def classify_albumentations(
    augment=True,
    size=224,
    scale=(0.08, 1.0),
    hflip=0.5,
    vflip=0.0,
    jitter=0.4,
    mean=(0.0, 0.0, 0.0),  # IMAGENET_MEAN
    std=(1.0, 1.0, 1.0),  # IMAGENET_STD
    auto_aug=False,
):
    # YOLOv8 classification Albumentations (optional, only used if package is installed)
    prefix = colorstr("albumentations: ")
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2

        check_version(A.__version__, "1.0.3", hard=True)  # version requirement
        if augment:  # Resize and crop
            T = [A.RandomResizedCrop(height=size, width=size, scale=scale)]
            if auto_aug:
                # TODO: implement AugMix, AutoAug & RandAug in albumentation
                LOGGER.info(f"{prefix}auto augmentations are currently not supported")
            else:
                if hflip > 0:
                    T += [A.HorizontalFlip(p=hflip)]
                if vflip > 0:
                    T += [A.VerticalFlip(p=vflip)]
                if jitter > 0:
                    jitter = float(jitter)
                    T += [
                        A.ColorJitter(jitter, jitter, jitter, 0)
                    ]  # brightness, contrast, saturation, 0 hue
        else:  # Use fixed crop for eval set (reproducibility)
            T = [
                A.SmallestMaxSize(max_size=size),
                A.CenterCrop(height=size, width=size),
            ]
        T += [
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]  # Normalize and convert to Tensor
        LOGGER.info(
            prefix
            + ", ".join(f"{x}".replace("always_apply=False, ", "") for x in T if x.p)
        )
        return A.Compose(T)

    except ImportError:  # package not installed, skip
        pass
    except Exception as e:
        LOGGER.info(f"{prefix}{e}")


class ClassifyLetterBox:
    # YOLOv8 LetterBox class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, size=(640, 640), auto=False, stride=32):
        """Resizes image and crops it to center with max dimensions 'h' and 'w'."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size
        self.auto = auto  # pass max size integer, automatically solve for short side using stride
        self.stride = stride  # used with auto

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        r = min(self.h / imh, self.w / imw)  # ratio of new/old
        h, w = round(imh * r), round(imw * r)  # resized image
        hs, ws = (
            math.ceil(x / self.stride) * self.stride for x in (h, w)
        ) if self.auto else self.h, self.w
        top, left = round((hs - h) / 2 - 0.1), round((ws - w) / 2 - 0.1)
        im_out = np.full((self.h, self.w, 3), 114, dtype=im.dtype)
        im_out[top : top + h, left : left + w] = cv2.resize(
            im, (w, h), interpolation=cv2.INTER_LINEAR
        )
        return im_out


class CenterCrop:
    # YOLOv8 CenterCrop class for image preprocessing, i.e. T.Compose([CenterCrop(size), ToTensor()])
    def __init__(self, size=640):
        """Converts an image from numpy array to PyTorch tensor."""
        super().__init__()
        self.h, self.w = (size, size) if isinstance(size, int) else size

    def __call__(self, im):  # im = np.array HWC
        imh, imw = im.shape[:2]
        m = min(imh, imw)  # min dimension
        top, left = (imh - m) // 2, (imw - m) // 2
        return cv2.resize(
            im[top : top + m, left : left + m],
            (self.w, self.h),
            interpolation=cv2.INTER_LINEAR,
        )


class ToTensor:
    # YOLOv8 ToTensor class for image preprocessing, i.e. T.Compose([LetterBox(size), ToTensor()])
    def __init__(self, half=False):
        """Initialize YOLOv8 ToTensor object with optional half-precision support."""
        super().__init__()
        self.half = half

    def __call__(self, im):  # im = np.array HWC in BGR order
        im = np.ascontiguousarray(
            im.transpose((2, 0, 1))[::-1]
        )  # HWC to CHW -> BGR to RGB -> contiguous
        im = torch.from_numpy(im)  # to torch
        im = im.half() if self.half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0-255 to 0.0-1.0
        return im
