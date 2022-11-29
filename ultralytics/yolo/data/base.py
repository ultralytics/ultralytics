import glob
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from ..utils import NUM_THREADS, TQDM_BAR_FORMAT
from .utils import HELP_URL, IMG_FORMATS, LOCAL_RANK


class BaseDataset(Dataset):
    """Base Dataset.
    Args:
        img_path (str): image path.
        pipeline (dict): a dict of image transforms.
        label_path (str): label path, this can also be an ann_file or other custom label path.
    """

    def __init__(
        self,
        img_path,
        img_size=640,
        label_path=None,
        cache=False,
        augment=True,
        hyp=None,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.5,
        single_cls=False,
    ):
        super().__init__()
        self.img_path = img_path
        self.img_size = img_size
        self.label_path = label_path
        self.augment = augment
        self.prefix = prefix

        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        if single_cls:
            self.update_labels(include_class=[], single_cls=single_cls)

        self.ni = len(self.im_files)

        # rect stuff
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
            assert self.batch_size is not None
            self.set_rectangle()

        # cache stuff
        self.ims = [None] * self.ni
        self.npy_files = [Path(f).with_suffix(".npy") for f in self.im_files]
        if cache:
            self.cache_images()

        # transforms
        self.transforms = self.build_transforms(hyp=hyp)

    def get_img_files(self, img_path):
        """Read image files."""
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # f = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p) as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # f += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.split(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found"
        except Exception as e:
            raise Exception(f"{self.prefix}Error loading data from {img_path}: {e}\n{HELP_URL}")
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """include_class, filter labels to include only these classes (optional)"""
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = segments[j]
            if self.single_cls:
                self.labels[i]["cls"] = 0

    def load_image(self, i):
        # Loads 1 image from dataset index 'i', returns (im, resized hw)
        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                im = np.load(fn)
            else:  # read image
                im = cv2.imread(f)  # BGR
                assert im is not None, f"Image Not Found {f}"
            h0, w0 = im.shape[:2]  # orig hw
            r = self.img_size / max(h0, w0)  # ratio
            if r != 1:  # if sizes are not equal
                interp = cv2.INTER_LINEAR if (self.augment or r > 1) else cv2.INTER_AREA
                im = cv2.resize(im, (int(w0 * r), int(h0 * r)), interpolation=interp)
            return im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
        return self.ims[i], self.im_hw0[i], self.im_hw[i]  # im, hw_original, hw_resized

    def cache_images(self):
        # cache images to memory or disk
        gb = 0  # Gigabytes of cached images
        self.im_hw0, self.im_hw = [None] * self.ni, [None] * self.ni
        fcn = self.cache_images_to_disk if self.cache == "disk" else self.load_image
        results = ThreadPool(NUM_THREADS).imap(fcn, range(self.ni))
        pbar = tqdm(enumerate(results), total=self.ni, bar_format=TQDM_BAR_FORMAT, disable=LOCAL_RANK > 0)
        for i, x in pbar:
            if self.cache == "disk":
                gb += self.npy_files[i].stat().st_size
            else:  # 'ram'
                self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                gb += self.ims[i].nbytes
            pbar.desc = f"{self.prefix}Caching images ({gb / 1E9:.1f}GB {self.cache})"
        pbar.close()

    def cache_images_to_disk(self, i):
        # Saves an image as an *.npy file for faster loading
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), cv2.imread(self.im_files[i]))

    def set_rectangle(self):
        bi = np.floor(np.arange(self.ni) / self.batch_size).astype(int)  # batch index
        nb = bi[-1] + 1  # number of batches

        s = np.array([x.pop("shape") for x in self.labels])  # hw
        ar = s[:, 0] / s[:, 1]  # aspect ratio
        irect = ar.argsort()
        self.im_files = [self.im_files[i] for i in irect]
        self.labels = [self.labels[i] for i in irect]
        ar = ar[irect]

        # Set training image shapes
        shapes = [[1, 1]] * nb
        for i in range(nb):
            ari = ar[bi == i]
            mini, maxi = ari.min(), ari.max()
            if maxi < 1:
                shapes[i] = [maxi, 1]
            elif mini > 1:
                shapes[i] = [1, 1 / mini]

        self.batch_shapes = np.ceil(np.array(shapes) * self.img_size / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

    def __getitem__(self, index):
        label = self.get_label_info(index)
        if self.augment:
            label["dataset"] = self
        return self.transforms(label)

    def get_label_info(self, index):
        label = self.labels[index].copy()
        img, (h0, w0), (h, w) = self.load_image(index)
        label["img"] = img
        label["ori_shape"] = (h0, w0)
        label["resized_shape"] = (h, w)
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        label = self.update_labels_info(label)
        return label

    def __len__(self):
        return len(self.im_files)

    def update_labels_info(self, label):
        """custom your label format here"""
        return label

    def build_transforms(self, hyp=None):
        """Users can custom augmentations here
        like:
            if self.augment:
                # training transforms
                return Compose([])
            else:
                # val transforms
                return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """Users can custom their own format here.
        Make sure your output is a list with each element like below:
            dict(
                im_file=im_file,
                shape=shape,  # format: (height, width)
                cls=cls,
                bboxes=bboxes, # xywh
                segments=segments,  # xy
                keypoints=keypoints, # xy
                normalized=True, # or False
                bbox_format="xyxy",  # or xywh, ltwh
            )
        """
        raise NotImplementedError
