# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import glob
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, check_file_speeds
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """
    Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference
    in object detection tasks.

    Attributes:
        img_path (str): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        cv2_flag (int): OpenCV flag for reading images.
        im_files (List[str]): List of image file paths.
        labels (List[Dict]): List of label data dictionaries.
        ni (int): Number of images in the dataset.
        rect (bool): Whether to use rectangular training.
        batch_size (int): Size of batches.
        stride (int): Stride used in the model.
        pad (float): Padding value.
        buffer (list): Buffer for mosaic images.
        max_buffer_length (int): Maximum buffer size.
        ims (list): List of loaded images.
        im_hw0 (list): List of original image dimensions (h, w).
        im_hw (list): List of resized image dimensions (h, w).
        npy_files (List[Path]): List of numpy file paths.
        cache (str): Cache images to RAM or disk during training.
        transforms (callable): Image transformation function.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Set the shape of bounding boxes as rectangles.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        cache=False,
        augment=True,
        hyp=DEFAULT_CFG,
        prefix="",
        rect=False,
        batch_size=16,
        stride=32,
        pad=0.5,
        single_cls=False,
        classes=None,
        fraction=1.0,
        channels=3,
    ):
        """
        Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str): Path to the folder containing images.
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
        self.img_path = img_path
        self.imgsz = imgsz
        self.augment = augment
        self.single_cls = single_cls
        self.prefix = prefix
        self.fraction = fraction
        self.channels = channels
        self.cv2_flag = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
        self.im_files = self.get_img_files(self.img_path)
        self.labels = self.get_labels()
        self.update_labels(include_class=classes)  # single_cls and include_class
        self.ni = len(self.labels)  # number of images
        self.rect = rect
        self.batch_size = batch_size
        self.stride = stride
        self.pad = pad
        if self.rect:
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

    def get_img_files(self, img_path):
        """
        Read image files from the specified path.

        Args:
            img_path (str | List[str]): Path or list of paths to image directories or files.

        Returns:
            (List[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(p / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent) if x.startswith("./") else x for x in t]  # local to global path
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global path (pathlib)
                else:
                    raise FileNotFoundError(f"{self.prefix}{p} does not exist")
            im_files = sorted(x.replace("/", os.sep) for x in f if x.rpartition(".")[-1].lower() in IMG_FORMATS)
            # self.img_files = sorted([x for x in f if x.suffix[1:].lower() in IMG_FORMATS])  # pathlib
            assert im_files, f"{self.prefix}No images found in {img_path}. {FORMATS_HELP_MSG}"
        except Exception as e:
            raise FileNotFoundError(f"{self.prefix}Error loading data from {img_path}\n{HELP_URL}") from e
        if self.fraction < 1:
            im_files = im_files[: round(len(im_files) * self.fraction)]  # retain a fraction of the dataset
        check_file_speeds(im_files, prefix=self.prefix)  # check image read speeds
        return im_files

    def update_labels(self, include_class: Optional[list]):
        """
        Update labels to include only specified classes.

        Args:
            include_class (list, optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i]["keypoints"]
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

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

            h0, w0 = im.shape[:2]  # orig hw
            if rect_mode:  # resize long side to imgsz while maintaining aspect ratio
                r = self.imgsz / max(h0, w0)  # ratio
                if r != 1:  # if sizes are not equal
                    w, h = (min(math.ceil(w0 * r), self.imgsz), min(math.ceil(h0 * r), self.imgsz))
                    im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
            elif not (h0 == w0 == self.imgsz):  # resize by stretching image to square imgsz
                im = cv2.resize(im, (self.imgsz, self.imgsz), interpolation=cv2.INTER_LINEAR)
            if im.ndim == 2:
                im = im[..., None]

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

    def cache_images(self):
        """Cache images to memory or disk for faster training."""
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        fcn, storage = (self.cache_images_to_disk, "Disk") if self.cache == "disk" else (self.load_image, "RAM")
        with ThreadPool(NUM_THREADS) as pool:
            results = pool.imap(fcn, range(self.ni))
            pbar = TQDM(enumerate(results), total=self.ni, disable=LOCAL_RANK > 0)
            for i, x in pbar:
                if self.cache == "disk":
                    b += self.npy_files[i].stat().st_size
                else:  # 'ram'
                    self.ims[i], self.im_hw0[i], self.im_hw[i] = x  # im, hw_orig, hw_resized = load_image(self, i)
                    b += self.ims[i].nbytes
                pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB {storage})"
            pbar.close()

    def cache_images_to_disk(self, i):
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            np.save(f.as_posix(), imread(self.im_files[i]), allow_pickle=False)

    def check_cache_disk(self, safety_margin=0.5):
        """
        Check if there's enough disk space for caching images.

        Args:
            safety_margin (float, optional): Safety margin factor for disk space calculation.

        Returns:
            (bool): True if there's enough disk space, False otherwise.
        """
        import shutil

        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im_file = random.choice(self.im_files)
            im = imread(im_file)
            if im is None:
                continue
            b += im.nbytes
            if not os.access(Path(im_file).parent, os.W_OK):
                self.cache = None
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writeable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin=0.5):
        """
        Check if there's enough RAM for caching images.

        Args:
            safety_margin (float, optional): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        b, gb = 0, 1 << 30  # bytes of cached images, bytes per gigabytes
        n = min(self.ni, 30)  # extrapolate from 30 random images
        for _ in range(n):
            im = imread(random.choice(self.im_files))  # sample image
            if im is None:
                continue
            ratio = self.imgsz / max(im.shape[0], im.shape[1])  # max(h, w)  # ratio
            b += im.nbytes * ratio**2
        mem_required = b * self.ni / n * (1 + safety_margin)  # GB required to cache dataset into RAM
        mem = __import__("psutil").virtual_memory()
        if mem_required > mem.available:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{mem_required / gb:.1f}GB RAM required to cache images "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{mem.available / gb:.1f}/{mem.total / gb:.1f}GB available, not caching images"
            )
            return False
        return True

    def set_rectangle(self):
        """Set the shape of bounding boxes for YOLO detections as rectangles."""
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

        self.batch_shapes = np.ceil(np.array(shapes) * self.imgsz / self.stride + self.pad).astype(int) * self.stride
        self.batch = bi  # batch index of image

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
            label["resized_shape"][0] / label["ori_shape"][0],
            label["resized_shape"][1] / label["ori_shape"][1],
        )  # for evaluation
        if self.rect:
            label["rect_shape"] = self.batch_shapes[self.batch[index]]
        return self.update_labels_info(label)

    def __len__(self):
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label):
        """Custom your label format here."""
        return label

    def build_transforms(self, hyp=None):
        """
        Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self):
        """
        Users can customize their own format here.

        Examples:
            Ensure output is a dictionary with the following keys:
            >>> dict(
            ...     im_file=im_file,
            ...     shape=shape,  # format: (height, width)
            ...     cls=cls,
            ...     bboxes=bboxes,  # xywh
            ...     segments=segments,  # xy
            ...     keypoints=keypoints,  # xy
            ...     normalized=True,  # or False
            ...     bbox_format="xyxy",  # or xywh, ltwh
            ... )
        """
        raise NotImplementedError
