# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import glob
import inspect
import math
import os
import random
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from torch.utils.data import Dataset

from ultralytics.data.utils import FORMATS_HELP_MSG, HELP_URL, IMG_FORMATS, cache_images_to_ram, check_file_speeds
from ultralytics.data.utils import check_cache_ram as _check_cache_ram
from ultralytics.utils import DEFAULT_CFG, LOCAL_RANK, LOGGER, NUM_THREADS, TQDM
from ultralytics.utils.patches import imread


class BaseDataset(Dataset):
    """Base dataset class for loading and processing image data.

    This class provides core functionality for loading images, caching, and preparing data for training and inference in
    object detection tasks.

    Attributes:
        img_path (str | list[str]): Path to the folder containing images.
        imgsz (int): Target image size for resizing.
        augment (bool): Whether to apply data augmentation.
        single_cls (bool): Whether to treat all objects as a single class.
        prefix (str): Prefix to print in log messages.
        fraction (float): Fraction of dataset to utilize.
        channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with OpenCV
            are in BGR channel order.
        cv2_flag (int): OpenCV flag for reading images.
        im_files (list[str]): List of image file paths.
        labels (list[dict]): List of label data dictionaries.
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
        npy_files (list[Path]): List of numpy file paths.
        cache (str | None): Cache setting ('ram', 'disk', or None for no caching).
        transforms (callable): Image transformation function.
        batch_shapes (np.ndarray): Batch shapes for rectangular training.
        batch (np.ndarray): Batch index of each image.

    Methods:
        get_img_files: Read image files from the specified path.
        update_labels: Update labels to include only specified classes.
        load_image: Load an image from the dataset.
        cache_images: Cache images to memory or disk.
        cache_images_to_disk: Save an image as an *.npy file for faster loading.
        check_cache_disk: Check image caching requirements vs available disk space.
        check_cache_ram: Check image caching requirements vs available memory.
        set_rectangle: Sort images by aspect ratio and set batch shapes for rectangular training.
        get_image_and_label: Get and return label information from the dataset.
        update_labels_info: Custom label format method to be implemented by subclasses.
        build_transforms: Build transformation pipeline to be implemented by subclasses.
        get_labels: Get labels method to be implemented by subclasses.
    """

    def __init__(
        self,
        img_path: str | list[str],
        imgsz: int = 640,
        cache: bool | str = False,
        augment: bool = True,
        hyp: dict[str, Any] = DEFAULT_CFG,
        prefix: str = "",
        rect: bool = False,
        batch_size: int = 16,
        stride: int = 32,
        pad: float = 0.5,
        single_cls: bool = False,
        classes: list[int] | None = None,
        fraction: float = 1.0,
        channels: int = 3,
    ):
        """Initialize BaseDataset with given configuration and options.

        Args:
            img_path (str | list[str]): Path to the folder containing images or list of image paths.
            imgsz (int): Image size for resizing.
            cache (bool | str): Cache images to RAM or disk during training.
            augment (bool): If True, data augmentation is applied.
            hyp (dict[str, Any]): Hyperparameters to apply data augmentation.
            prefix (str): Prefix to print in log messages.
            rect (bool): If True, rectangular training is used.
            batch_size (int): Size of batches.
            stride (int): Stride used in the model.
            pad (float): Padding value.
            single_cls (bool): If True, single class training is used.
            classes (list[int], optional): List of included classes.
            fraction (float): Fraction of dataset to utilize.
            channels (int): Number of channels in the images (1 for grayscale, 3 for color). Color images loaded with
                OpenCV are in BGR channel order.
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
        # Shared RAM cache: single torch.uint8 byte buffer pinned via share_memory_() before workers fork.
        # The buffer holds raw bytes; per-image dtype is tracked so the fast path can view bytes back with the
        # correct dtype (supports uint8 JPG/PNG and uint16 TIFF alike). Cache is built with the same resize
        # params (rect_mode, resize_short) and imread() decoder load_image uses at runtime (see
        # cache_load_params), so probe and load can never disagree on shape/dtype/channels.
        self.img_cache = None
        self.img_offsets = None
        self.img_shapes = None
        self.img_dtypes = None
        self._cache_rect_mode, self._cache_resize_short = self.cache_load_params()
        # safety_margin=1.0 budgets ~2x the (resized) cache estimate: the two-pass build decodes full-resolution
        # originals concurrently (NUM_THREADS) while the cache holds small resized images, so peak working set is
        # dominated by transient decodes, not the cache itself (measured ~1.85x the resized estimate even with the
        # buffer allocated directly in shared memory, since the full-res decodes dwarf the small resized cache).
        if self.cache == "ram" and self.check_cache_ram(safety_margin=1.0):
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

    def get_img_files(self, img_path: str | list[str]) -> list[str]:
        """Read image files from the specified path.

        Args:
            img_path (str | list[str]): Path or list of paths to image directories or files.

        Returns:
            (list[str]): List of image file paths.

        Raises:
            FileNotFoundError: If no images are found or the path doesn't exist.
        """
        try:
            f = []  # image files
            for p in img_path if isinstance(img_path, list) else [img_path]:
                p = Path(p)  # os-agnostic
                if p.is_dir():  # dir
                    f += glob.glob(str(Path(glob.escape(p)) / "**" / "*.*"), recursive=True)
                    # F = list(p.rglob('*.*'))  # pathlib
                elif p.is_file():  # file
                    with open(p, encoding="utf-8") as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace("./", parent, 1) if x.startswith("./") else x for x in t]  # local to global
                        # F += [p.parent / x.lstrip(os.sep) for x in t]  # local to global (pathlib)
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

    def update_labels(self, include_class: list[int] | None) -> None:
        """Update labels to include only specified classes.

        Args:
            include_class (list[int], optional): List of classes to include. If None, all classes are included.
        """
        include_class_array = np.array(include_class).reshape(1, -1)
        for i in range(len(self.labels)):
            if include_class is not None:
                cls = self.labels[i]["cls"]
                bboxes = self.labels[i]["bboxes"]
                segments = self.labels[i]["segments"]
                keypoints = self.labels[i].get("keypoints")
                j = (cls == include_class_array).any(1)
                self.labels[i]["cls"] = cls[j]
                self.labels[i]["bboxes"] = bboxes[j]
                if segments:
                    self.labels[i]["segments"] = [segments[si] for si, idx in enumerate(j) if idx]
                if keypoints is not None:
                    self.labels[i]["keypoints"] = keypoints[j]
            if self.single_cls:
                self.labels[i]["cls"][:, 0] = 0

    def cache_load_params(self) -> tuple[bool, bool]:
        """Return the ``(rect_mode, resize_short)`` that ``load_image`` uses at runtime.

        The shared RAM cache is built with exactly these values so the cache fast path returns identically
        resized images as the uncached path. Subclasses that change the resize must override this (e.g.
        ``SemanticDataset`` resizes the short side during augmentation via ``resize_short=self.augment``).

        Returns:
            (tuple[bool, bool]): The ``rect_mode`` and ``resize_short`` flags used by ``load_image``.
        """
        param = inspect.signature(self.load_image).parameters.get("rect_mode")
        return (param.default if param is not None else True), False

    def _resized_hw(self, h0: int, w0: int, rect_mode: bool, resize_short: bool) -> tuple[int, int]:
        """Compute the post-resize ``(h, w)`` without decoding or resizing the image.

        Single source of truth shared by ``load_image`` (runtime), the cache probe pass (offset sizing) and
        the cache load pass (resize target), so they can never disagree on the cached image shape.

        Args:
            h0 (int): Original image height.
            w0 (int): Original image width.
            rect_mode (bool): Resize the long side to ``imgsz`` keeping aspect ratio.
            resize_short (bool): Resize the short side to ``imgsz`` keeping aspect ratio (overrides ``rect_mode``).

        Returns:
            (tuple[int, int]): The target ``(height, width)`` after resizing.
        """
        if rect_mode:
            if resize_short:  # resize short side to imgsz
                r = self.imgsz / min(h0, w0)
                if r == 1:
                    return h0, w0
                return (self.imgsz, math.ceil(w0 * r)) if h0 < w0 else (math.ceil(h0 * r), self.imgsz)
            r = self.imgsz / max(h0, w0)  # resize long side to imgsz
            if r == 1:
                return h0, w0
            return min(math.ceil(h0 * r), self.imgsz), min(math.ceil(w0 * r), self.imgsz)
        if h0 == w0 == self.imgsz:
            return h0, w0
        return self.imgsz, self.imgsz  # stretch to square imgsz

    def _resize_image(self, im: np.ndarray, rect_mode: bool, resize_short: bool) -> np.ndarray:
        """Resize a decoded image exactly as ``load_image`` does (single source of truth, see ``_resized_hw``)."""
        h0, w0 = im.shape[:2]
        h, w = self._resized_hw(h0, w0, rect_mode, resize_short)
        if (h, w) != (h0, w0):
            im = cv2.resize(im, (w, h), interpolation=cv2.INTER_LINEAR)
        return im if im.ndim == 3 else im[..., None]

    def load_image(
        self, i: int, rect_mode: bool = True, resize_short: bool = False
    ) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
        """Load an image from dataset index 'i'.

        Args:
            i (int): Index of the image to load.
            rect_mode (bool): Whether to use rectangular resizing (long side to imgsz).
            resize_short (bool): Whether to resize the shorter side to imgsz while maintaining aspect ratio. Overrides
                rect_mode when True.

        Returns:
            im (np.ndarray): Loaded image as a NumPy array.
            hw_original (tuple[int, int]): Original image dimensions in (height, width) format.
            hw_resized (tuple[int, int]): Resized image dimensions in (height, width) format.

        Raises:
            FileNotFoundError: If the image file is not found.
        """
        # Shared RAM cache fast path: workers map the same SHM region. The cache is built with the resize params
        # from cache_load_params(); only callers requesting those same params hit the fast path, others fall
        # through to decode (so a different rect_mode/resize_short never returns a wrongly-resized cached image).
        # .copy() prevents in-place augmentations (e.g. RandomHSV cv2.cvtColor with dst=img) from mutating the
        # shared backing tensor and corrupting the cache for other workers.
        if (
            (rect_mode, resize_short) == (self._cache_rect_mode, self._cache_resize_short)
            and self.cache == "ram"
            and self.img_cache is not None
        ):
            offset = self.img_offsets[i]
            h, w, c = self.img_shapes[i]
            dtype = self.img_dtypes[i]
            nb = h * w * c * dtype.itemsize
            im = self.img_cache[offset : offset + nb].numpy().view(dtype).reshape(h, w, c).copy()
            return im, self.im_hw0[i], (h, w)

        im, f, fn = self.ims[i], self.im_files[i], self.npy_files[i]
        if im is None:  # not cached in RAM
            if fn.exists():  # load npy
                try:
                    im = np.load(fn)
                    npy_channels = im.shape[-1] if im.ndim >= 3 else 1
                    if npy_channels != self.channels:
                        LOGGER.warning(
                            f"{self.prefix}Removing stale *.npy image file {fn} with {npy_channels} channels, expected {self.channels}"
                        )
                        Path(fn).unlink(missing_ok=True)
                        im = imread(f, flags=self.cv2_flag)
                except Exception as e:
                    LOGGER.warning(f"{self.prefix}Removing corrupt *.npy image file {fn} due to: {e}")
                    Path(fn).unlink(missing_ok=True)
                    im = imread(f, flags=self.cv2_flag)  # BGR
            else:  # read image
                im = imread(f, flags=self.cv2_flag)  # BGR
            if im is None:
                raise FileNotFoundError(f"Image Not Found {f}")

            h0, w0 = im.shape[:2]  # orig hw
            im = self._resize_image(im, rect_mode, resize_short)

            # Add to buffer if training with augmentations (skipped on the shared-cache path above)
            if self.augment:
                self.ims[i], self.im_hw0[i], self.im_hw[i] = im, (h0, w0), im.shape[:2]  # im, hw_original, hw_resized
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:  # prevent empty buffer
                    j = self.buffer.pop(0)
                    self.ims[j] = None  # release private fallback array; else it leaks atop the shared buffer
                    if self.img_cache is None:  # only the per-image list owns im_hw0/im_hw; the shared cache keeps them
                        self.im_hw0[j], self.im_hw[j] = None, None

            return im, (h0, w0), im.shape[:2]

        return self.ims[i], self.im_hw0[i], self.im_hw[i]

    def cache_images(self) -> None:
        """Cache images to memory (shared byte buffer) or disk (*.npy files) for faster training.

        ``cache='ram'`` builds a single ``torch.uint8`` buffer in two passes (probe shapes, then decode + stream)
        and pins it with ``share_memory_()`` so all DataLoader workers map the same POSIX shared-memory region
        instead of duplicating the cache per worker (the source of the ``num_workers > 0`` leak). imread plus the
        shared ``_resized_hw``/``_resize_image`` helpers are used in both passes and at runtime so probe, load and
        load_image can't disagree on shape/dtype/channels (matters for multi-frame TIFFs, uint16 TIFFs,
        EXIF-rotated JPGs, and subclasses using ``resize_short`` such as ``SemanticDataset``).
        """
        b, gb = 0, 1 << 30
        if self.cache == "disk":
            with ThreadPool(NUM_THREADS) as pool:
                pbar = TQDM(
                    enumerate(pool.imap(self.cache_images_to_disk, range(self.ni))),
                    total=self.ni,
                    disable=LOCAL_RANK > 0,
                )
                for i, _ in pbar:
                    b += self.npy_files[i].stat().st_size
                    pbar.desc = f"{self.prefix}Caching images ({b / gb:.1f}GB Disk)"
                pbar.close()
            return

        rect_mode, resize_short = self._cache_rect_mode, self._cache_resize_short

        def probe(i: int):  # (i, cache bytes, ((h0, w0), (h, w, c), dtype)) — decode matches load() for EXIF/dtype
            im = imread(self.im_files[i], flags=self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {self.im_files[i]}")
            h0, w0 = im.shape[:2]
            h, w = self._resized_hw(h0, w0, rect_mode, resize_short)
            c = im.shape[2] if im.ndim == 3 else 1
            return i, h * w * c * im.dtype.itemsize, ((h0, w0), (h, w, c), im.dtype)

        def decode(i: int):
            im = imread(self.im_files[i], flags=self.cv2_flag)
            if im is None:
                raise FileNotFoundError(f"Image Not Found {self.im_files[i]}")
            return i, self._resize_image(im, rect_mode, resize_short)

        cache, offsets, metas = cache_images_to_ram(self.ni, probe, decode, self.prefix)
        if cache is None:  # build failed (e.g. /dev/shm quota) — fall back to disk reads
            self.cache = None
            return
        self.img_cache, self.img_offsets = cache, offsets
        self.im_hw0 = [m[0] for m in metas]
        self.img_shapes = [m[1] for m in metas]
        self.img_dtypes = [m[2] for m in metas]
        self.im_hw = [m[1][:2] for m in metas]

    def cache_images_to_disk(self, i: int) -> None:
        """Save an image as an *.npy file for faster loading."""
        f = self.npy_files[i]
        if not f.exists():
            try:
                np.save(f.as_posix(), imread(self.im_files[i], flags=self.cv2_flag), allow_pickle=False)
            except Exception as e:
                f.unlink(missing_ok=True)
                LOGGER.warning(f"{self.prefix}WARNING ⚠️ Failed to cache image {f}: {e}")

    def check_cache_disk(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough disk space for caching images.

        Args:
            safety_margin (float): Safety margin factor for disk space calculation.

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
                LOGGER.warning(f"{self.prefix}Skipping caching images to disk, directory not writable")
                return False
        disk_required = b * self.ni / n * (1 + safety_margin)  # bytes required to cache dataset to disk
        total, _used, free = shutil.disk_usage(Path(self.im_files[0]).parent)
        if disk_required > free:
            self.cache = None
            LOGGER.warning(
                f"{self.prefix}{disk_required / gb:.1f}GB disk space required, "
                f"with {int(safety_margin * 100)}% safety margin but only "
                f"{free / gb:.1f}/{total / gb:.1f}GB free, not caching images to disk"
            )
            return False
        return True

    def check_cache_ram(self, safety_margin: float = 0.5) -> bool:
        """Check if there's enough RAM to cache the dataset images, disabling the cache if not.

        Thin wrapper over the shared ``data.utils.check_cache_ram`` (``scale=True`` because detection caches
        resized images). Passes ``_resized_hw`` so the estimate uses the same resize geometry the cache build
        uses (``_cache_rect_mode``/``_cache_resize_short``), not a fixed long-side assumption — otherwise
        short-side (``SemanticDataset``) or square-stretch (``rect_mode=False``) caches are under-budgeted.
        Kept as a method so the ``self.cache = None`` side effect and subclass overrides remain.

        Args:
            safety_margin (float): Safety margin factor for RAM calculation.

        Returns:
            (bool): True if there's enough RAM, False otherwise.
        """
        ok = _check_cache_ram(
            self.im_files,
            self.prefix,
            safety_margin,
            scale=True,
            sizer=lambda h0, w0: self._resized_hw(h0, w0, self._cache_rect_mode, self._cache_resize_short),
            flags=self.cv2_flag,  # sample with the dataset's decode flags so channel count matches the cache
        )
        if not ok:
            self.cache = None
        return ok

    def set_rectangle(self) -> None:
        """Sort images by aspect ratio and set batch shapes for rectangular training."""
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

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return transformed label information for given index."""
        return self.transforms(self.get_image_and_label(index))

    def get_image_and_label(self, index: int) -> dict[str, Any]:
        """Get and return label information from the dataset.

        Args:
            index (int): Index of the image to retrieve.

        Returns:
            (dict[str, Any]): Label dictionary with image and metadata.
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

    def __len__(self) -> int:
        """Return the length of the labels list for the dataset."""
        return len(self.labels)

    def update_labels_info(self, label: dict[str, Any]) -> dict[str, Any]:
        """Customize your label format here."""
        return label

    def build_transforms(self, hyp: dict[str, Any] | None = None):
        """Users can customize augmentations here.

        Examples:
            >>> if self.augment:
            ...     # Training transforms
            ...     return Compose([])
            >>> else:
            ...    # Val transforms
            ...    return Compose([])
        """
        raise NotImplementedError

    def get_labels(self) -> list[dict[str, Any]]:
        """Users can customize their own format here.

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
