# Ultralytics YOLO 🚀, GPL-3.0 license

from itertools import repeat
from multiprocessing.pool import Pool, ThreadPool
from pathlib import Path

import torchvision
from tqdm import tqdm

from ..utils import NUM_THREADS, TQDM_BAR_FORMAT, is_dir_writeable
from .augment import *
from .base import BaseDataset
from .utils import HELP_URL, LOCAL_RANK, get_hash, img2label_paths, verify_image_label


class YOLODataset(BaseDataset):
    cache_version = 1.0  # dataset labels *.cache version, >= 1.0 for YOLOv8
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    """YOLO Dataset.
    Args:
        img_path (str): image path.
        prefix (str): prefix.
    """

    def __init__(
        self,
        img_path,
        imgsz=640,
        label_path=None,
        cache=False,
        augment=True,
        hyp=None,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.0,
        single_cls=False,
        use_segments=False,
        use_keypoints=False,
    ):
        self.use_segments = use_segments
        self.use_keypoints = use_keypoints
        assert not (self.use_segments and self.use_keypoints), "Can not use both segments and keypoints."
        super().__init__(img_path, imgsz, label_path, cache, augment, hyp, prefix, rect, batch_size, stride, pad,
                         single_cls)

    def cache_labels(self, path=Path("./labels.cache")):
        # Cache dataset labels, check images and read shapes
        x = {"labels": []}
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning {path.parent / path.stem}..."
        total = len(self.im_files)
        with (Pool if total > 10000 else ThreadPool)(NUM_THREADS) as pool:
            results = pool.imap(func=verify_image_label,
                                iterable=zip(self.im_files, self.label_files, repeat(self.prefix),
                                             repeat(self.use_keypoints)))
            pbar = tqdm(results, desc=desc, total=total, bar_format=TQDM_BAR_FORMAT)
            for im_file, lb, shape, segments, keypoint, nm_f, nf_f, ne_f, nc_f, msg in pbar:
                nm += nm_f
                nf += nf_f
                ne += ne_f
                nc += nc_f
                if im_file:
                    x["labels"].append(
                        dict(
                            im_file=im_file,
                            shape=shape,
                            cls=lb[:, 0:1],  # n, 1
                            bboxes=lb[:, 1:],  # n, 4
                            segments=segments,
                            keypoints=keypoint,
                            normalized=True,
                            bbox_format="xywh"))
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc} {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            pbar.close()

        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        self.im_files = [lb["im_file"] for lb in x["labels"]]
        if is_dir_writeable(path.parent):
            np.save(str(path), x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{self.prefix}New cache created: {path}")
        else:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable")  # not writeable
        return x

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = np.load(str(cache_path), allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except (FileNotFoundError, AssertionError, AttributeError):
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning {cache_path}... {nf} images, {nm + ne} backgrounds, {nc} corrupt"
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=TQDM_BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0, f"{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        nl = len(np.concatenate([label["cls"] for label in labels], 0))  # number of labels
        assert nl > 0, f"{self.prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        return labels

    # TODO: use hyp config to set all these augmentations
    def build_transforms(self, hyp=None):
        if self.augment:
            mosaic = self.augment and not self.rect
            transforms = mosaic_transforms(self, self.imgsz, hyp) if mosaic else affine_transforms(self.imgsz, hyp)
        else:
            transforms = Compose([LetterBox(new_shape=(self.imgsz, self.imgsz), scaleup=False)])
        transforms.append(
            Format(bbox_format="xywh",
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True))
        return transforms

    def close_mosaic(self, hyp):
        self.transforms = affine_transforms(self.imgsz, hyp)
        self.transforms.append(
            Format(bbox_format="xywh",
                   normalize=True,
                   return_mask=self.use_segments,
                   return_keypoint=self.use_keypoints,
                   batch_idx=True))

    def update_labels_info(self, label):
        """custom your label format here"""
        # NOTE: cls is not with bboxes now, classification and semantic segmentation need an independent cls label
        # we can make it also support classification and semantic segmentation by add or remove some dict keys there.
        bboxes = label.pop("bboxes")
        segments = label.pop("segments")
        keypoints = label.pop("keypoints", None)
        bbox_format = label.pop("bbox_format")
        normalized = label.pop("normalized")
        label["instances"] = Instances(bboxes, segments, keypoints, bbox_format=bbox_format, normalized=normalized)
        return label

    @staticmethod
    def collate_fn(batch):
        # TODO: returning a dict can make thing easier and cleaner when using dataset in training
        # but I don't know if this will slow down a little bit.
        new_batch = {}
        keys = batch[0].keys()
        values = list(zip(*[list(b.values()) for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["masks", "keypoints", "bboxes", "cls"]:
                value = torch.cat(value, 0)
            new_batch[k] = value
        new_batch["batch_idx"] = list(new_batch["batch_idx"])
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] += i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch


# Classification dataloaders -------------------------------------------------------------------------------------------
class ClassificationDataset(torchvision.datasets.ImageFolder):
    """
    YOLOv5 Classification Dataset.
    Arguments
        root:  Dataset path
        transform:  torchvision transforms, used by default
        album_transform: Albumentations transforms, used if installed
    """

    def __init__(self, root, augment, imgsz, cache=False):
        super().__init__(root=root)
        self.torch_transforms = classify_transforms(imgsz)
        self.album_transforms = classify_albumentations(augment, imgsz) if augment else None
        self.cache_ram = cache is True or cache == "ram"
        self.cache_disk = cache == "disk"
        self.samples = [list(x) + [Path(x[0]).with_suffix(".npy"), None] for x in self.samples]  # file, index, npy, im

    def __getitem__(self, i):
        f, j, fn, im = self.samples[i]  # filename, index, filename.with_suffix('.npy'), image
        if self.cache_ram and im is None:
            im = self.samples[i][3] = cv2.imread(f)
        elif self.cache_disk:
            if not fn.exists():  # load npy
                np.save(fn.as_posix(), cv2.imread(f))
            im = np.load(fn)
        else:  # read image
            im = cv2.imread(f)  # BGR
        if self.album_transforms:
            sample = self.album_transforms(image=cv2.cvtColor(im, cv2.COLOR_BGR2RGB))["image"]
        else:
            sample = self.torch_transforms(im)
        return {'img': sample, 'cls': j}

    def __len__(self) -> int:
        return len(self.samples)


# TODO: support semantic segmentation
class SemanticDataset(BaseDataset):

    def __init__(self):
        pass
