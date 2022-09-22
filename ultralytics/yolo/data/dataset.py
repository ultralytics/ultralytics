from .base import BaseDataset
from .utils import img2label_paths, get_hash, verify_image_label, LOCAL_RANK, BAR_FORMAT, HELP_URL
# from .augment import *
from ..utils.general import NUM_THREADS, LOGGER
from multiprocessing.pool import Pool
from itertools import repeat
from pathlib import Path
from tqdm import tqdm
import cv2
import numpy as np
import torch


class YOLODetectionDataset(BaseDataset):
    cache_version = 0.6  # dataset labels *.cache version
    rand_interp_methods = [cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4]
    """YOLO Dataset.
    Args:
        img_path (str): image path.
        pipeline (dict): a dict of image transforms.
        prefix (str): prefix.
    """

    def __init__(
        self,
        img_path,
        img_size=640,
        label_path=None,
        cache_images=False,
        augment=True,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.0,
    ):
        self.mask = False
        self.keypoint = False
        super().__init__(img_path, img_size, label_path, cache_images, augment, prefix, rect, batch_size, stride, pad)

    def cache_labels(self, path=Path("./labels.cache")):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        x["labels"] = []
        nm, nf, ne, nc, msgs = 0, 0, 0, 0, []  # number missing, found, empty, corrupt, messages
        desc = f"{self.prefix}Scanning '{path.parent / path.stem}' images and labels..."
        with Pool(NUM_THREADS) as pool:
            pbar = tqdm(
                pool.imap(verify_image_label, zip(self.im_files, self.label_files, repeat(self.prefix), repeat(self.keypoint))),
                desc=desc,
                total=len(self.im_files),
                bar_format=BAR_FORMAT,
            )
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
                            keypoint=keypoint,
                            normalized=True,
                            bbox_format="xywh"
                        )
                    )
                if msg:
                    msgs.append(msg)
                pbar.desc = f"{desc}{nf} found, {nm} missing, {ne} empty, {nc} corrupt"

        pbar.close()
        if msgs:
            LOGGER.info("\n".join(msgs))
        if nf == 0:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ No labels found in {path}. {HELP_URL}")
        x["hash"] = get_hash(self.label_files + self.im_files)
        x["results"] = nf, nm, ne, nc, len(self.im_files)
        x["msgs"] = msgs  # warnings
        x["version"] = self.cache_version  # cache version
        try:
            np.save(path, x)  # save cache for next time
            path.with_suffix(".cache.npy").rename(path)  # remove .npy suffix
            LOGGER.info(f"{self.prefix}New cache created: {path}")
        except Exception as e:
            LOGGER.warning(f"{self.prefix}WARNING ⚠️ Cache directory {path.parent} is not writeable: {e}")  # not writeable
        return x

    def get_labels(self):
        self.label_files = img2label_paths(self.im_files)
        cache_path = Path(self.label_files[0]).parent.with_suffix(".cache")
        try:
            cache, exists = np.load(cache_path, allow_pickle=True).item(), True  # load dict
            assert cache["version"] == self.cache_version  # matches current version
            assert cache["hash"] == get_hash(self.label_files + self.im_files)  # identical hash
        except Exception:
            cache, exists = self.cache_labels(cache_path), False  # run cache ops

        # Display cache
        nf, nm, ne, nc, n = cache.pop("results")  # found, missing, empty, corrupt, total
        if exists and LOCAL_RANK in {-1, 0}:
            d = f"Scanning '{cache_path}' images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupt"
            tqdm(None, desc=self.prefix + d, total=n, initial=n, bar_format=BAR_FORMAT)  # display cache results
            if cache["msgs"]:
                LOGGER.info("\n".join(cache["msgs"]))  # display warnings
        assert nf > 0, f"{self.prefix}No labels found in {cache_path}, can not start training. {HELP_URL}"

        # Read cache
        [cache.pop(k) for k in ("hash", "version", "msgs")]  # remove items
        labels = cache["labels"]
        nl = len(np.concatenate([label["cls"] for label in labels], 0))  # number of labels
        assert nl > 0, f"{self.prefix}All labels empty in {cache_path}, can not start training. {HELP_URL}"
        return labels

    @staticmethod
    def collate_fn(batch):
        # TODO: returning a dict can make thing easier and cleaner when using dataset in training
        # but I don't know if this will slow down a little bit.
        new_batch = dict()
        keys = batch[0].keys()
        values = list(zip([b.values() for b in batch]))
        for i, k in enumerate(keys):
            value = values[i]
            if k == "img":
                value = torch.stack(value, 0)
            if k in ["mask", "keypoint", "bboxes", "cls"]:
                value = torch.cat(value, 0)
            new_batch[k] = values[i]
        for i in range(len(new_batch["batch_idx"])):
            new_batch["batch_idx"][i] = i  # add target image index for build_targets()
        new_batch["batch_idx"] = torch.cat(new_batch["batch_idx"], 0)
        return new_batch

    # def build_transforms(self):
    #     # TODO: use hyp config to set these augmentations
    #     mosaic = self.augment and not self.rect
    #     if self.augment:
    #         if mosaic:
    #             transforms = Compose(
    #                 [
    #                     Mosaic(img_size=self.img_size, p=1.0),
    #                     CopyPaste(p=0.0),
    #                     RandomPerspective(border=[-self.img_size // 2, -self.img_size // 2]),
    #                     MixUp(p=0.0),
    #                     Albumentations(p=1.0),
    #                     RandomHSV(),
    #                     RandomFlip(direction="vertical", p=0.5),
    #                     RandomFlip(direction="horizontal", p=0.5),
    #                 ]
    #             )
    #         else:
    #             # rect, randomperspective, albumentation, hsv, flipud, fliplr
    #             transforms = Compose(
    #                 [
    #                     LetterBox(new_shape=(self.img_size, self.img_size)),
    #                     RandomPerspective(border=[-self.img_size // 2, -self.img_size // 2]),
    #                     Albumentations(p=1.0),
    #                     RandomHSV(),
    #                     RandomFlip(direction="vertical", p=0.5),
    #                     RandomFlip(direction="horizontal", p=0.5),
    #                 ]
    #             )
    #     else:
    #         transforms = Compose([LetterBox(new_shape=(self.img_size, self.img_size))])
    #     transforms.append(Format(bbox_format="xywh", normalize=True, mask=self.mask, batch_idx=True))
    #     return transforms


class YOLOSegmentDataset(YOLODetectionDataset):
    def __init__(
        self,
        img_path,
        img_size=640,
        label_path=None,
        cache_images=False,
        augment=True,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.5,
    ):
        self.keypoint = False
        super().__init__(img_path, img_size, label_path, cache_images, augment, prefix, rect, batch_size, stride, pad)
        # TODO
        self.mask = True


class YOLOPoseDataset(YOLODetectionDataset):
    def __init__(
        self,
        img_path,
        img_size=640,
        label_path=None,
        cache_images=False,
        augment=True,
        prefix="",
        rect=False,
        batch_size=None,
        stride=32,
        pad=0.5,
    ):
        self.keypoint = True
        super().__init__(img_path, img_size, label_path, cache_images, augment, prefix, rect, batch_size, stride, pad)
