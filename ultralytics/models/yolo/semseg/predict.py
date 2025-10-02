# Ultralytics YOLO 🚀, AGPL-3.0 license
from __future__ import annotations

import os
import re
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics.engine.results import Results
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils import DEFAULT_CFG, SEMSEG_CFG, YAML


class SemSegPredictor(DetectionPredictor):
    """
    A class extending the DetectionPredictor class for prediction based on a segmentation model.

    This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
    prediction results.

    Attributes:
        args (dict): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded YOLO segmentation model.
        batch (list): Current batch of images being processed.

    Methods:
        postprocess: Apply non-max suppression and process segmentation detections.
        construct_results: Construct a list of result objects from predictions.
        construct_result: Construct a single result object from a prediction.

    Examples:
        >>> from ultralytics.utils import ASSETS
        >>> from ultralytics.models.yolo.segment import SegmentationPredictor
        >>> args = dict(model="yolo11n-seg.pt", source=ASSETS)
        >>> predictor = SegmentationPredictor(overrides=args)
        >>> predictor.predict_cli()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the SegmentationPredictor with configuration, overrides, and callbacks.

        This class specializes in processing segmentation model outputs, handling both bounding boxes and masks in the
        prediction results.

        Args:
            cfg (dict): Configuration for the predictor.
            overrides (dict, optional): Configuration overrides that take precedence over cfg.
            _callbacks (list, optional): List of callback functions to be invoked during prediction.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.args.task = "semseg"

    def postprocess(self, preds, img, orig_imgs):
        """
        Apply non-max suppression and process segmentation detections for each image in the input batch.

        Args:
            preds (tuple): Model predictions, containing bounding boxes, scores, classes, and mask coefficients.
            img (torch.Tensor): Input image tensor in model format, with shape (B, C, H, W).
            orig_imgs (list | torch.Tensor | np.ndarray): Original image or batch of images.

        Returns:
            (list): List of Results objects containing the segmentation predictions for each image in the batch.
                Each Results object includes both bounding boxes and segmentation masks.

        Examples:
            >>> predictor = SegmentationPredictor(overrides=dict(model="yolo11n-seg.pt"))
            >>> results = predictor.postprocess(preds, img, orig_img)
        """
        # Extract protos - tuple if PyTorch model or array if exported
        result_list = []
        for pred, orig_img, img_path in zip(preds, orig_imgs, self.batch[0]):
            result_list.append(
                Results(
                    orig_img, path=img_path, names=self.model.names, masks=torch.softmax(pred.detach().cpu(), dim=0)
                )
            )
        return result_list

    def write_results(self, i: int, p: Path, im: torch.Tensor, s: list[str]) -> str:
        """
        Write inference results to a file or directory.

        Args:
            i (int): Index of the current image in the batch.
            p (Path): Path to the current image.
            im (torch.Tensor): Preprocessed image tensor.
            s (List[str]): List of result strings.

        Returns:
            (str): String with result information.
        """
        string = ""  # print string
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        if self.source_type.stream or self.source_type.from_img or self.source_type.tensor:  # batch_size >= 1
            string += f"{i}: "
            frame = self.dataset.count
        else:
            match = re.search(r"frame (\d+)/", s[i])
            frame = int(match[1]) if match else None  # 0 if frame undetermined

        self.txt_path = self.save_dir / "labels" / (p.stem + ("" if self.dataset.mode == "image" else f"_{frame}"))
        string += "{:g}x{:g} ".format(*im.shape[2:])
        result = self.results[i]
        result.save_dir = self.save_dir.__str__()  # used in other locations
        string += f"{result.verbose()}{result.speed['inference']:.1f}ms"

        # Add predictions to image
        imagename = result.path.split(os.sep)[-1]
        if self.args.save or self.args.show:
            image_dir, mask_dir = os.path.join(self.save_dir, "image"), os.path.join(self.save_dir, "mask")
            os.mkdir(image_dir) if not os.path.exists(image_dir) else None
            os.mkdir(mask_dir) if not os.path.exists(mask_dir) else None
            self.plot_predict_samples(
                result.orig_img,
                result.masks,
                nc=YAML.load(self.data)["nc"],
                colors=YAML.load(self.data)["colors"],
                fname=self.save_dir / "image" / imagename,
                mname=self.save_dir / "mask" / imagename,
                one_hot=True,
                overlap=True,
            )

        # Save results
        if self.args.save_txt:
            result.save_txt(f"{self.txt_path}.txt", save_conf=self.args.save_conf)
        if self.args.save_crop:
            result.save_crop(save_dir=self.save_dir / "crops", file_name=self.txt_path.stem)
        if self.args.show:
            self.show(str(p))
        if self.args.save:
            self.save_predicted_images(self.save_dir / p.name, frame)

        return string

    def plot_predict_samples(self, image, masks, nc, colors, fname, mname, one_hot=True, overlap=False):
        """
        Plot and visualize the predicting results
        Args:
            image(torch.Tensor| numpy.ndarray): input image
            masks: (torch.Tensor| numpy.ndarray): predict mask
            nc(int): number of categories
            colors(List): colors for each categories
            fname(str): saved image path
            mname(str): save mask path
            one_hot(bool): is the format of mask one-hot
            overlap(bool): plot mask on image.

        Returns:
            None
        """
        if isinstance(image, torch.Tensor):
            image = image.cpu().float().numpy()

        if isinstance(masks.data, torch.Tensor):
            masks = masks.data.cpu().numpy()

        if np.max(image) <= 1:
            image *= 255  # de-normalise (optional)

        if np.max(masks.data) <= 1:
            masks *= 255

        h, w, _ = image.shape  # batch size, _, height, width
        _, hm, wm = masks.shape
        mask_bgr = np.ones((hm, wm, 3), dtype=np.uint8) * 255
        if one_hot:
            mask = masks.copy().transpose(1, 2, 0)
            for j in range(nc):
                r, g, b = colors[j]
                mask_bgr[mask[:, :, j] > 114, :] = np.array([b, g, r]).astype(np.uint8)
        else:
            for j in range(nc):
                r, g, b = colors[j]
                mask_bgr[masks == j, :] = np.array([b, g, r]).astype(np.uint8)

        msk = cv2.resize(mask_bgr, dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        if overlap:
            img = cv2.addWeighted(image, 0.7, msk, 0.3, 0)
            cv2.imwrite(fname, img)
        else:
            cv2.imwrite(fname, image)
            cv2.imwrite(mname, msk)


def predict(cfg=DEFAULT_CFG):
    """Train a YOLO segmentation model based on passed arguments."""
    model = cfg.model or "yolov11n-seg.pt"
    data = cfg.data or "coco128-seg.yaml"  # or yolo.ClassificationDataset("mnist")
    device = cfg.device if cfg.device is not None else ""
    cfg.name = os.path.join(cfg.name, "predict")
    args = dict(model=model, data=data, device=device, name=cfg.name, task="semseg", plots=True)

    predictor = SemSegPredictor(cfg, args)
    predictor(model=model)


if __name__ == "__main__":
    predict(cfg=SEMSEG_CFG)
