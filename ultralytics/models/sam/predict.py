# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate predictions using the Segment Anything Model (SAM).

SAM is an advanced image segmentation model offering features like promptable segmentation and zero-shot performance.
This module contains the implementation of the prediction logic and auxiliary utilities required to perform segmentation
using SAM. It forms an integral part of the Ultralytics framework and is designed for high-performance, real-time image
segmentation tasks.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from copy import deepcopy
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, LOGGER, ops
from ultralytics.utils.metrics import box_iou, mask_iou
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

from .amg import (
    batch_iterator,
    batched_mask_to_box,
    build_all_layer_point_grids,
    calculate_stability_score,
    generate_crop_boxes,
    is_box_near_crop_edge,
    remove_small_regions,
    uncrop_boxes_xyxy,
    uncrop_masks,
)
from .sam3.geometry_encoders import Prompt


class Predictor(BasePredictor):
    """Predictor class for SAM, enabling real-time image segmentation with promptable capabilities.

    This class extends BasePredictor and implements the Segment Anything Model (SAM) for advanced image segmentation
    tasks. It supports various input prompts like points, bounding boxes, and masks for fine-grained control over
    segmentation results.

    Attributes:
        args (SimpleNamespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded SAM model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        im (torch.Tensor): The preprocessed input image.
        features (torch.Tensor): Extracted image features.
        prompts (dict[str, Any]): Dictionary to store various types of prompts (e.g., bboxes, points, masks).
        segment_all (bool): Flag to indicate if full image segmentation should be performed.
        mean (torch.Tensor): Mean values for image normalization.
        std (torch.Tensor): Standard deviation values for image normalization.

    Methods:
        preprocess: Prepare input images for model inference.
        pre_transform: Perform initial transformations on the input image.
        inference: Perform segmentation inference based on input prompts.
        prompt_inference: Internal function for prompt-based segmentation inference.
        generate: Generate segmentation masks for an entire image.
        setup_model: Initialize the SAM model for inference.
        get_model: Build and return a SAM model.
        postprocess: Post-process model outputs to generate final results.
        setup_source: Set up the data source for inference.
        set_image: Set and preprocess a single image for inference.
        get_im_features: Extract image features using the SAM image encoder.
        set_prompts: Set prompts for subsequent inference.
        reset_image: Reset the current image and its features.
        remove_small_regions: Remove small disconnected regions and holes from masks.

    Examples:
        >>> predictor = Predictor()
        >>> predictor.setup_model(model_path="sam_model.pt")
        >>> predictor.set_image("image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> results = predictor(bboxes=bboxes)
    """

    stride = 16

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the Predictor with configuration, overrides, and callbacks.

        Sets up the Predictor object for SAM (Segment Anything Model) and applies any configuration overrides or
        callbacks provided. Initializes task-specific settings for SAM, such as retina_masks being set to True for
        optimal results.

        Args:
            cfg (dict): Configuration dictionary containing default settings.
            overrides (dict | None): Dictionary of values to override default configuration.
            _callbacks (dict | None): Dictionary of callback functions to customize behavior.
        """
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict", batch=1))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess(self, im):
        """Preprocess the input image for model inference.

        This method prepares the input image by applying transformations and normalization. It supports both
        torch.Tensor and list of np.ndarray as input formats. For OpenCV-loaded images, the input is typically BGR and
        is converted to RGB during preprocessing.

        Args:
            im (torch.Tensor | list[np.ndarray]): Input image(s) in BCHW tensor format or a list of HWC NumPy arrays.
                NumPy arrays are expected to be in BGR order (as returned by OpenCV) and will be converted to RGB.

        Returns:
            (torch.Tensor): The preprocessed image tensor, normalized and converted to the appropriate dtype.

        Examples:
            >>> predictor = Predictor()
            >>> image = torch.rand(1, 3, 640, 640)
            >>> preprocessed_image = predictor.preprocess(image)
        """
        if self.im is not None:
            return self.im
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))
            im = np.ascontiguousarray(im)
            im = torch.from_numpy(im)

        im = im.to(self.device)
        if not_tensor:
            im = (im - self.mean) / self.std
        im = im.half() if self.model.fp16 else im.float()
        return im

    def pre_transform(self, im):
        """Perform initial transformations on the input image for preprocessing.

        This method applies transformations such as resizing to prepare the image for further preprocessing. Currently,
        batched inference is not supported; hence the list length should be 1.

        Args:
            im (list[np.ndarray]): List containing a single image in HWC numpy array format.

        Returns:
            (list[np.ndarray]): List containing the transformed image.

        Raises:
            AssertionError: If the input list contains more than one image.

        Examples:
            >>> predictor = Predictor()
            >>> image = np.random.rand(480, 640, 3)  # Single HWC image
            >>> transformed = predictor.pre_transform([image])
            >>> print(len(transformed))
            1
        """
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.imgsz, auto=False, center=False)
        return [letterbox(image=x) for x in im]

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """Perform image segmentation inference based on the given input cues, using the currently loaded image.

        This method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt encoder,
        and mask decoder for real-time and promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | list | None): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | list | None): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | list | None): Labels for point prompts, shape (N,). 1 = foreground, 0 = background.
            masks (np.ndarray | None): Low-resolution masks from previous predictions, shape (N, H, W). For SAM H=W=256.
            multimask_output (bool): Flag to return multiple masks. Helpful for ambiguous prompts.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            pred_masks (torch.Tensor): The output masks in shape (C, H, W), where C is the number of generated masks.
            pred_scores (torch.Tensor): An array of length C containing quality scores predicted by the model for each
                mask.

        Examples:
            >>> predictor = Predictor()
            >>> predictor.setup_model(model_path="sam_model.pt")
            >>> predictor.set_image("image.jpg")
            >>> results = predictor(bboxes=[[0, 0, 100, 100]])
        """
        # Override prompts if any stored in self.prompts
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)
        labels = self.prompts.pop("labels", labels)

        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(im, *args, **kwargs)

        return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """Perform image segmentation inference based on input cues using SAM's specialized architecture.

        This internal function leverages the Segment Anything Model (SAM) for prompt-based, real-time segmentation. It
        processes various input prompts such as bounding boxes, points, and masks to generate segmentation masks.

        Args:
            im (torch.Tensor): Preprocessed input image tensor with shape (N, C, H, W).
            bboxes (np.ndarray | list | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | list | None): Points indicating object locations with shape (N, 2) or (N, num_points,
                2), in pixels.
            labels (np.ndarray | list | None): Point prompt labels with shape (N) or (N, num_points). 1 for foreground,
                0 for background.
            masks (np.ndarray | None): Low-res masks from previous predictions with shape (N, H, W). For SAM, H=W=256.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.

        Returns:
            pred_masks (torch.Tensor): Output masks with shape (C, H, W), where C is the number of generated masks.
            pred_scores (torch.Tensor): Quality scores predicted by the model for each mask, with length C.

        Examples:
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> masks, scores, logits = predictor.prompt_inference(im, bboxes=bboxes)
        """
        features = self.get_im_features(im) if self.features is None else self.features

        prompts = self._prepare_prompts(im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks)
        return self._inference_features(features, *prompts, multimask_output)

    def _inference_features(
        self,
        features,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        """Perform inference on image features using the SAM model.

        Args:
            features (torch.Tensor): Extracted image features with shape (B, C, H, W) from the SAM model image encoder.
            bboxes (np.ndarray | list[list[float]] | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | list[list[float]] | None): Object location points with shape (N, 2), in pixels.
            labels (np.ndarray | list[int] | None): Point prompt labels with shape (N,). 1 = foreground, 0 = background.
            masks (list[np.ndarray] | np.ndarray | None): Masks for the objects, where each mask is a 2D array.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.

        Returns:
            pred_masks (torch.Tensor): Output masks with shape (C, H, W), where C is the number of generated masks.
            pred_scores (torch.Tensor): Quality scores for each mask, with length C.
        """
        points = (points, labels) if points is not None else None
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(points=points, boxes=bboxes, masks=masks)

        # Predict masks
        pred_masks, pred_scores = self.model.mask_decoder(
            image_embeddings=features,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
        )

        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` could be 1 or 3 depends on `multimask_output`.
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        """Prepare and transform the input prompts for processing based on the destination shape.

        Args:
            dst_shape (tuple[int, int]): The target shape (height, width) for the prompts.
            src_shape (tuple[int, int]): The source shape (height, width) of the input image.
            bboxes (np.ndarray | list | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | list | None): Points indicating object locations with shape (N, 2) or (N, num_points,
                2), in pixels.
            labels (np.ndarray | list | None): Point prompt labels with shape (N) or (N, num_points). 1 for foreground,
                0 for background.
            masks (list[np.ndarray] | np.ndarray | None): Masks for the objects, where each mask is a 2D array with
                shape (H, W).

        Returns:
            bboxes (torch.Tensor | None): Transformed bounding boxes.
            points (torch.Tensor | None): Transformed points.
            labels (torch.Tensor | None): Transformed labels.
            masks (torch.Tensor | None): Transformed masks.

        Raises:
            AssertionError: If the number of points don't match the number of labels, in case labels were passed.
        """
        r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        # Transform input prompts
        if points is not None:
            points = torch.as_tensor(points, dtype=self.torch_dtype, device=self.device)
            points = points[None] if points.ndim == 1 else points
            # Assuming labels are all positive if users don't pass labels.
            if labels is None:
                labels = np.ones(points.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert points.shape[-2] == labels.shape[-1], (
                f"Number of points {points.shape[-2]} should match number of labels {labels.shape[-1]}."
            )
            points *= r
            if points.ndim == 2:
                # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
                points, labels = points[:, None, :], labels[:, None]
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r
        if masks is not None:
            masks = np.asarray(masks, dtype=np.uint8)
            masks = masks[None] if masks.ndim == 2 else masks
            letterbox = LetterBox(dst_shape, auto=False, center=False, padding_value=0, interpolation=cv2.INTER_NEAREST)
            masks = np.stack([letterbox(image=x).squeeze() for x in masks], axis=0)
            masks = torch.tensor(masks, dtype=self.torch_dtype, device=self.device)
        return bboxes, points, labels, masks

    def generate(
        self,
        im,
        crop_n_layers=0,
        crop_overlap_ratio=512 / 1500,
        crop_downscale_factor=1,
        point_grids=None,
        points_stride=32,
        points_batch_size=64,
        conf_thres=0.88,
        stability_score_thresh=0.95,
        stability_score_offset=0.95,
        crop_nms_thresh=0.7,
    ):
        """Perform image segmentation using the Segment Anything Model (SAM).

        This method segments an entire image into constituent parts by leveraging SAM's advanced architecture and
        real-time performance capabilities. It can optionally work on image crops for finer segmentation.

        Args:
            im (torch.Tensor): Input tensor representing the preprocessed image with shape (N, C, H, W).
            crop_n_layers (int): Number of layers for additional mask predictions on image crops.
            crop_overlap_ratio (float): Overlap between crops, scaled down in subsequent layers.
            crop_downscale_factor (int): Scaling factor for sampled points-per-side in each layer.
            point_grids (list[np.ndarray] | None): Custom grids for point sampling normalized to [0,1].
            points_stride (int): Number of points to sample along each side of the image.
            points_batch_size (int): Batch size for the number of points processed simultaneously.
            conf_thres (float): Confidence threshold [0,1] for filtering based on mask quality prediction.
            stability_score_thresh (float): Stability threshold [0,1] for mask filtering based on stability.
            stability_score_offset (float): Offset value for calculating stability score.
            crop_nms_thresh (float): IoU cutoff for NMS to remove duplicate masks between crops.

        Returns:
            pred_masks (torch.Tensor): Segmented masks with shape (N, H, W).
            pred_scores (torch.Tensor): Confidence scores for each mask with shape (N,).
            pred_bboxes (torch.Tensor): Bounding boxes for each mask with shape (N, 4).

        Examples:
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)  # Example input image
            >>> masks, scores, boxes = predictor.generate(im)
        """
        import torchvision  # scope for faster 'import ultralytics'

        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_regions, layer_idxs = generate_crop_boxes((ih, iw), crop_n_layers, crop_overlap_ratio)
        if point_grids is None:
            point_grids = build_all_layer_point_grids(points_stride, crop_n_layers, crop_downscale_factor)
        pred_masks, pred_scores, pred_bboxes, region_areas = [], [], [], []
        for crop_region, layer_idx in zip(crop_regions, layer_idxs):
            x1, y1, x2, y2 = crop_region
            w, h = x2 - x1, y2 - y1
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])  # w, h
            # Crop image and interpolate to input size
            crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode="bilinear", align_corners=False)
            # (num_points, 2)
            points_for_image = point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = [], [], []
            for (points,) in batch_iterator(points_batch_size, points_for_image):
                pred_mask, pred_score = self.prompt_inference(crop_im, points=points, multimask_output=True)
                # Interpolate predicted masks to input size
                pred_mask = F.interpolate(pred_mask[None], (h, w), mode="bilinear", align_corners=False)[0]
                idx = pred_score > conf_thres
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]

                stability_score = calculate_stability_score(
                    pred_mask, self.model.mask_threshold, stability_score_offset
                )
                idx = stability_score > stability_score_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]
                # Bool type is much more memory-efficient.
                pred_mask = pred_mask > self.model.mask_threshold
                # (N, 4)
                pred_bbox = batched_mask_to_box(pred_mask).float()
                keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_region, [0, 0, iw, ih])
                if not torch.all(keep_mask):
                    pred_bbox, pred_mask, pred_score = pred_bbox[keep_mask], pred_mask[keep_mask], pred_score[keep_mask]

                crop_masks.append(pred_mask)
                crop_bboxes.append(pred_bbox)
                crop_scores.append(pred_score)

            # Do nms within this crop
            crop_masks = torch.cat(crop_masks)
            crop_bboxes = torch.cat(crop_bboxes)
            crop_scores = torch.cat(crop_scores)
            keep = torchvision.ops.nms(crop_bboxes, crop_scores, self.args.iou)  # NMS
            crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_region)
            crop_masks = uncrop_masks(crop_masks[keep], crop_region, ih, iw)
            crop_scores = crop_scores[keep]

            pred_masks.append(crop_masks)
            pred_bboxes.append(crop_bboxes)
            pred_scores.append(crop_scores)
            region_areas.append(area.expand(crop_masks.shape[0]))

        pred_masks = torch.cat(pred_masks)
        pred_bboxes = torch.cat(pred_bboxes)
        pred_scores = torch.cat(pred_scores)
        region_areas = torch.cat(region_areas)

        # Remove duplicate masks between crops
        if len(crop_regions) > 1:
            scores = 1 / region_areas
            keep = torchvision.ops.nms(pred_bboxes, scores, crop_nms_thresh)
            pred_masks, pred_bboxes, pred_scores = pred_masks[keep], pred_bboxes[keep], pred_scores[keep]

        return pred_masks, pred_scores, pred_bboxes

    def setup_model(self, model=None, verbose=True):
        """Initialize the Segment Anything Model (SAM) for inference.

        This method sets up the SAM model by allocating it to the appropriate device and initializing the necessary
        parameters for image normalization and other Ultralytics compatibility settings.

        Args:
            model (torch.nn.Module | None): A pretrained SAM model. If None, a new model is built based on config.
            verbose (bool): If True, prints selected device information.

        Examples:
            >>> predictor = Predictor()
            >>> predictor.setup_model(model=sam_model, verbose=True)
        """
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = self.get_model()
        model.eval()
        model = model.to(device)
        self.model = model.half() if self.args.half else model.float()
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

        # Ultralytics compatibility settings
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = self.args.half
        self.done_warmup = True
        self.torch_dtype = torch.float16 if self.model.fp16 else torch.float32

    def get_model(self):
        """Retrieve or build the Segment Anything Model (SAM) for image segmentation tasks."""
        from .build import build_sam  # slow import

        return build_sam(self.args.model)

    def postprocess(self, preds, img, orig_imgs):
        """Post-process SAM's inference outputs to generate object detection masks and bounding boxes.

        This method scales masks and boxes to the original image size and applies a threshold to the mask
        predictions. It leverages SAM's advanced architecture for real-time, promptable segmentation tasks.

        Args:
            preds (tuple): The output from SAM model inference, containing:
                - pred_masks (torch.Tensor): Predicted masks with shape (N, 1, H, W).
                - pred_scores (torch.Tensor): Confidence scores for each mask with shape (N, 1).
                - pred_bboxes (torch.Tensor, optional): Predicted bounding boxes if segment_all is True.
            img (torch.Tensor): The processed input image tensor with shape (C, H, W).
            orig_imgs (list[np.ndarray] | torch.Tensor): The original, unprocessed images.

        Returns:
            (list[Results]): List of Results objects containing detection masks, bounding boxes, and other metadata for
                each processed image.

        Examples:
            >>> predictor = Predictor()
            >>> preds = predictor.inference(img)
            >>> results = predictor.postprocess(preds, img, orig_imgs)
        """
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(pred_masks.shape[0])))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]

        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
                masks = masks > self.model.mask_threshold  # to bool
                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                # NOTE: SAM models do not return cls info. This `cls` here is just a placeholder for consistency.
                cls = torch.arange(pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device)
                idx = pred_scores > self.args.conf
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)[idx]
                masks = masks[idx]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # Reset segment-all mode.
        self.segment_all = False
        return results

    def set_image(self, image):
        """Preprocess and set a single image for inference.

        This method prepares the model for inference on a single image by setting up the model if not already
        initialized, configuring the data source, and preprocessing the image for feature extraction. It ensures that
        only one image is set at a time and extracts image features for subsequent use.

        Args:
            image (str | np.ndarray): Path to the image file as a string, or a numpy array representing an image read by
                cv2 (BGR channel order).

        Raises:
            AssertionError: If more than one image is attempted to be set.

        Examples:
            >>> predictor = Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(cv2.imread("path/to/image.jpg"))

        Notes:
            - This method should be called before performing inference on a new image.
            - The extracted features are stored in the `self.features` attribute for later use.
        """
        if self.model is None:
            self.setup_model()
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def setup_source(self, source):
        """Set up the data source for SAM inference."""
        if source is None:  # handle the situation when set_imgsz in advance
            return
        super().setup_source(source, self.stride)
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM models only support square image size, but got {self.imgsz}."
        )
        self.model.set_imgsz(self.imgsz)

    def get_im_features(self, im):
        """Extract image features using the SAM model's image encoder for subsequent mask prediction."""
        return self.model.image_encoder(im)

    def set_prompts(self, prompts):
        """Set prompts for subsequent inference operations."""
        self.prompts = prompts

    def reset_image(self):
        """Reset the current image and its features, clearing them for subsequent inference."""
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """Remove small disconnected regions and holes from segmentation masks.

        This function performs post-processing on segmentation masks generated by the Segment Anything Model (SAM). It
        removes small disconnected regions and holes from the input masks, and then performs Non-Maximum Suppression
        (NMS) to eliminate any newly created duplicate boxes.

        Args:
            masks (torch.Tensor): Segmentation masks to be processed, with shape (N, H, W) where N is the number of
                masks, H is height, and W is width.
            min_area (int): Minimum area threshold for removing disconnected regions and holes. Regions smaller than
                this will be removed.
            nms_thresh (float): IoU threshold for the NMS algorithm to remove duplicate boxes.

        Returns:
            new_masks (torch.Tensor): Processed masks with small regions removed, shape (N, H, W).
            keep (list[int]): Indices of remaining masks after NMS, for filtering corresponding boxes.

        Examples:
            >>> masks = torch.rand(5, 640, 640) > 0.5  # 5 random binary masks
            >>> new_masks, keep = remove_small_regions(masks, min_area=100, nms_thresh=0.7)
            >>> print(f"Original masks: {masks.shape}, Processed masks: {new_masks.shape}")
            >>> print(f"Indices of kept masks: {keep}")
        """
        import torchvision  # scope for faster 'import ultralytics'

        if masks.shape[0] == 0:
            return masks

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for mask in masks:
            mask = mask.cpu().numpy().astype(np.uint8)
            mask, changed = remove_small_regions(mask, min_area, mode="holes")
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode="islands")
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and 1 to unchanged masks so NMS prefers masks not needing postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        new_masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(new_masks)
        keep = torchvision.ops.nms(boxes.float(), torch.as_tensor(scores), nms_thresh)

        return new_masks[keep].to(device=masks.device, dtype=masks.dtype), keep

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        dst_shape=None,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
    ):
        """Perform prompts preprocessing and inference on provided image features using the SAM model.

        Args:
            features (torch.Tensor | dict[str, Any]): Extracted image features from the SAM/SAM2 model image encoder.
            src_shape (tuple[int, int]): The source shape (height, width) of the input image.
            dst_shape (tuple[int, int] | None): The target shape (height, width) for the prompts. If None, defaults to
                (imgsz, imgsz).
            bboxes (np.ndarray | list[list[float]] | None): Bounding boxes in xyxy format with shape (N, 4).
            points (np.ndarray | list[list[float]] | None): Points indicating object locations with shape (N, 2), in
                pixels.
            labels (np.ndarray | list[int] | None): Point prompt labels with shape (N, ).
            masks (list[np.ndarray] | np.ndarray | None): Masks for the objects, where each mask is a 2D array.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.

        Returns:
            pred_masks (torch.Tensor): The output masks in shape (C, H, W), where C is the number of generated masks.
            pred_bboxes (torch.Tensor): Bounding boxes for each mask with shape (N, 6), where N is the number of boxes.
                Each box is in xyxy format with additional columns for score and class.

        Notes:
            - The input features is a torch.Tensor of shape (B, C, H, W) if performing on SAM, or a dict[str, Any] if performing on SAM2.
        """
        dst_shape = dst_shape or (self.args.imgsz, self.args.imgsz)
        prompts = self._prepare_prompts(dst_shape, src_shape, bboxes, points, labels, masks)
        pred_masks, pred_scores = self._inference_features(features, *prompts, multimask_output)
        if pred_masks.shape[0] == 0:
            pred_masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            pred_masks = ops.scale_masks(pred_masks[None].float(), src_shape, padding=False)[0]
            pred_masks = pred_masks > self.model.mask_threshold  # to bool
            pred_bboxes = batched_mask_to_box(pred_masks)
            # NOTE: SAM models do not return cls info. This `cls` here is just a placeholder for consistency.
            cls = torch.arange(pred_masks.shape[0], dtype=torch.int32, device=pred_masks.device)
            pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
        return pred_masks, pred_bboxes


class SAM2Predictor(Predictor):
    """SAM2Predictor class for advanced image segmentation using Segment Anything Model 2 architecture.

    This class extends the base Predictor class to implement SAM2-specific functionality for image segmentation tasks.
    It provides methods for model initialization, feature extraction, and prompt-based inference.

    Attributes:
        _bb_feat_sizes (list[tuple]): Feature sizes for different backbone levels.
        model (torch.nn.Module): The loaded SAM2 model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        features (dict): Cached image features for efficient inference.
        segment_all (bool): Flag to indicate if all segments should be predicted.
        prompts (dict[str, Any]): Dictionary to store various types of prompts for inference.

    Methods:
        get_model: Retrieve and initialize the SAM2 model.
        prompt_inference: Perform image segmentation inference based on various prompts.
        set_image: Preprocess and set a single image for inference.
        get_im_features: Extract and process image features using SAM2's image encoder.

    Examples:
        >>> predictor = SAM2Predictor(cfg)
        >>> predictor.set_image("path/to/image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> result = predictor(bboxes=bboxes)[0]
        >>> print(f"Predicted {len(result.masks)} masks with average score {result.boxes.conf.mean():.2f}")
    """

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]
    stride = 16

    def get_model(self):
        """Retrieve and initialize the Segment Anything Model 2 (SAM2) for image segmentation tasks."""
        from .build import build_sam  # slow import

        return build_sam(self.args.model)

    def _prepare_prompts(self, dst_shape, src_shape, bboxes=None, points=None, labels=None, masks=None):
        """Prepare and transform the input prompts for processing based on the destination shape.

        Args:
            dst_shape (tuple[int, int]): The target shape (height, width) for the prompts.
            src_shape (tuple[int, int]): The source shape (height, width) of the input image.
            bboxes (np.ndarray | list | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | list | None): Points indicating object locations with shape (N, 2) or (N, num_points,
                2), in pixels.
            labels (np.ndarray | list | None): Point prompt labels with shape (N,) or (N, num_points). 1 for foreground,
                0 for background.
            masks (list | np.ndarray | None): Masks for the objects, where each mask is a 2D array.

        Returns:
            points (torch.Tensor | None): Transformed points.
            labels (torch.Tensor | None): Transformed labels.
            masks (torch.Tensor | None): Transformed masks.

        Raises:
            AssertionError: If the number of points don't match the number of labels, in case labels were passed.
        """
        bboxes, points, labels, masks = super()._prepare_prompts(dst_shape, src_shape, bboxes, points, labels, masks)
        if bboxes is not None:
            bboxes = bboxes.view(-1, 2, 2)
            bbox_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=bboxes.device).expand(bboxes.shape[0], -1)
            # NOTE: merge "boxes" and "points" into a single "points" input
            # (where boxes are added at the beginning) to model.sam_prompt_encoder
            if points is not None:
                points = torch.cat([bboxes, points], dim=1)
                labels = torch.cat([bbox_labels, labels], dim=1)
            else:
                points, labels = bboxes, bbox_labels
        return points, labels, masks

    def setup_source(self, source):
        """Set up the data source and image size for SAM2 inference."""
        super().setup_source(source)
        self._bb_feat_sizes = [[int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]]

    def get_im_features(self, im):
        """Extract image features from the SAM image encoder for subsequent processing."""
        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size) for feat, feat_size in zip(vision_feats, self._bb_feat_sizes)
        ]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}

    def _inference_features(
        self,
        features,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        img_idx=-1,
    ):
        """Perform inference on image features using the SAM2 model.

        Args:
            features (torch.Tensor | dict[str, Any]): Extracted image features with shape (B, C, H, W) from the SAM2
            model image encoder, it could also be a dictionary including:
                - image_embed (torch.Tensor): Image embedding with shape (B, C, H, W).
                - high_res_feats (list[torch.Tensor]): List of high-resolution feature maps from the backbone, each with shape (B, C, H, W).
            points (np.ndarray | list[list[float]] | None): Object location points with shape (N, 2), in pixels.
            labels (np.ndarray | list[int] | None): Point prompt labels with shape (N,). 1 = foreground, 0 = background.
            masks (list[np.ndarray] | np.ndarray | None): Masks for the objects, where each mask is a 2D array.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.
            img_idx (int): Index of the image in the batch to process.

        Returns:
            pred_masks (torch.Tensor): Output masks with shape (C, H, W), where C is the number of generated masks.
            pred_scores (torch.Tensor): Quality scores for each mask, with length C.
        """
        points = (points, labels) if points is not None else None
        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=masks,
        )
        # Predict masks
        batched_mode = points is not None and points[0].shape[0] > 1  # multi object prediction
        high_res_features = None
        if isinstance(features, dict):
            high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
            features = features["image_embed"][[img_idx]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )
        # (N, d, H, W) --> (N*d, H, W), (N, d) --> (N*d, )
        # `d` could be 1 or 3 depends on `multimask_output`.
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)


class SAM2VideoPredictor(SAM2Predictor):
    """SAM2VideoPredictor to handle user interactions with videos and manage inference states.

    This class extends the functionality of SAM2Predictor to support video processing and maintains the state of
    inference operations. It includes configurations for managing non-overlapping masks, clearing memory for
    non-conditional inputs, and setting up callbacks for prediction events.

    Attributes:
        inference_state (dict): A dictionary to store the current state of inference operations.
        non_overlap_masks (bool): A flag indicating whether masks should be non-overlapping.
        clear_non_cond_mem_around_input (bool): A flag to control clearing non-conditional memory around inputs.
        clear_non_cond_mem_for_multi_obj (bool): A flag to control clearing non-conditional memory for multi-object
            scenarios.
        callbacks (dict): A dictionary of callbacks for various prediction lifecycle events.

    Methods:
        get_model: Retrieve and configure the model with binarization enabled.
        inference: Perform image segmentation inference based on the given input cues.
        postprocess: Post-process the predictions to apply non-overlapping constraints if required.
        add_new_prompts: Add new points or masks to a specific frame for a given object ID.
        propagate_in_video_preflight: Prepare inference_state and consolidate temporary outputs before tracking.
        init_state: Initialize an inference state for the predictor.
        get_im_features: Extract image features using SAM2's image encoder for subsequent segmentation tasks.

    Examples:
        >>> predictor = SAM2VideoPredictor(cfg=DEFAULT_CFG)
        >>> predictor.set_image("path/to/video_frame.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> results = predictor(bboxes=bboxes)

    Notes:
        The `fill_hole_area` attribute is defined but not used in the current implementation.
    """

    # fill_hole_area = 8  # not used

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the predictor with configuration and optional overrides.

        This constructor initializes the SAM2VideoPredictor with a given configuration, applies any specified overrides,
        and sets up the inference state along with certain flags that control the behavior of the predictor.

        Args:
            cfg (dict): Configuration dictionary containing default settings.
            overrides (dict | None): Dictionary of values to override default configuration.
            _callbacks (dict | None): Dictionary of callback functions to customize behavior.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}
        self.non_overlap_masks = True
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.callbacks["on_predict_start"].append(self.init_state)
        self.clear_non_cond_mem = True  # Whether to clear non-conditioning memory periodically

    def get_model(self):
        """Retrieve and configure the model with binarization enabled.

        Notes:
            This method overrides the base class implementation to set the binarize flag to True.
        """
        model = super().get_model()
        model.set_binarize(True)
        return model

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
        """Perform image segmentation inference based on the given input cues, using the currently loaded image. This
        method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt
        encoder, and mask decoder for real-time and promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | list, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | list, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | list, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            masks (np.ndarray, optional): Low-resolution masks from previous predictions shape (N,H,W). For SAM H=W=256.

        Returns:
            pred_masks (torch.Tensor): The output masks in shape CxHxW, where C is the number of generated masks.
            pred_scores (torch.Tensor): An array of length C containing predicted quality scores for each mask.
        """
        # Override prompts if any stored in self.prompts
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        frame = self.dataset.frame
        self.inference_state["im"] = im
        output_dict = self.inference_state["output_dict"]
        if len(output_dict["cond_frame_outputs"]) == 0:  # initialize prompts
            points, labels, masks = self._prepare_prompts(
                im.shape[2:], self.batch[1][0].shape[:2], bboxes, points, labels, masks
            )
            if points is not None:
                for i in range(len(points)):
                    self.add_new_prompts(obj_id=i, points=points[[i]], labels=labels[[i]], frame_idx=frame)
            elif masks is not None:
                for i in range(len(masks)):
                    self.add_new_prompts(obj_id=i, masks=masks[[i]], frame_idx=frame)
        self.propagate_in_video_preflight()

        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        batch_size = len(self.inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                # clear non-conditioning memory of the surrounding frames
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame)
        # Create slices of per-object outputs for subsequent interaction with each
        # individual object after tracking.
        self._add_output_per_object(frame, current_out, storage_key)
        self.inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        pred_masks = pred_masks[(pred_masks > self.model.mask_threshold).sum((1, 2)) > 0]  # filter blank masks

        return pred_masks, torch.ones(pred_masks.shape[0], dtype=pred_masks.dtype, device=pred_masks.device)

    def postprocess(self, preds, img, orig_imgs):
        """Post-process the predictions to apply non-overlapping constraints if required.

        This method extends the post-processing functionality by applying non-overlapping constraints to the predicted
        masks if the `non_overlap_masks` flag is set to True. This ensures that the masks do not overlap, which can be
        useful for certain applications.

        Args:
            preds (tuple[torch.Tensor, torch.Tensor]): The predicted masks and scores from the model.
            img (torch.Tensor): The processed image tensor.
            orig_imgs (list[np.ndarray]): The original images before processing.

        Returns:
            (list): The post-processed predictions.

        Notes:
            If `non_overlap_masks` is True, the method applies constraints to ensure non-overlapping masks.
        """
        results = super().postprocess(preds, img, orig_imgs)
        if self.non_overlap_masks:
            for result in results:
                if result.masks is None or len(result.masks) == 0:
                    continue
                result.masks.data = self.model._apply_non_overlapping_constraints(result.masks.data.unsqueeze(0))[0]
        return results

    @smart_inference_mode()
    def add_new_prompts(
        self,
        obj_id,
        points=None,
        labels=None,
        masks=None,
        frame_idx=0,
        inference_state: dict[str, Any] | None = None,
    ):
        """Add new points or masks to a specific frame for a given object ID.

        This method updates the inference state with new prompts (points or masks) for a specified object and frame
        index. It ensures that the prompts are either points or masks, but not both, and updates the internal state
        accordingly. It also handles the generation of new segmentations based on the provided prompts and the existing
        state.

        Args:
            obj_id (int): The ID of the object to which the prompts are associated.
            points (torch.Tensor, optional): The coordinates of the points of interest.
            labels (torch.Tensor, optional): The labels corresponding to the points.
            masks (torch.Tensor, optional): Binary masks for the object.
            frame_idx (int, optional): The index of the frame to which the prompts are applied.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.

        Returns:
            pred_masks (torch.Tensor): The flattened predicted masks.
            pred_scores (torch.Tensor): A tensor of ones indicating the number of objects.

        Raises:
            AssertionError: If both `masks` and `points` are provided, or neither is provided.

        Notes:
            - Only one type of prompt (either points or masks) can be added per call.
            - If the frame is being tracked for the first time, it is treated as an initial conditioning frame.
            - The method handles the consolidation of outputs and resizing of masks to the original video resolution.
        """
        inference_state = inference_state or self.inference_state
        assert (masks is None) ^ (points is None), "'masks' and 'points' prompts are not compatible with each other."
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)

        point_inputs = None
        pop_key = "point_inputs_per_obj"
        if points is not None:
            point_inputs = {"point_coords": points, "point_labels": labels}
            inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
            pop_key = "mask_inputs_per_obj"
        inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = masks
        inference_state[pop_key][obj_idx].pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.model.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        if point_inputs is not None:
            prev_out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )

            if prev_out is not None and prev_out.get("pred_masks") is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].to(
                    device=self.device, non_blocking=self.device.type == "cuda"
                )
                # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
                prev_sam_mask_logits.clamp_(-32.0, 32.0)
        current_out = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=masks,
            reverse=False,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
            inference_state=inference_state,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            inference_state=inference_state,
        )
        pred_masks = consolidated_out["pred_masks"].flatten(0, 1)
        return pred_masks.flatten(0, 1), torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device)

    @smart_inference_mode()
    def propagate_in_video_preflight(self, inference_state: dict[str, Any] | None = None):
        """Prepare inference_state and consolidate temporary outputs before tracking.

        This method marks the start of tracking, disallowing the addition of new objects until the session is reset. It
        consolidates temporary outputs from `temp_output_dict_per_obj` and merges them into `output_dict`. Additionally,
        it clears non-conditioning memory around input frames and ensures that the state is consistent with the provided
        inputs.

        Args:
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.
        """
        inference_state = inference_state or self.inference_state
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = len(inference_state["obj_idx_to_id"])

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in {False, True}:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True, inference_state=inference_state
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(frame_idx, consolidated_out, storage_key, inference_state=inference_state)
                if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"] | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    @staticmethod
    def init_state(predictor):
        """Initialize an inference state for the predictor.

        This function sets up the initial state required for performing inference on video data. It includes
        initializing various dictionaries and ordered dictionaries that will store inputs, outputs, and other metadata
        relevant to the tracking process.

        Args:
            predictor (SAM2VideoPredictor): The predictor object for which to initialize the state.
        """
        if len(predictor.inference_state) > 0:  # means initialized
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        predictor.inference_state = predictor._init_state(predictor.dataset.frames)

    @staticmethod
    def _init_state(num_frames):
        """Initialize an inference state.

        This function sets up the initial state required for performing inference on video data. It includes
        initializing various dictionaries and ordered dictionaries that will store inputs, outputs, and other metadata
        relevant to the tracking process.

        Args:
            num_frames (int): The number of frames in the video.
        """
        inference_state = {
            "num_frames": num_frames,  # TODO: see if there's any chance to remove it
            "point_inputs_per_obj": {},  # inputs points on each frame
            "mask_inputs_per_obj": {},  # inputs mask on each frame
            "constants": {},  # values that don't change across frames (so we only need to hold one copy of them)
            # mapping between client-side object id and model-side object index
            "obj_id_to_idx": OrderedDict(),
            "obj_idx_to_id": OrderedDict(),
            "obj_ids": [],
            # A storage to hold the model's tracking results and states on each frame
            "output_dict": {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            },
            # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
            "output_dict_per_obj": {},
            # A temporary storage to hold new outputs when user interact with a frame
            # to add clicks or mask (it's merged into "output_dict" before propagation starts)
            "temp_output_dict_per_obj": {},
            # Frames that already holds consolidated outputs from click or mask inputs
            # (we directly use their consolidated outputs during tracking)
            "consolidated_frame_inds": {
                "cond_frame_outputs": set(),  # set containing frame indices
                "non_cond_frame_outputs": set(),  # set containing frame indices
            },
            # metadata for each tracking frame (e.g. which direction it's tracked)
            "tracking_has_started": False,
            "frames_already_tracked": [],
        }
        return inference_state

    def get_im_features(self, im, batch=1):
        """Extract and process image features using SAM2's image encoder for subsequent segmentation tasks.

        Args:
            im (torch.Tensor): The input image tensor.
            batch (int, optional): The batch size for expanding features if there are multiple prompts.

        Returns:
            vis_feats (torch.Tensor): The visual features extracted from the image.
            vis_pos_embed (torch.Tensor): The positional embeddings for the visual features.
            feat_sizes (list[tuple]): A list containing the sizes of the extracted features.

        Notes:
            - If `batch` is greater than 1, the features are expanded to fit the batch size.
            - The method leverages the model's `_prepare_backbone_features` method to prepare the backbone features.
        """
        # check if there's precomputed backbone output
        backbone_out = getattr(self, "backbone_out", None)
        if backbone_out is None:
            backbone_out = self.model.forward_image(im)
        _, vis_feats, vis_pos_embed, feat_sizes = self.model._prepare_backbone_features(backbone_out, batch=batch)
        return vis_feats, vis_pos_embed, feat_sizes

    def _obj_id_to_idx(self, obj_id, inference_state: dict[str, Any] | None = None):
        """Map client-side object id to model-side object index.

        Args:
            obj_id (int): The unique identifier of the object provided by the client side.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.

        Returns:
            (int): The index of the object on the model side.

        Raises:
            RuntimeError: If an attempt is made to add a new object after tracking has started.

        Notes:
            - The method updates or retrieves mappings between object IDs and indices stored in
              `inference_state`.
            - It ensures that new objects can only be added before tracking commences.
            - It maintains two-way mappings between IDs and indices (`obj_id_to_idx` and `obj_idx_to_id`).
            - Additional data structures are initialized for the new object to store inputs and outputs.
        """
        inference_state = inference_state or self.inference_state
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _run_single_frame_inference(
        self,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        inference_state: dict[str, Any] | None = None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory.

        Args:
            output_dict (dict): The dictionary containing the output states of the tracking process.
            frame_idx (int): The index of the current frame.
            batch_size (int): The batch size for processing the frame.
            is_init_cond_frame (bool): Indicates if the current frame is an initial conditioning frame.
            point_inputs (dict | None): Input points and their labels.
            mask_inputs (torch.Tensor | None): Input binary masks.
            reverse (bool): Indicates if the tracking should be performed in reverse order.
            run_mem_encoder (bool): Indicates if the memory encoder should be executed.
            prev_sam_mask_logits (torch.Tensor | None): Previous mask logits for the current object.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.

        Returns:
            (dict): A dictionary containing the output of the tracking step, including updated features and predictions.

        Raises:
            AssertionError: If both `point_inputs` and `mask_inputs` are provided, or neither is provided.

        Notes:
            - The method assumes that `point_inputs` and `mask_inputs` are mutually exclusive.
            - The method retrieves image features using the `get_im_features` method.
            - The `maskmem_pos_enc` is assumed to be constant across frames, hence only one copy is stored.
            - The `fill_holes_in_mask_scores` function is commented out and currently unsupported due to CUDA extension requirements.
        """
        inference_state = inference_state or self.inference_state
        # Retrieve correct image features
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(
            inference_state["im"], batch_size
        )

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )

        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            current_out["maskmem_features"] = maskmem_features.to(
                dtype=torch.float16, device=self.device, non_blocking=self.device.type == "cuda"
            )
        # NOTE: Do not support the `fill_holes_in_mask_scores` function since it needs cuda extensions
        # potentially fill holes in the predicted masks
        # if self.fill_hole_area > 0:
        #     pred_masks = current_out["pred_masks"].to(self.device, non_blocking=self.device.type == "cuda")
        #     pred_masks = fill_holes_in_mask_scores(pred_masks, self.fill_hole_area)

        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        current_out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(current_out["maskmem_pos_enc"], inference_state)
        return current_out

    def _get_maskmem_pos_enc(self, out_maskmem_pos_enc, inference_state: dict[str, Any] | None = None):
        """Cache and manage the positional encoding for mask memory across frames and objects.

        This method optimizes storage by caching the positional encoding (`maskmem_pos_enc`) for mask memory, which is
        constant across frames and objects, thus reducing the amount of redundant information stored during an inference
        session. It checks if the positional encoding has already been cached; if not, it caches a slice of the provided
        encoding. If the batch size is greater than one, it expands the cached positional encoding to match the current
        batch size.

        Args:
            out_maskmem_pos_enc (list[torch.Tensor] | None): The positional encoding for mask memory. Should be a list
                of tensors or None.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.

        Returns:
            (list[torch.Tensor]): The positional encoding for mask memory, either cached or expanded.

        Notes:
            - The method assumes that `out_maskmem_pos_enc` is a list of tensors or None.
            - Only a single object's slice is cached since the encoding is the same across objects.
            - The method checks if the positional encoding has already been cached in the session's constants.
            - If the batch size is greater than one, the cached encoding is expanded to fit the batch size.
        """
        inference_state = inference_state or self.inference_state
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].shape[0]
            if batch_size > 1:
                out_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        return out_maskmem_pos_enc

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond=False,
        run_mem_encoder=False,
        inference_state: dict[str, Any] | None = None,
    ):
        """Consolidate per-object temporary outputs into a single output for all objects.

        This method combines the temporary outputs for each object on a given frame into a unified
        output. It fills in any missing objects either from the main output dictionary or leaves
        placeholders if they do not exist in the main output. Optionally, it can re-run the memory encoder after
        applying non-overlapping constraints to the object scores.

        Args:
            frame_idx (int): The index of the frame for which to consolidate outputs.
            is_cond (bool, optional): Indicates if the frame is considered a conditioning frame.
            run_mem_encoder (bool, optional): Specifies whether to run the memory encoder after consolidating the
                outputs.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.

        Returns:
            (dict): A consolidated output dictionary containing the combined results for all objects.

        Notes:
            - The method initializes the consolidated output with placeholder values for missing objects.
            - It searches for outputs in both the temporary and main output dictionaries.
            - If `run_mem_encoder` is True, it applies non-overlapping constraints and re-runs the memory encoder.
            - The `maskmem_features` and `maskmem_pos_enc` are only populated when `run_mem_encoder` is True.
        """
        inference_state = inference_state or self.inference_state
        batch_size = len(inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                # size=(batch_size, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                size=(batch_size, 1, *self._bb_feat_sizes[0]),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(batch_size, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value=10.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = (
                obj_temp_output_dict[storage_key].get(frame_idx)
                # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
                # we fall back and look up its previous output in "output_dict_per_obj".
                # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
                # "output_dict_per_obj" to find a previous output for this object.
                or obj_output_dict["cond_frame_outputs"].get(frame_idx)
                or obj_output_dict["non_cond_frame_outputs"].get(frame_idx)
            )
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = self._get_empty_mask_ptr(frame_idx)
                continue
            # Add the temporary object output mask to consolidated output mask
            consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = out["pred_masks"]
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

        # Optionally, apply non-overlapping constraints on the consolidated scores and rerun the memory encoder
        if run_mem_encoder:
            high_res_masks = F.interpolate(
                consolidated_out["pred_masks"],
                size=self.imgsz,
                mode="bilinear",
                align_corners=False,
            )
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
            consolidated_out["maskmem_features"], consolidated_out["maskmem_pos_enc"] = self._run_memory_encoder(
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # these frames are what the user interacted with
                object_score_logits=consolidated_out["object_score_logits"],
                inference_state=inference_state,
            )

        return consolidated_out

    def _get_empty_mask_ptr(self, frame_idx, inference_state: dict[str, Any] | None = None):
        """Get a dummy object pointer based on an empty mask on the current frame.

        Args:
            frame_idx (int): The index of the current frame for which to generate the dummy object pointer.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.

        Returns:
            (torch.Tensor): A tensor representing the dummy object pointer generated from the empty mask.
        """
        inference_state = inference_state or self.inference_state
        # Retrieve correct image features
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(inference_state["im"])

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            # A dummy (empty) mask with a single object
            mask_inputs=torch.zeros((1, 1, *self.imgsz), dtype=self.torch_dtype, device=self.device),
            output_dict={},
            num_frames=inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def _run_memory_encoder(
        self,
        batch_size,
        high_res_masks,
        object_score_logits,
        is_mask_from_pts,
        inference_state: dict[str, Any] | None = None,
    ):
        """Run the memory encoder on masks.

        This is usually after applying non-overlapping constraints to object scores. Since their scores changed, their
        memory also needs to be computed again with the memory encoder.

        Args:
            batch_size (int): The batch size for processing the frame.
            high_res_masks (torch.Tensor): High-resolution masks for which to compute the memory.
            object_score_logits (torch.Tensor): Logits representing the object scores.
            is_mask_from_pts (bool): Indicates if the mask is derived from point interactions.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.

        Returns:
            maskmem_features (torch.Tensor): The encoded mask features.
            maskmem_pos_enc (torch.Tensor): The positional encoding.
        """
        inference_state = inference_state or self.inference_state
        # Retrieve correct image features
        current_vision_feats, _, feat_sizes = self.get_im_features(inference_state["im"], batch_size)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits,
        )

        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(maskmem_pos_enc, inference_state)
        return maskmem_features.to(
            dtype=torch.float16, device=self.device, non_blocking=self.device.type == "cuda"
        ), maskmem_pos_enc

    def _add_output_per_object(
        self, frame_idx, current_out, storage_key, inference_state: dict[str, Any] | None = None
    ):
        """Split a multi-object output into per-object output slices and add them into Output_Dict_Per_Obj.

        The resulting slices share the same tensor storage.

        Args:
            frame_idx (int): The index of the current frame.
            current_out (dict): The current output dictionary containing multi-object outputs.
            storage_key (str): The key used to store the output in the per-object output dictionary.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.
        """
        inference_state = inference_state or self.inference_state
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        for obj_idx, obj_output_dict in inference_state["output_dict_per_obj"].items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

    def _clear_non_cond_mem_around_input(self, frame_idx, inference_state: dict[str, Any] | None = None):
        """Remove the non-conditioning memory around the input frame.

        When users provide correction clicks, the surrounding frames' non-conditioning memories can still contain
        outdated object appearance information and could confuse the model. This method clears those non-conditioning
        memories surrounding the interacted frame to avoid giving the model both old and new information about the
        object.

        Args:
            frame_idx (int): The index of the current frame where user interaction occurred.
            inference_state (dict[str, Any], optional): The current inference state. If None, uses the instance's
                inference state.
        """
        inference_state = inference_state or self.inference_state
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        for t in range(frame_idx_begin, frame_idx_end + 1):
            inference_state["output_dict"]["non_cond_frame_outputs"].pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    @smart_inference_mode()
    def remove_object(self, inference_state, obj_id, strict=False):
        """Remove an object id from the tracking state. If strict is True, we check whether the object id actually
        exists and raise an error if it doesn't exist.
        """
        old_obj_idx_to_rm = inference_state["obj_id_to_idx"].get(obj_id, None)
        # Check whether this object_id to remove actually exists and possibly raise an error.
        if old_obj_idx_to_rm is None:
            if not strict:
                return inference_state["obj_ids"]
            raise RuntimeError(
                f"Cannot remove object id {obj_id} as it doesn't exist. "
                f"All existing object ids: {inference_state['obj_ids']}."
            )

        # If this is the only remaining object id, we simply reset the state.
        if len(inference_state["obj_id_to_idx"]) == 1:
            self.clear_all_points_in_video(inference_state)
            return inference_state["obj_ids"]

        # There are still remaining objects after removing this object id. In this case,
        # we need to delete the object storage from inference state tensors.
        # Step 0: clear the input on those frames where this object id has point or mask input
        # (note that this step is required as it might downgrade conditioning frames to
        # non-conditioning ones)
        obj_input_frames_inds = set()
        obj_input_frames_inds.update(inference_state["point_inputs_per_obj"][old_obj_idx_to_rm])
        obj_input_frames_inds.update(inference_state["mask_inputs_per_obj"][old_obj_idx_to_rm])
        for frame_idx in obj_input_frames_inds:
            self.clear_all_points_in_frame(inference_state, frame_idx, obj_id)

        # Step 1: Update the object id mapping (note that it must be done after Step 0,
        # since Step 0 still requires the old object id mappings in inference_state)
        old_obj_ids = inference_state["obj_ids"]
        old_obj_inds = list(range(len(old_obj_ids)))
        remain_old_obj_inds = old_obj_inds.copy()
        remain_old_obj_inds.remove(old_obj_idx_to_rm)
        new_obj_ids = [old_obj_ids[old_idx] for old_idx in remain_old_obj_inds]
        new_obj_inds = list(range(len(new_obj_ids)))
        # build new mappings
        old_idx_to_new_idx = dict(zip(remain_old_obj_inds, new_obj_inds))
        inference_state["obj_id_to_idx"] = dict(zip(new_obj_ids, new_obj_inds))
        inference_state["obj_idx_to_id"] = dict(zip(new_obj_inds, new_obj_ids))
        inference_state["obj_ids"] = new_obj_ids

        # Step 2: For per-object tensor storage, we shift their obj_idx in the dict keys.
        # (note that "consolidated_frame_inds" doesn't need to be updated in this step as
        # it's already handled in Step 0)
        def _map_keys(container):
            new_kvs = []
            for k in old_obj_inds:
                v = container.pop(k)
                if k in old_idx_to_new_idx:
                    new_kvs.append((old_idx_to_new_idx[k], v))
            container.update(new_kvs)

        _map_keys(inference_state["point_inputs_per_obj"])
        _map_keys(inference_state["mask_inputs_per_obj"])
        _map_keys(inference_state["output_dict_per_obj"])
        _map_keys(inference_state["temp_output_dict_per_obj"])

        # Step 3: For packed tensor storage, we index the remaining ids and rebuild the per-object slices.
        def _slice_state(output_dict, storage_key):
            for frame_idx, out in output_dict[storage_key].items():
                out["maskmem_features"] = out["maskmem_features"][remain_old_obj_inds]
                out["maskmem_pos_enc"] = [x[remain_old_obj_inds] for x in out["maskmem_pos_enc"]]
                # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
                out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(out["maskmem_pos_enc"], inference_state)
                out["pred_masks"] = out["pred_masks"][remain_old_obj_inds]
                out["obj_ptr"] = out["obj_ptr"][remain_old_obj_inds]
                out["object_score_logits"] = out["object_score_logits"][remain_old_obj_inds]
                # also update the per-object slices
                self._add_output_per_object(frame_idx, out, storage_key, inference_state=inference_state)

        _slice_state(inference_state["output_dict"], "cond_frame_outputs")
        _slice_state(inference_state["output_dict"], "non_cond_frame_outputs")

        return inference_state["obj_ids"]

    @smart_inference_mode()
    def clear_all_points_in_frame(self, inference_state, frame_idx, obj_id):
        """Remove all input points or mask in a specific frame for a given object."""
        obj_idx = self._obj_id_to_idx(obj_id, inference_state)

        # Clear the conditioning information on the given frame
        inference_state["point_inputs_per_obj"][obj_idx].pop(frame_idx, None)
        inference_state["mask_inputs_per_obj"][obj_idx].pop(frame_idx, None)

        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        temp_output_dict_per_obj[obj_idx]["cond_frame_outputs"].pop(frame_idx, None)
        temp_output_dict_per_obj[obj_idx]["non_cond_frame_outputs"].pop(frame_idx, None)

        # Check and see if there are still any inputs left on this frame
        batch_size = len(inference_state["obj_idx_to_id"])
        frame_has_input = False
        for obj_idx2 in range(batch_size):
            if frame_idx in inference_state["point_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break
            if frame_idx in inference_state["mask_inputs_per_obj"][obj_idx2]:
                frame_has_input = True
                break

        # If this frame has no remaining inputs for any objects, we further clear its
        # conditioning frame status
        if not frame_has_input:
            output_dict = inference_state["output_dict"]
            consolidated_frame_inds = inference_state["consolidated_frame_inds"]
            consolidated_frame_inds["cond_frame_outputs"].discard(frame_idx)
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)
            # Remove the frame's conditioning output (possibly downgrading it to non-conditioning)
            out = output_dict["cond_frame_outputs"].pop(frame_idx, None)
            if out is not None:
                # The frame is not a conditioning frame anymore since it's not receiving inputs,
                # so we "downgrade" its output (if exists) to a non-conditioning frame output.
                output_dict["non_cond_frame_outputs"][frame_idx] = out
                inference_state["frames_already_tracked"].pop(frame_idx, None)
            # Similarly, do it for the sliced output on each object.
            for obj_idx2 in range(batch_size):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx2]
                obj_out = obj_output_dict["cond_frame_outputs"].pop(frame_idx, None)
                if obj_out is not None:
                    obj_output_dict["non_cond_frame_outputs"][frame_idx] = obj_out

            # If all the conditioning frames have been removed, we also clear the tracking outputs
            if len(output_dict["cond_frame_outputs"]) == 0:
                self._reset_tracking_results(inference_state)

    @smart_inference_mode()
    def clear_all_points_in_video(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    @staticmethod
    def _reset_tracking_results(inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()
        inference_state["first_ann_frame_idx"] = None

    def _prune_non_cond_memory(self, frame_idx, inference_state=None):
        """Prune old non-conditioning frames to bound memory usage."""
        if not self.clear_non_cond_mem:
            return
        inference_state = inference_state or self.inference_state

        # Determine window size
        min_frame = frame_idx - self.model.num_maskmem * self.model.memory_temporal_stride_for_eval
        output_dict = inference_state["output_dict"]

        # Prune global non_cond_frame_outputs
        for f in [k for k in output_dict["non_cond_frame_outputs"] if k < min_frame]:
            output_dict["non_cond_frame_outputs"].pop(f, None)

        # Prune per-object non_cond_frame_outputs
        for obj_output_dict in inference_state.get("output_dict_per_obj", {}).values():
            for f in [k for k in obj_output_dict["non_cond_frame_outputs"] if k < min_frame]:
                obj_output_dict["non_cond_frame_outputs"].pop(f, None)


class SAM2DynamicInteractivePredictor(SAM2Predictor):
    """SAM2DynamicInteractivePredictor extends SAM2Predictor to support dynamic interactions with video frames or a
    sequence of images.

    Attributes:
        memory_bank (list): OrderedDict: Stores the states of each image with prompts.
        obj_idx_set (set): A set to keep track of the object indices that have been added.
        obj_id_to_idx (OrderedDict): Maps object IDs to their corresponding indices.
        obj_idx_to_id (OrderedDict): Maps object indices to their corresponding IDs.

    Methods:
        get_model: Retrieves and configures the model with binarization enabled.
        inference: Performs inference on a single image with optional prompts and object IDs.
        postprocess: Post-processes the predictions to apply non-overlapping constraints if required.
        update_memory: Append the imgState to the memory_bank and update the memory for the model.
        track_step: Tracking step for the current image state to predict masks.
        get_maskmem_enc: Get memory and positional encoding from the memory bank.

    Examples:
            >>> predictor = SAM2DynamicInteractivePredictor(cfg=DEFAULT_CFG)
            >>> predictor(source=support_img1, bboxes=bboxes1, obj_ids=labels1, update_memory=True)
            >>> results1 = predictor(source=query_img1)
            >>> predictor(source=support_img2, bboxes=bboxes2, obj_ids=labels2, update_memory=True)
            >>> results2 = predictor(source=query_img2)
    """

    def __init__(
        self,
        cfg: Any = DEFAULT_CFG,
        overrides: dict[str, Any] | None = None,
        max_obj_num: int = 3,
        _callbacks: dict[str, Any] | None = None,
    ) -> None:
        """Initialize the predictor with configuration and optional overrides.

        This constructor initializes the SAM2DynamicInteractivePredictor with a given configuration, applies any
        specified overrides

        Args:
            cfg (dict[str, Any]): Configuration dictionary containing default settings.
            overrides (dict[str, Any] | None): Dictionary of values to override default configuration.
            max_obj_num (int): Maximum number of objects to track. Default is 3. this is set to keep fix feature size
                for the model.
            _callbacks (dict[str, Any] | None): Dictionary of callback functions to customize behavior.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.non_overlap_masks = True

        # Initialize the memory bank to store image states
        # NOTE: probably need to use dict for better query
        self.memory_bank = []

        # Initialize the object index set and mappings
        self.obj_idx_set = set()
        self.obj_id_to_idx = self.obj_idx_to_id = OrderedDict(enumerate(range(max_obj_num)))
        self._max_obj_num = max_obj_num

    @smart_inference_mode()
    def inference(
        self,
        im: torch.Tensor | np.ndarray,
        bboxes: list[list[float]] | None = None,
        masks: torch.Tensor | np.ndarray | None = None,
        points: list[list[float]] | None = None,
        labels: list[int] | None = None,
        obj_ids: list[int] | None = None,
        update_memory: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform inference on a single image with optional bounding boxes, masks, points and object IDs. It has two
        modes: one is to run inference on a single image without updating the memory, and the other is to update
        the memory with the provided prompts and object IDs. When update_memory is True, it will update the
        memory with the provided prompts and obj_ids. When update_memory is False, it will only run inference on
        the provided image without updating the memory.

        Args:
            im (torch.Tensor | np.ndarray): The input image tensor or numpy array.
            bboxes (list[list[float]] | None): Optional list of bounding boxes to update the memory.
            masks (list[torch.Tensor | np.ndarray] | None): Optional masks to update the memory.
            points (list[list[float]] | None): Optional list of points to update the memory, each point is [x, y].
            labels (list[int] | None): Optional list of object IDs corresponding to the points (>0 for positive, 0 for
                negative).
            obj_ids (list[int] | None): Optional list of object IDs corresponding to the prompts.
            update_memory (bool): Flag to indicate whether to update the memory with new objects.

        Returns:
            res_masks (torch.Tensor): The output masks in shape (C, H, W)
            object_score_logits (torch.Tensor): Quality scores for each mask
        """
        self.get_im_features(im)
        points, labels, masks = self._prepare_prompts(
            dst_shape=self.imgsz,
            src_shape=self.batch[1][0].shape[:2],
            points=points,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
        )

        if update_memory:
            if isinstance(obj_ids, int):
                obj_ids = [obj_ids]
            assert obj_ids is not None, "obj_ids must be provided when update_memory is True"
            assert masks is not None or points is not None, (
                "bboxes, masks, or points must be provided when update_memory is True"
            )
            if points is None:  # placeholder
                points = torch.zeros((len(obj_ids), 0, 2), dtype=self.torch_dtype, device=self.device)
                labels = torch.zeros((len(obj_ids), 0), dtype=torch.int32, device=self.device)
            if masks is not None:
                assert len(masks) == len(obj_ids), "masks and obj_ids must have the same length."
            assert len(points) == len(obj_ids), "points and obj_ids must have the same length."
            self.update_memory(obj_ids, points, labels, masks)

        current_out = self.track_step()
        pred_masks, pred_scores = current_out["pred_masks"], current_out["object_score_logits"]
        # filter the masks and logits based on the object indices
        if len(self.obj_idx_set) == 0:
            raise RuntimeError("No objects have been added to the state. Please add objects before inference.")
        idx = list(self.obj_idx_set)  # cls id
        pred_masks, pred_scores = pred_masks[idx], pred_scores[idx]
        # the original score are in [-32,32], and a object score larger than 0 means the object is present, we map it to [-1,1] range,
        # and use a activate function to make sure the object score logits are non-negative, so that we can use it as a mask
        pred_scores = torch.clamp_(pred_scores / 32, min=0)
        return pred_masks.flatten(0, 1), pred_scores.flatten(0, 1)

    def get_im_features(self, img: torch.Tensor | np.ndarray) -> None:
        """Initialize the image state by processing the input image and extracting features.

        Args:
            img (torch.Tensor | np.ndarray): The input image tensor or numpy array.
        """
        vis_feats, vis_pos_embed, feat_sizes = SAM2VideoPredictor.get_im_features(self, img, batch=self._max_obj_num)
        self.high_res_features = [
            feat.permute(1, 2, 0).view(*feat.shape[1:], *feat_size)
            for feat, feat_size in zip(vis_feats[:-1], feat_sizes[:-1])
        ]

        self.vision_feats = vis_feats
        self.vision_pos_embeds = vis_pos_embed
        self.feat_sizes = feat_sizes

    @smart_inference_mode()
    def update_memory(
        self,
        obj_ids: list[int] | None = None,
        points: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        masks: torch.Tensor | None = None,
    ) -> None:
        """Append the imgState to the memory_bank and update the memory for the model.

        Args:
            obj_ids (list[int]): List of object IDs corresponding to the prompts.
            points (torch.Tensor | None): Tensor of shape (B, N, 2) representing the input points for N objects.
            labels (torch.Tensor | None): Tensor of shape (B, N) representing the labels for the input points.
            masks (torch.Tensor | None): Optional tensor of shape (N, H, W) representing the input masks for N objects.
        """
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(self._max_obj_num, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(self._max_obj_num, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(self._max_obj_num, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value=-32,  # 10.0,
                dtype=self.torch_dtype,
                device=self.device,
            ),
        }

        for i, obj_id in enumerate(obj_ids):
            assert obj_id < self._max_obj_num
            obj_idx = self._obj_id_to_idx(int(obj_id))
            self.obj_idx_set.add(obj_idx)
            point, label = points[[i]], labels[[i]]
            mask = masks[[i]][None] if masks is not None else None
            # Currently, only bbox prompt or mask prompt is supported, so we assert that bbox is not None.
            assert point is not None or mask is not None, "Either bbox, points or mask is required"
            out = self.track_step(obj_idx, point, label, mask)
            if out is not None:
                obj_mask = out["pred_masks"]
                assert obj_mask.shape[-2:] == consolidated_out["pred_masks"].shape[-2:], (
                    f"Expected mask shape {consolidated_out['pred_masks'].shape[-2:]} but got {obj_mask.shape[-2:]} for object {obj_idx}."
                )
                consolidated_out["pred_masks"][obj_idx : obj_idx + 1] = obj_mask
                consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

                if "object_score_logits" in out.keys():
                    consolidated_out["object_score_logits"][obj_idx : obj_idx + 1] = out["object_score_logits"]

        high_res_masks = F.interpolate(
            consolidated_out["pred_masks"].to(self.device, non_blocking=self.device.type == "cuda"),
            size=self.imgsz,
            mode="bilinear",
            align_corners=False,
        )

        if self.model.non_overlap_masks_for_mem_enc:
            high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=self.vision_feats,
            feat_sizes=self.feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=consolidated_out["object_score_logits"],
            is_mask_from_pts=True,
        )
        consolidated_out["maskmem_features"] = maskmem_features
        consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
        self.memory_bank.append(consolidated_out)

    def _prepare_memory_conditioned_features(self, obj_idx: int | None) -> torch.Tensor:
        """Prepare memory-conditioned features for the current image state.

        If ``obj_idx`` is provided, features are prepared for a specific prompted object in the image. If ``obj_idx`` is
        None, features are prepared for all objects. If no memory is available, a no-memory embedding is added to the
        current vision features. Otherwise, memory from previous frames is used to condition the current vision features
        via a transformer attention mechanism.

        Args:
            obj_idx (int | None): The index of the object for which to prepare the features.

        Returns:
            pix_feat_with_mem (torch.Tensor): The memory-conditioned pixel features.
        """
        if len(self.memory_bank) == 0 or isinstance(obj_idx, int):
            # For initial conditioning frames, encode without using any previous memory.
            # Directly add the no-memory embedding (instead of using the transformer encoder).
            pix_feat_with_mem = self.vision_feats[-1] + self.model.no_mem_embed
        else:
            # for inference frames, use the memory features from previous frames
            memory, memory_pos_embed = self.get_maskmem_enc()
            pix_feat_with_mem = self.model.memory_attention(
                curr=self.vision_feats[-1:],
                curr_pos=self.vision_pos_embeds[-1:],
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=0,  # num_obj_ptr_tokens
            )
        # Reshape output (HW)BC => BCHW
        return pix_feat_with_mem.permute(1, 2, 0).view(
            self._max_obj_num,
            self.model.memory_attention.d_model,
            *self.feat_sizes[-1],
        )

    def get_maskmem_enc(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get memory and positional encoding from memory, which is used to condition the current image features."""
        to_cat_memory, to_cat_memory_pos_embed = [], []
        for consolidated_out in self.memory_bank:
            to_cat_memory.append(consolidated_out["maskmem_features"].flatten(2).permute(2, 0, 1))  # (H*W, B, C)
            maskmem_enc = consolidated_out["maskmem_pos_enc"][-1].flatten(2).permute(2, 0, 1)
            maskmem_enc = maskmem_enc + self.model.maskmem_tpos_enc[self.model.num_maskmem - 1]
            to_cat_memory_pos_embed.append(maskmem_enc)

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        return memory, memory_pos_embed

    def _obj_id_to_idx(self, obj_id: int) -> int | None:
        """Map client-side object id to model-side object index.

        Args:
            obj_id (int): The client-side object ID.

        Returns:
            (int): The model-side object index, or None if not found.
        """
        return self.obj_id_to_idx.get(obj_id, None)

    def track_step(
        self,
        obj_idx: int | None = None,
        point: torch.Tensor | None = None,
        label: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        """Tracking step for the current image state to predict masks.

        This method processes the image features and runs the SAM heads to predict masks. If obj_idx is provided, it
        processes the features for a specific prompted object in the image. If obj_idx is None, it processes the
        features for all objects in the image. The method supports both mask-based output without SAM and full SAM
        processing with memory-conditioned features.

        Args:
            obj_idx (int | None): The index of the object for which to predict masks. If None, it processes all objects.
            point (torch.Tensor | None): The coordinates of the points of interest with shape (N, 2).
            label (torch.Tensor | None): The labels corresponding to the points where 1 means positive clicks, 0 means
                negative clicks.
            mask (torch.Tensor | None): The mask input for the object with shape (H, W).

        Returns:
            current_out (dict[str, Any]): A dictionary containing the current output with mask predictions and object
                pointers. Keys include 'point_inputs', 'mask_inputs', 'pred_masks', 'pred_masks_high_res',
                'obj_ptr', 'object_score_logits'.
        """
        if mask is not None and self.model.use_mask_input_as_output_without_sam:
            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = self.vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.model.memory_attention.d_model, *self.feat_sizes[-1])
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = self.model._use_mask_as_output(mask)
        else:
            # Fuse visual features with previous memory features in the memory bank.
            pix_feat_with_mem = self._prepare_memory_conditioned_features(obj_idx)
            # If ``obj_idx`` is provided (i.e., prompts are being added), keep only the first feature map.
            pix_feat_with_mem = pix_feat_with_mem[:1] if obj_idx is not None else pix_feat_with_mem
            _, _, _, low_res_masks, high_res_masks, obj_ptr, object_score_logits = self.model._forward_sam_heads(
                backbone_features=pix_feat_with_mem,
                point_inputs={"point_coords": point, "point_labels": label} if obj_idx is not None else None,
                mask_inputs=mask,
                multimask_output=False,
                high_res_features=[feat[: pix_feat_with_mem.shape[0]] for feat in self.high_res_features],
            )
        return {
            "pred_masks": low_res_masks,
            "pred_masks_high_res": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }


class SAM3Predictor(SAM2Predictor):
    """Segment Anything Model 3 (SAM3) Interactive Predictor for image segmentation tasks."""

    _bb_feat_sizes = [
        (288, 288),
        (144, 144),
        (72, 72),
    ]
    stride = 14

    def setup_model(self, model=None, verbose=True):
        """Setup the SAM3 model with appropriate mean and standard deviation for preprocessing."""
        super().setup_model(model, verbose)
        # update mean and std
        self.mean = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)
        self.std = torch.tensor([127.5, 127.5, 127.5]).view(-1, 1, 1).to(self.device)

    def get_model(self):
        """Retrieve and initialize the Segment Anything Model 3 (SAM3) for image segmentation tasks."""
        from .build_sam3 import build_interactive_sam3  # slow import

        return build_interactive_sam3(self.args.model, compile=self.args.compile)


class SAM3SemanticPredictor(SAM3Predictor):
    """Segment Anything Model 3 (SAM3) Predictor for image segmentation tasks."""

    def get_model(self):
        """Retrieve and initialize the Segment Anything Model 3 (SAM3) for image segmentation tasks."""
        from .build_sam3 import build_sam3_image_model  # slow import

        return build_sam3_image_model(self.args.model, compile=self.args.compile)

    @smart_inference_mode()
    def get_im_features(self, im):
        """Extract image features using the model's backbone."""
        return self.model.backbone.forward_image(im)

    def pre_transform(self, im):
        """Perform initial transformations on the input image for preprocessing.

        This method applies transformations such as resizing to prepare the image for further preprocessing. Currently,
        batched inference is not supported; hence the list length should be 1.

        Args:
            im (list[np.ndarray]): List containing a single image in HWC numpy array format.

        Returns:
            (list[np.ndarray]): List containing the transformed image.

        Raises:
            AssertionError: If the input list contains more than one image.

        Examples:
            >>> predictor = Predictor()
            >>> image = np.random.rand(480, 640, 3)  # Single HWC image
            >>> transformed = predictor.pre_transform([image])
            >>> print(len(transformed))
            1
        """
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.imgsz, auto=False, center=False, scale_fill=True)  # hardcode here for sam3
        return [letterbox(image=x) for x in im]

    def _prepare_geometric_prompts(self, src_shape, bboxes=None, labels=None):
        """Prepare prompts by normalizing bounding boxes and points to the destination shape."""
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=self.torch_dtype, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            # needs xywh as input
            bboxes = ops.xyxy2xywh(bboxes)
            bboxes[:, 0::2] /= src_shape[1]
            bboxes[:, 1::2] /= src_shape[0]
            # Assuming labels are all positive if users don't pass labels.
            if labels is None:
                labels = np.ones(bboxes.shape[:-1])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            assert bboxes.shape[-2] == labels.shape[-1], (
                f"Number of points {bboxes.shape[-2]} should match number of labels {labels.shape[-1]}."
            )
            bboxes = bboxes.view(-1, 1, 4)  # (N, 1, 4)
            labels = labels.view(-1, 1)  # (N, 1)
        return bboxes, labels

    def _inference_features(self, features, bboxes=None, labels=None, text: list[str] | None = None):
        """Run inference on the extracted features with optional bounding boxes and labels."""
        # NOTE: priority: bboxes > text > pre-set classes
        nc = 1 if bboxes is not None else len(text) if text is not None else len(self.model.names)
        geometric_prompt = self._get_dummy_prompt(nc)
        if bboxes is not None:
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
            if text is None:
                text = ["visual"]  # bboxes needs this `visual` text prompt if no text passed
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)
        outputs = self.model.forward_grounding(
            backbone_out=features,
            text_ids=torch.arange(nc, device=self.device, dtype=torch.long),
            geometric_prompt=geometric_prompt,
        )
        return outputs

    def postprocess(self, preds, img, orig_imgs):
        """Post-process the predictions to apply non-overlapping constraints if required."""
        pred_boxes = preds["pred_boxes"]  # (nc, num_query, 4)
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        keep = pred_scores > self.args.conf
        pred_masks = pred_masks[keep]
        pred_boxes = pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        names = getattr(self.model, "names", [str(i) for i in range(pred_scores.shape[0])])
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, boxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = F.interpolate(masks.float()[None], orig_img.shape[:2], mode="bilinear")[0] > 0.5
                boxes[..., [0, 2]] *= orig_img.shape[1]
                boxes[..., [1, 3]] *= orig_img.shape[0]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    def inference(self, im, bboxes=None, labels=None, text: list[str] | None = None, *args, **kwargs):
        """Perform inference on a single image with optional prompts."""
        bboxes = self.prompts.pop("bboxes", bboxes)
        labels = self.prompts.pop("labels", labels)
        text = self.prompts.pop("text", text)
        features = self.get_im_features(im) if self.features is None else self.features
        prompts = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        return self._inference_features(features, *prompts, text=text)

    @smart_inference_mode()
    def inference_features(
        self,
        features,
        src_shape,
        bboxes=None,
        labels=None,
        text: list[str] | None = None,
    ):
        """Perform prompts preprocessing and inference on provided image features using the SAM model.

        Args:
            features (dict[str, Any]): Extracted image features from the SAM3 model image encoder.
            src_shape (tuple[int, int]): The source shape (height, width) of the input image.
            bboxes (np.ndarray | list[list[float]] | None): Bounding boxes in xyxy format with shape (N, 4). pixels.
            labels (np.ndarray | list[int] | None): Point prompt labels with shape (N, ).
            text (list[str] | None): List of text prompts corresponding to the classes.

        Returns:
            pred_masks (torch.Tensor): The output masks in shape (C, H, W), where C is the number of generated masks.
            pred_bboxes (torch.Tensor): Bounding boxes for each mask with shape (N, 6), where N is the number of boxes.
                Each box is in xyxy format with additional columns for score and class.

        Notes:
            - The input features is a torch.Tensor of shape (B, C, H, W) if performing on SAM, or a dict[str, Any] if performing on SAM2.
        """
        prompts = self._prepare_geometric_prompts(src_shape[:2], bboxes, labels)
        preds = self._inference_features(features, *prompts, text=text)
        pred_boxes = preds["pred_boxes"]  # (nc, num_query, 4)
        pred_logits = preds["pred_logits"]
        pred_masks = preds["pred_masks"]
        pred_scores = pred_logits.sigmoid()
        presence_score = preds["presence_logit_dec"].sigmoid().unsqueeze(1)
        pred_scores = (pred_scores * presence_score).squeeze(-1)
        pred_cls = torch.tensor(
            list(range(pred_scores.shape[0])),
            dtype=pred_scores.dtype,
            device=pred_scores.device,
        )[:, None].expand_as(pred_scores)
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        keep = pred_scores > self.args.conf
        pred_masks = pred_masks[keep]
        pred_boxes = pred_boxes[keep]
        pred_boxes[:, :4] = ops.xywh2xyxy(pred_boxes[:, :4])

        if pred_masks.shape[0] == 0:
            pred_masks, pred_boxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            pred_masks = F.interpolate(pred_masks.float()[None], src_shape[:2], mode="bilinear")[0] > 0.5
            pred_boxes[..., 0] *= src_shape[1]
            pred_boxes[..., 1] *= src_shape[0]
            pred_boxes[..., 2] *= src_shape[1]
            pred_boxes[..., 3] *= src_shape[0]
        return pred_masks, pred_boxes

    def reset_prompts(self):
        """Reset the prompts for the predictor."""
        self.prompts = {}
        self.model.text_embeddings = {}

    def _get_dummy_prompt(self, num_prompts=1):
        """Get a dummy geometric prompt with zero boxes."""
        geometric_prompt = Prompt(
            box_embeddings=torch.zeros(0, num_prompts, 4, device=self.device),
            box_mask=torch.zeros(num_prompts, 0, device=self.device, dtype=torch.bool),
        )
        return geometric_prompt


class SAM3VideoPredictor(SAM2VideoPredictor, SAM3Predictor):
    """Segment Anything Model 3 (SAM3) Video Predictor for video segmentation tasks."""

    def propagate_in_video(self, inference_state, frame_idx):
        """Perform image segmentation inference based on the given input cues, using the currently loaded image. This
        method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt
        encoder, and mask decoder for real-time and promptable segmentation tasks.

        Args:
            inference_state (dict): The current state of inference, including input cues and previous outputs.
            frame_idx (int): The index of the current frame in the video sequence.
        """
        frame = frame_idx
        output_dict = inference_state["output_dict"]
        obj_ids = inference_state["obj_ids"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        batch_size = len(inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            if self.clear_non_cond_mem_around_input and (self.clear_non_cond_mem_for_multi_obj or batch_size <= 1):
                # clear non-conditioning memory of the surrounding frames
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out = self._run_single_frame_inference(
                output_dict=output_dict,
                frame_idx=frame,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=False,
                run_mem_encoder=True,
                inference_state=inference_state,
            )
            output_dict[storage_key][frame] = current_out
            self._prune_non_cond_memory(frame, inference_state=inference_state)
        # Create slices of per-object outputs for subsequent interaction with each
        # individual object after tracking.
        self._add_output_per_object(frame, current_out, storage_key, inference_state=inference_state)
        inference_state["frames_already_tracked"].append(frame)
        pred_masks = current_out["pred_masks"].flatten(0, 1)
        obj_scores = current_out["object_score_logits"]

        return obj_ids, pred_masks, obj_scores


class SAM3VideoSemanticPredictor(SAM3SemanticPredictor):
    """Segment Anything Model 3 (SAM3) Video Semantic Predictor."""

    HIGH_CONF_THRESH = 0.8
    HIGH_IOU_THRESH = 0.8
    NO_OBJ_LOGIT = -10.0
    NEVER_OCCLUDED = -1
    ALWAYS_OCCLUDED = 100000

    UNCONFIRMED = 1  # newly added masklet, not confirmed by any detection yet
    CONFIRMED = 2  # confirmed by at least one detection
    _bb_feat_sizes = [
        (288, 288),
        (144, 144),
        (72, 72),
    ]
    stride = 14

    def __init__(
        self,
        cfg=DEFAULT_CFG,
        overrides=None,
        _callbacks=None,
        # prob threshold for detection outputs -- only keep detections above this threshold
        # enters NMS and det-to-track matching
        score_threshold_detection=0.5,
        # IoU threshold for detection NMS
        det_nms_thresh=0.0,
        # IoU threshold for det-to-track matching -- a detection is considered "matched" to a tracklet it
        # overlaps with a tracklet above this threshold -- it is often a loose threshold like 0.1
        assoc_iou_thresh=0.5,
        # IoU threshold for det-to-track matching, which is used to determine whether a masklet is "unmatched"
        # by any detections -- it is often a stricter threshold like 0.5
        trk_assoc_iou_thresh=0.5,
        # prob threshold for a detection to be added as a new object
        new_det_thresh=0.0,
        # hotstart parameters: we hold off the outputs for `hotstart_delay` frames and
        # 1) remove those tracklets unmatched by any detections based on `hotstart_unmatch_thresh`
        # 2) remove those tracklets overlapping with one another based on `hotstart_dup_thresh`
        hotstart_delay=0,
        hotstart_unmatch_thresh=3,
        hotstart_dup_thresh=3,
        init_trk_keep_alive=30,
        max_trk_keep_alive=30,
        min_trk_keep_alive=-4,
        # Threshold for suppressing overlapping objects based on recent occlusion
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=True,
        o2o_matching_masklets_enable=False,  # Enable hungarian matching to match existing masklets
        suppress_det_close_to_boundary=False,
        fill_hole_area=16,
        # The maximum number of objects (masklets) to track across all GPUs (for no limit, set it to -1)
        max_num_objects=-1,
        recondition_every_nth_frame=-1,
        # masket confirmation status (to suppress unconfirmed masklets)
        masklet_confirmation_enable=False,
        # a masklet is confirmed after being consecutively detected and matched for
        # `masklet_confirmation_consecutive_det_thresh`
        masklet_confirmation_consecutive_det_thresh=3,
        # bbox heuristic parameters
        reconstruction_bbox_iou_thresh=0.0,
        reconstruction_bbox_det_score=0.0,
    ):
        """Initialize the SAM3VideoSemanticPredictor with configuration and optional overrides."""
        super().__init__(cfg, overrides, _callbacks)
        self.score_threshold_detection = score_threshold_detection
        self.det_nms_thresh = det_nms_thresh
        self.assoc_iou_thresh = assoc_iou_thresh
        self.trk_assoc_iou_thresh = trk_assoc_iou_thresh
        self.new_det_thresh = new_det_thresh

        # hotstart parameters
        if hotstart_delay > 0:
            assert hotstart_unmatch_thresh <= hotstart_delay
            assert hotstart_dup_thresh <= hotstart_delay
        self.hotstart_delay = hotstart_delay
        self.hotstart_unmatch_thresh = hotstart_unmatch_thresh
        self.hotstart_dup_thresh = hotstart_dup_thresh
        self.init_trk_keep_alive = init_trk_keep_alive
        self.max_trk_keep_alive = max_trk_keep_alive
        self.min_trk_keep_alive = min_trk_keep_alive
        self.suppress_overlapping_based_on_recent_occlusion_threshold = (
            suppress_overlapping_based_on_recent_occlusion_threshold
        )
        self.suppress_det_close_to_boundary = suppress_det_close_to_boundary
        self.decrease_trk_keep_alive_for_empty_masklets = decrease_trk_keep_alive_for_empty_masklets
        self.o2o_matching_masklets_enable = o2o_matching_masklets_enable
        self.fill_hole_area = fill_hole_area
        self._dist_pg_cpu = None  # CPU process group (lazy-initialized on first use)

        max_num_objects = 10000  # no limit
        num_obj_for_compile = 16
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = masklet_confirmation_consecutive_det_thresh
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

        # build SAM3 tracker
        self.tracker = SAM3VideoPredictor(overrides=overrides)

        self.inference_state = {}
        self.callbacks["on_predict_start"].append(self.init_state)

    def setup_model(self, model=None, verbose=True):
        """Setup the SAM3VideoSemanticPredictor model."""
        super().setup_model(model, verbose)
        from .build_sam3 import build_interactive_sam3

        # Initialize the SAM3 tracker model without backbone (backbone is handled in the detector)
        model = build_interactive_sam3(self.args.model, with_backbone=False)
        self.tracker.setup_model(model=model, verbose=False)

    def setup_source(self, source):
        """Setup the source for the SAM3VideoSemanticPredictor model."""
        super().setup_source(source)
        self.tracker.imgsz = self.imgsz
        self.tracker.model.set_imgsz(self.imgsz)
        self.tracker._bb_feat_sizes = [[int(x / (self.stride * i)) for x in self.imgsz] for i in [1 / 4, 1 / 2, 1]]
        self.interpol_size = self.tracker.model.memory_encoder.mask_downsampler.interpol_size

    @staticmethod
    def init_state(predictor):
        """Initialize an inference state for the predictor.

        This function sets up the initial state required for performing inference on video data. It includes
        initializing various dictionaries and ordered dictionaries that will store inputs, outputs, and other metadata
        relevant to the tracking process.

        Args:
            predictor (SAM3VideoSemanticPredictor): The predictor object for which to initialize the state.
        """
        if len(predictor.inference_state) > 0:  # means initialized
            return
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"
        num_frames = predictor.dataset.frames
        inference_state = {
            "num_frames": num_frames,
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "text_prompt": None,
            "per_frame_geometric_prompt": [None] * num_frames,
        }
        predictor.inference_state = inference_state

    def inference(self, im, bboxes=None, labels=None, text: list[str] | None = None, *args, **kwargs):
        """Perform inference on a video sequence with optional prompts."""
        frame = self.dataset.frame - 1  # align frame index to be 0-based
        self.inference_state["im"] = im  # only pass image for subsequent frames
        if "text_ids" not in self.inference_state:  # first frame processing
            self.add_prompt(frame_idx=frame, text=text, bboxes=bboxes, labels=labels)
        return self._run_single_frame_inference(frame, reverse=False)

    def postprocess(self, preds, img, orig_imgs):
        """Post-process the predictions to apply non-overlapping constraints if required."""
        obj_id_to_mask = preds["obj_id_to_mask"]  # low res masks
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        names = []
        if len(curr_obj_ids) == 0:
            pred_masks, pred_boxes = None, torch.zeros((0, 7), device=self.device)
        else:
            pred_masks = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)
            pred_masks = F.interpolate(pred_masks.float()[None], orig_imgs[0].shape[:2], mode="bilinear")[0] > 0.5
            pred_ids = torch.tensor(curr_obj_ids, dtype=torch.int32, device=pred_masks.device)
            pred_scores = torch.tensor(
                [preds["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids], device=pred_masks.device
            )
            pred_cls = torch.tensor(
                [preds["obj_id_to_cls"][obj_id] for obj_id in curr_obj_ids], device=pred_masks.device
            )
            keep = (pred_scores > self.args.conf) & pred_masks.any(dim=(1, 2))
            pred_masks = pred_masks[keep]
            pred_boxes = batched_mask_to_box(pred_masks)
            pred_boxes = torch.cat(
                [pred_boxes, pred_ids[keep][:, None], pred_scores[keep][..., None], pred_cls[keep][..., None]], dim=-1
            )
            if pred_masks.shape[0] > 1:
                tracker_scores = torch.tensor(
                    [
                        (
                            preds["obj_id_to_tracker_score"][obj_id]
                            if obj_id in preds["obj_id_to_tracker_score"]
                            else 0.0
                        )
                        for obj_id in curr_obj_ids
                    ],
                    device=pred_masks.device,
                )[keep]
                pred_masks = (
                    self._apply_object_wise_non_overlapping_constraints(
                        pred_masks.unsqueeze(1),
                        tracker_scores.unsqueeze(1),
                        background_value=0,
                    ).squeeze(1)
                ) > 0
                names = self.model.names or dict(enumerate(str(i) for i in range(pred_boxes[:, 6].int().max())))

        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    def _run_single_frame_inference(self, frame_idx, reverse=False, inference_state=None):
        """Perform inference on a single frame and get its inference results."""
        inference_state = inference_state or self.inference_state
        # prepare inputs
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        # run inference for the current frame
        (
            obj_id_to_mask,
            obj_id_to_score,
            obj_id_to_cls,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            im=inference_state["im"],
            text_ids=inference_state["text_ids"],
            geometric_prompt=(
                self._get_dummy_prompt(num_prompts=len(inference_state["text_ids"]))
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        # update inference state
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # first frame detection score
            "obj_id_to_cls": obj_id_to_cls,  # first frame detection score
            "obj_id_to_tracker_score": tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
        }
        # removed_obj_ids is only needed on rank 0 to handle hotstart delay buffer
        metadata = tracker_metadata_new["metadata"]
        removed_obj_ids = metadata["removed_obj_ids"]
        out["removed_obj_ids"] = removed_obj_ids
        out["frame_stats"] = frame_stats
        if self.masklet_confirmation_enable:
            status = metadata["masklet_confirmation"]["status"]
            is_unconfirmed = status == self.UNCONFIRMED
            out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][is_unconfirmed].tolist()
        else:
            out["unconfirmed_obj_ids"] = []
        return out

    @smart_inference_mode()
    def add_prompt(
        self,
        frame_idx,
        text=None,
        bboxes=None,
        labels=None,
        inference_state=None,
    ):
        """Add text, point or box prompts on a single frame. This method returns the inference outputs only on the
        prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.
        """
        inference_state = inference_state or self.inference_state
        assert text is not None or bboxes is not None, "at least one type of prompt (text, boxes) must be provided"

        # 1) handle text prompt
        use_text = text is not None
        text = text if use_text else "visual"
        text_batch = [text] if isinstance(text, str) else text
        inference_state["text_prompt"] = text if use_text else None
        n = len(text_batch)
        text_ids = torch.arange(n, device=self.device, dtype=torch.long)
        inference_state["text_ids"] = text_ids
        if text is not None and self.model.names != text:
            self.model.set_classes(text=text)

        # 2) handle box prompt
        bboxes, labels = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        assert (bboxes is not None) == (labels is not None)
        geometric_prompt = self._get_dummy_prompt(num_prompts=n)
        if bboxes is not None:
            for i in range(len(bboxes)):
                geometric_prompt.append_boxes(bboxes[[i]], labels[[i]])
        inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, out

    def _apply_object_wise_non_overlapping_constraints(self, pred_masks, obj_scores, background_value=-10.0):
        """Applies non-overlapping constraints object wise (i.e. only one object can claim the overlapping region)."""
        # Replace pixel scores with object scores
        pred_masks_single_score = torch.where(pred_masks > 0, obj_scores[..., None, None], background_value)
        # Apply pixel-wise non-overlapping constraint based on mask scores
        pixel_level_non_overlapping_masks = self.tracker.model._apply_non_overlapping_constraints(
            pred_masks_single_score
        )
        # Replace object scores with pixel scores. Note, that now only one object can claim the overlapping region
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks

    def _det_track_one_frame(
        self,
        im: torch.Tensor,
        text_ids: torch.Tensor,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        geometric_prompt: Prompt,
        tracker_states_local: list[Any],
        tracker_metadata_prev: dict[str, Any],
        allow_new_detections: bool = True,
    ):
        """This function handles one-step inference for the DenseTracking model in an SPMD manner. At a high-level, all
        GPUs execute the same function calls as if it's done on a single GPU, while under the hood, some
        function calls involve distributed computation based on sharded SAM2 states.

        - `input_batch` contains image and other inputs on the entire video; it should be identical across GPUs
        - `tracker_states_local` holds the local masklet information in this GPU shard
        - `tracker_metadata_prev` manages the metadata for SAM2 objects, such as which masklet is hold on which GPUs
          it contains both global and local masklet information
        """
        # Step 1: run backbone and detector in a distributed manner -- this is done via Sam3ImageOnVideoMultiGPU,
        # a MultiGPU model (assigned to `self.detector`) that shards frames in a round-robin manner.
        det_out = self.run_backbone_and_detection(
            im=im,
            text_ids=text_ids,
            geometric_prompt=geometric_prompt,
            allow_new_detections=allow_new_detections,
        )

        # Step 2: each GPU propagates its local SAM2 states to get the SAM2 prediction masks.
        # the returned `tracker_low_res_masks_global` contains the concatenated masklet predictions
        # gathered from all GPUs (as if they are propagated on a single GPU). Note that this step only
        # runs the SAM2 propagation step, but doesn't encode new memory for the predicted masks;
        # we defer memory encoding to `run_tracker_update_execution_phase` after resolving all heuristics.
        if tracker_metadata_prev == {}:
            # initialize masklet metadata if it's uninitialized (empty dict)
            tracker_metadata_prev.update(self._initialize_metadata())
        tracker_low_res_masks_global, tracker_obj_scores_global = self.run_tracker_propagation(
            frame_idx=frame_idx,
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=tracker_metadata_prev,
        )

        # Step 3: based on detection outputs and the propagated SAM2 prediction masks, we make plans
        # for SAM2 masklet updates (i.e. which objects to add and remove, how to load-balance them, etc).
        # We also run SAM2 memory encoder globally in this step to resolve non-overlapping constraints.
        # **This step should involve all the heuristics needed for any updates.** Most of the update
        # planning will be done on the master rank (GPU 0) and the resulting plan `tracker_update_plan` is
        # broadcasted to other GPUs (to be executed in a distributed manner). This step also generates the
        # new masklet metadata `tracker_metadata_new` (based on its previous version `tracker_metadata_prev`).
        tracker_update_plan, tracker_metadata_new = self.run_tracker_update_planning_phase(
            frame_idx=frame_idx,
            reverse=reverse,
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_obj_scores_global=tracker_obj_scores_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_states_local=tracker_states_local,
        )

        # Get reconditioning info from the update plan
        reconditioned_obj_ids = tracker_update_plan.get("reconditioned_obj_ids", set())

        # Step 4: based on `tracker_update_plan`, each GPU executes the update w.r.t. its local SAM2 inference states
        tracker_states_local_new = self.run_tracker_update_execution_phase(
            frame_idx=frame_idx,
            num_frames=num_frames,
            det_out=det_out,
            tracker_states_local=tracker_states_local,
            tracker_update_plan=tracker_update_plan,
        )

        # Step 5: finally, build the outputs for this frame (it only needs to be done on GPU 0 since
        # only GPU 0 will send outputs to the server).
        obj_id_to_mask = self.build_outputs(
            det_out=det_out,
            tracker_low_res_masks_global=tracker_low_res_masks_global,
            tracker_metadata_prev=tracker_metadata_prev,
            tracker_update_plan=tracker_update_plan,
            reconditioned_obj_ids=reconditioned_obj_ids,
        )
        obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        obj_id_to_cls = tracker_metadata_new["obj_id_to_cls"]
        # a few statistics for the current frame as a part of the output
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        # add tracker scores to metadata, it should be fired for frames except the first frame
        if tracker_obj_scores_global.shape[0] > 0:
            # Convert tracker_obj_scores_global to sigmoid scores before updating
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                dict(zip(tracker_obj_ids, tracker_obj_scores_global))
            )
        return (
            obj_id_to_mask,  # a dict: obj_id --> output mask
            obj_id_to_score,  # a dict: obj_id --> output score (prob)
            obj_id_to_cls,  # a dict: obj_id --> output cls (int)
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,  # a dict: obj_id --> tracker frame-level scores
        )

    @staticmethod
    def _suppress_detections_close_to_boundary(boxes, margin=0.025):
        """Suppress detections too close to image edges (for normalized boxes).

        boxes: (N, 4) in xyxy format, normalized [0,1]
        margin: fraction of image
        """
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (x_c > margin) & (x_c < 1.0 - margin) & (y_c > margin) & (y_c < 1.0 - margin)

        return keep

    def run_backbone_and_detection(
        self, im: torch.Tensor, text_ids: torch.Tensor, geometric_prompt: Prompt, allow_new_detections: bool
    ):
        """Run backbone and detection for a single frame."""
        features = self.get_im_features(im)
        sam3_image_out = self.model.forward_grounding(
            backbone_out=features, text_ids=text_ids, geometric_prompt=geometric_prompt
        )
        det_out = self._extract_detection_outputs(sam3_image_out, allow_new_detections)
        self._cache_backbone_features(sam3_image_out)
        return det_out

    def _extract_detection_outputs(self, sam3_image_out, allow_new_detections):
        """Extract and filter detection outputs."""
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        if not allow_new_detections:
            pred_probs = pred_probs - 1e8

        pred_cls = torch.tensor(
            list(range(pred_probs.shape[0])),
            dtype=pred_probs.dtype,
            device=pred_probs.device,
        )[:, None].expand_as(pred_probs)

        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]

        keep = pred_probs > self.score_threshold_detection
        return {
            "bbox": pred_boxes_xyxy[keep],
            "mask": pred_masks[keep],
            "scores": pred_probs[keep],
            "cls": pred_cls[keep],
        }

    def _cache_backbone_features(self, sam3_image_out):
        """Build and cache SAM2 backbone features."""
        sam_mask_decoder = self.tracker.model.sam_mask_decoder
        feats = sam3_image_out["backbone_out"]["sam2_backbone_out"]
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(feats["backbone_fpn"][0]),
            sam_mask_decoder.conv_s1(feats["backbone_fpn"][1]),
            feats["backbone_fpn"][2],
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],
            "vision_pos_enc": feats["vision_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        # cache the SAM2 backbone features for `frame_idx` in the tracker
        self.tracker.backbone_out = tracker_backbone_out

    def run_tracker_propagation(
        self, frame_idx: int, tracker_states_local: list[Any], tracker_metadata_prev: dict[str, np.ndarray]
    ):
        """Run the tracker propagation phase for a single frame in an SPMD manner."""
        # Step 1: propagate the local SAM2 states to get the current frame's prediction
        # `low_res_masks_local` of the existing masklets on this GPU
        # - obj_ids_local: list[int] -- list of object IDs
        # - low_res_masks_local: Tensor -- (num_local_obj, H_mask, W_mask)
        obj_ids_local, low_res_masks_local, obj_scores_local = self._propogate_tracker_one_frame_local_gpu(
            tracker_states_local, frame_idx=frame_idx
        )

        assert np.all(obj_ids_local == tracker_metadata_prev["obj_ids"]), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids"]
        )

        # Step 2: all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        low_res_masks_global = low_res_masks_local
        obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: dict[str, torch.Tensor],
        trk_id_to_max_iou_high_conf_det: list[int],
        tracker_states_local: list[Any],
        tracker_metadata: dict[str, np.ndarray],
        tracker_obj_scores_global: torch.Tensor,
    ):
        """Recondition masklets based on new high-confidence detections."""
        # Recondition the masklets based on the new detections
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            new_mask_binary = (
                F.interpolate(new_mask.unsqueeze(1), size=self.interpol_size, mode="bilinear", align_corners=False) > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids"] == trk_obj_id)[0].item()
            obj_score = tracker_obj_scores_global[obj_idx]
            for state_idx, inference_state in enumerate(tracker_states_local):
                if (
                    trk_obj_id in inference_state["obj_ids"]
                    # NOTE: Goal of this condition is to avoid reconditioning masks that are occluded/low qualiy.
                    # Unfortunately, these can get reconditioned anyway due to batching. We should consider removing these heuristics.
                    and obj_score > HIGH_CONF_THRESH
                ):
                    LOGGER.debug(
                        f"Adding new mask for track {trk_obj_id} at frame {frame_idx}. Objects {inference_state['obj_ids']} are all reconditioned."
                    )
                    self.tracker.add_new_prompts(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=trk_obj_id,
                        masks=new_mask_binary,
                    )
                    reconditioned_states_idx.add(state_idx)

            for idx in reconditioned_states_idx:
                self.tracker.propagate_in_video_preflight(tracker_states_local[idx])
        return tracker_states_local

    def run_tracker_update_planning_phase(
        self,
        frame_idx: int,
        reverse: bool,
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_obj_scores_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_states_local: list[Any],
    ):
        """Run the tracker update planning phase for a single frame in an SPMD manner."""
        # initialize new metadata from previous metadata (its values will be updated later)
        tracker_metadata_new = {
            "obj_ids": deepcopy(tracker_metadata_prev["obj_ids"]),
            "num_obj": deepcopy(tracker_metadata_prev["num_obj"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_cls": deepcopy(tracker_metadata_prev["obj_id_to_cls"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]),
            "obj_id_to_last_occluded": {},  # will be filled later
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }

        # Initialize reconditioned_obj_ids early to avoid UnboundLocalError
        reconditioned_obj_ids = set()

        # Step 1: make the update plan and resolve heuristics on GPU 0
        det_mask_preds: torch.Tensor = det_out["mask"]  # low-res mask logits
        det_scores_np: np.ndarray = det_out["scores"].float().cpu().numpy()
        det_cls_np: np.ndarray = det_out["cls"].float().cpu().numpy()
        det_bbox_xyxy: torch.Tensor = det_out["bbox"]
        # a) match detector and tracker masks and find new objects
        (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        ) = self._associate_det_trk(
            det_masks=det_mask_preds,
            det_scores_np=det_scores_np,
            trk_masks=tracker_low_res_masks_global,
            trk_obj_ids=tracker_metadata_prev["obj_ids"],
        )
        if self.suppress_det_close_to_boundary:
            keep = self._suppress_detections_close_to_boundary(det_bbox_xyxy[new_det_fa_inds])
            new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

        # check whether we've hit the maximum number of objects we can track (and if so, drop some detections)
        prev_obj_num = np.sum(tracker_metadata_prev["num_obj"])
        new_det_num = len(new_det_fa_inds)
        num_obj_dropped_due_to_limit = 0
        if prev_obj_num + new_det_num > self.max_num_objects:
            LOGGER.warning(f"hitting {self.max_num_objects=} with {new_det_num=} and {prev_obj_num=}")
            new_det_num_to_keep = self.max_num_objects - prev_obj_num
            num_obj_dropped_due_to_limit = new_det_num - new_det_num_to_keep
            new_det_fa_inds = self._drop_new_det_with_obj_limit(new_det_fa_inds, det_scores_np, new_det_num_to_keep)
            assert len(new_det_fa_inds) == new_det_num_to_keep
            new_det_num = len(new_det_fa_inds)

        # assign object IDs to new detections and decide which GPU to place them
        new_det_obj_ids = tracker_metadata_prev["max_obj_id"] + 1 + np.arange(new_det_num)

        # b) handle hotstart heuristics to remove objects
        # here `metadata` contains metadata stored on (and only accessible to) GPU 0;
        # we avoid broadcasting them to other GPUs to save communication cost, assuming
        # that `metadata` is not needed by other GPUs
        metadata_new = deepcopy(tracker_metadata_prev["metadata"])
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            obj_ids_newly_removed, metadata_new = self._process_hotstart(
                frame_idx=frame_idx,
                reverse=reverse,
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
                empty_trk_obj_ids=empty_trk_obj_ids,
                unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                metadata=metadata_new,
            )
        else:
            # if warm-up is not complete, we don't remove any objects
            obj_ids_newly_removed = set()
        tracker_metadata_new["metadata"] = metadata_new

        # `tracker_update_plan` should be identical on all GPUs after broadcasting
        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,  # np.ndarray
            "new_det_obj_ids": new_det_obj_ids,  # np.ndarray
            # "new_det_gpu_ids": new_det_gpu_ids,  # np.ndarray
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,  # np.ndarray
            "det_to_matched_trk_obj_ids": det_to_matched_trk_obj_ids,  # dict
            "obj_ids_newly_removed": obj_ids_newly_removed,  # set
            "num_obj_dropped_due_to_limit": num_obj_dropped_due_to_limit,  # int
            "trk_id_to_max_iou_high_conf_det": trk_id_to_max_iou_high_conf_det,  # dict
            "reconditioned_obj_ids": reconditioned_obj_ids,  # set
        }

        # Step 3 (optional): recondition masklets based on high-confidence detections before memory encoding
        # NOTE: Running this in execution phase (after memory encoding) can lead to suboptimal results
        should_recondition_iou = False

        # Evaluate tracklets for reconditioning based on bbox IoU mismatch with detections
        if self.reconstruction_bbox_iou_thresh > 0 and len(trk_id_to_max_iou_high_conf_det) > 0:
            for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
                det_box = det_out["bbox"][det_idx]
                det_score = det_out["scores"][det_idx]

                try:
                    trk_idx = list(tracker_metadata_prev["obj_ids"]).index(trk_obj_id)
                except ValueError:
                    continue  # Skip if tracklet not found

                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()

                if mask_area == 0:
                    continue  # Skip tracklets with zero mask area

                # Get bounding box from SAM2 mask and convert to normalized coordinates
                tracker_box_pixels = batched_mask_to_box(mask_binary.unsqueeze(0)).squeeze(0)
                mask_height, mask_width = tracker_mask.shape[-2:]
                tracker_box_normalized = torch.tensor(
                    [
                        tracker_box_pixels[0] / mask_width,
                        tracker_box_pixels[1] / mask_height,
                        tracker_box_pixels[2] / mask_width,
                        tracker_box_pixels[3] / mask_height,
                    ],
                    device=tracker_box_pixels.device,
                )

                # Compute IoU between detection and SAM2 tracklet bounding boxes
                det_box_batch = det_box.unsqueeze(0)
                tracker_box_batch = tracker_box_normalized.unsqueeze(0)
                iou = box_iou(det_box_batch, tracker_box_batch)[0]

                if iou < self.reconstruction_bbox_iou_thresh and det_score >= self.reconstruction_bbox_det_score:
                    should_recondition_iou = True
                    reconditioned_obj_ids.add(trk_obj_id)

        should_recondition_periodic = (
            self.recondition_every_nth_frame > 0
            and frame_idx % self.recondition_every_nth_frame == 0
            and len(trk_id_to_max_iou_high_conf_det) > 0
        )

        # Recondition if periodic or IoU condition met
        if should_recondition_periodic or should_recondition_iou:
            self._recondition_masklets(
                frame_idx,
                det_out,
                trk_id_to_max_iou_high_conf_det,
                tracker_states_local,
                tracker_metadata_prev,
                tracker_obj_scores_global,
            )

        # Step 4: Run SAM2 memory encoder on the current frame's prediction masks
        # This is done on all GPUs
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                if self.suppress_overlapping_based_on_recent_occlusion_threshold > 0.0:
                    # NOTE: tracker_low_res_masks_global is updated in-place then returned
                    tracker_low_res_masks_global = self._suppress_overlapping_based_on_recent_occlusion(
                        frame_idx,
                        tracker_low_res_masks_global,
                        tracker_metadata_prev,
                        tracker_metadata_new,
                        obj_ids_newly_removed,
                        reverse,
                    )

            self._tracker_update_memories(tracker_states_local, frame_idx, low_res_masks=tracker_low_res_masks_global)

        # Step 4: update the SAM2 metadata based on the update plan
        updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids"]
        if len(new_det_obj_ids) > 0:
            updated_obj_ids_this_gpu = np.concatenate([updated_obj_ids_this_gpu, new_det_obj_ids])
        if len(obj_ids_newly_removed) > 0:
            is_removed = np.isin(updated_obj_ids_this_gpu, list(obj_ids_newly_removed))
            updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
        tracker_metadata_new["obj_ids"] = updated_obj_ids_this_gpu
        tracker_metadata_new["num_obj"] = len(updated_obj_ids_this_gpu)
        # update object scores and the maximum object ID assigned so far
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            tracker_metadata_new["obj_id_to_cls"].update(zip(new_det_obj_ids, det_cls_np[new_det_fa_inds]))
            # tracker scores are not available for new objects, use det score instead.
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            tracker_metadata_new["max_obj_id"] = max(tracker_metadata_new["max_obj_id"], np.max(new_det_obj_ids))
        # for removed objects, we set their scores to a very low value (-1e4) but still
        # keep them in "obj_id_to_score" (it's easier to handle outputs this way)
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        # check that "metadata" is in tracker_metadata_new if and only if it's GPU 0
        assert "metadata" in tracker_metadata_new
        if self.masklet_confirmation_enable:
            metadata = self.update_masklet_confirmation_status(
                metadata=tracker_metadata_new["metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["metadata"] = metadata

        return tracker_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, Any],
        tracker_metadata_new: dict[str, Any],
        obj_ids_newly_removed: set[int],
        reverse: bool = False,
    ):
        """Suppress overlapping masks based on the most recent occlusion information. If an object is removed by
        hotstart, we always suppress it if it overlaps with any other object.

        Args:
            frame_idx (int): The current frame index.
            tracker_low_res_masks_global (torch.Tensor): The low-resolution masks for the current frame.
            tracker_metadata_prev (dict[str, Any]): The metadata from the previous frame.
            tracker_metadata_new (dict[str, Any]): The metadata for the current frame.
            obj_ids_newly_removed (set[int]): The object IDs that have been removed.
            reverse (bool): Whether the tracking is in reverse order.

        Returns:
            (torch.Tensor): The updated low-resolution masks with some objects suppressed.
        """
        obj_ids_global = tracker_metadata_prev["obj_ids"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert len(obj_ids_global) == batch_size, (
                f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            )
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(
                                self.NEVER_OCCLUDED if obj_id not in obj_ids_newly_removed else self.ALWAYS_OCCLUDED
                            ),
                            device=binary_tracker_low_res_masks_global.device,
                            dtype=torch.long,
                        ),
                    )
                    for obj_id in obj_ids_global
                ],
                dim=0,
            )
            to_suppress = self._get_objects_to_suppress_based_on_most_recently_occluded(
                binary_tracker_low_res_masks_global,
                last_occluded_prev,
                obj_ids_global,
                frame_idx,
                reverse,
            )

            # Update metadata with occlusion information
            is_obj_occluded = ~(binary_tracker_low_res_masks_global.any(dim=(-1, -2)))
            is_obj_occluded_or_suppressed = is_obj_occluded | to_suppress
            last_occluded_new = last_occluded_prev.clone()
            last_occluded_new[is_obj_occluded_or_suppressed] = frame_idx
            # Slice out the last occluded frame for each object
            tracker_metadata_new["obj_id_to_last_occluded"] = {
                obj_id: last_occluded_new[obj_idx : obj_idx + 1] for obj_idx, obj_id in enumerate(obj_ids_global)
            }

            # Zero out suppressed masks before memory encoding
            tracker_low_res_masks_global[to_suppress] = self.NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        det_out: dict[str, torch.Tensor],
        tracker_states_local: list[Any],
        tracker_update_plan: dict[str, np.ndarray],
    ):
        """Execute the tracker update plan for a single frame in an SPMD manner."""
        # initialize tracking scores with detection scores
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        # new_det_gpu_ids: np.ndarray = tracker_update_plan["new_det_gpu_ids"]
        new_det_obj_ids_local: np.ndarray = new_det_obj_ids
        new_det_fa_inds_local: np.ndarray = new_det_fa_inds
        obj_ids_newly_removed: set[int] = tracker_update_plan["obj_ids_newly_removed"]

        # Step 1: add new objects from the detector to SAM2 inference states
        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: torch.Tensor = det_out["mask"][new_det_fa_inds_local_t]
            # initialize SAM2 with new object masks
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
            )

        # Step 2: remove from SAM2 inference states those objects removed by heuristics
        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        return tracker_states_local

    @staticmethod
    def build_outputs(
        det_out: dict[str, torch.Tensor],
        tracker_low_res_masks_global: torch.Tensor,
        tracker_metadata_prev: dict[str, np.ndarray],
        tracker_update_plan: dict[str, np.ndarray],
        reconditioned_obj_ids: set | None = None,
    ):
        """Build the output masks for the current frame."""
        new_det_fa_inds: np.ndarray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: np.ndarray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}  # obj_id --> output mask tensor

        # Part 1: masks from previous SAM2 propagation
        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids"]
        existing_masklet_binary = tracker_low_res_masks_global.unsqueeze(1)
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # Part 2: masks from new detections
        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        assert len(new_det_obj_ids) == len(new_det_low_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_det_low_res_masks):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # Part 3: Override masks for reconditioned objects using detection masks
        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get("trk_id_to_max_iou_high_conf_det", {})

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)

                if det_idx is not None:
                    obj_id_to_mask[obj_id] = det_out["mask"][det_idx].unsqueeze(0)

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: torch.Tensor,
        last_occluded: list[int],
        obj_ids: list[int],
        frame_idx: int | None = None,
        reverse: bool = False,
    ):
        # Suppress overlapping masks for objects that were most recently occluded
        assert binary_low_res_masks.dtype == torch.bool, f"Expected boolean tensor, got {binary_low_res_masks.dtype}"
        to_suppress = torch.zeros(
            binary_low_res_masks.size(0),
            device=binary_low_res_masks.device,
            dtype=torch.bool,
        )
        if len(obj_ids) <= 1:
            return to_suppress

        iou = mask_iou(binary_low_res_masks.flatten(1), binary_low_res_masks.flatten(1))  # [N,N]

        # Create masks for upper triangular matrix (i < j) and IoU threshold
        mask_iou_thresh = iou >= self.suppress_overlapping_based_on_recent_occlusion_threshold
        overlapping_pairs = torch.triu(mask_iou_thresh, diagonal=1)  # [N,N]

        last_occ_expanded_i = last_occluded.unsqueeze(1)  # (N, 1)
        last_occ_expanded_j = last_occluded.unsqueeze(0)  # (1, N)
        # Suppress most recently occluded
        cmp_op = torch.gt if not reverse else torch.lt
        suppress_i_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_i, last_occ_expanded_j)  # (last_occ_expanded_i > last_occ_expanded_j)
            & (last_occ_expanded_j > -1)  # j can suppress i only if i was previously occluded
        )
        suppress_j_mask = (
            overlapping_pairs
            & cmp_op(last_occ_expanded_j, last_occ_expanded_i)
            & (last_occ_expanded_i > -1)  # i can suppress j only if j was previously occluded
        )
        # Apply suppression
        to_suppress = suppress_i_mask.any(dim=1) | suppress_j_mask.any(dim=0)

        # Log for debugging
        if LOGGER.isEnabledFor(10) and frame_idx is not None:
            suppress_i_mask = suppress_i_mask.cpu().numpy()
            suppress_j_mask = suppress_j_mask.cpu().numpy()
            last_occluded = last_occluded.cpu().numpy()

            # Find all suppression pairs without using torch.where
            batch_size = suppress_i_mask.shape[0]

            # Log i-suppression cases (where i gets suppressed in favor of j)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_i_mask[i, j]:
                        LOGGER.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[i]} last occluded {last_occluded[i]} in favor of {obj_ids[j]} last occluded {last_occluded[j]}"
                        )

            # Log j-suppression cases (where j gets suppressed in favor of i)
            for i in range(batch_size):
                for j in range(batch_size):
                    if suppress_j_mask[i, j]:
                        LOGGER.debug(
                            f"{frame_idx=}: Suppressing obj {obj_ids[j]} last occluded {last_occluded[j]} in favor of {obj_ids[i]} last occluded {last_occluded[i]}"
                        )

        return to_suppress

    def _propogate_tracker_one_frame_local_gpu(self, inference_states: list[Any], frame_idx: int):
        """Inference_states: list of inference states, each state corresponds to a different set of objects."""
        obj_ids_local = []
        low_res_masks_list = []
        obj_scores_list = []
        for inference_state in inference_states:
            if len(inference_state["obj_ids"]) == 0:
                continue  # skip propagation on empty inference states

            out_obj_ids, out_low_res_masks, out_obj_scores = self.tracker.propagate_in_video(
                inference_state, frame_idx=frame_idx
            )
            assert isinstance(out_obj_ids, list)
            obj_ids_local.extend(out_obj_ids)
            low_res_masks_list.append(out_low_res_masks.squeeze(1))
            obj_scores_list.append(out_obj_scores.squeeze(1))

        # concatenate the output masklets from all local inference states
        if len(low_res_masks_list) > 0:
            low_res_masks_local = torch.cat(low_res_masks_list, dim=0)
            obj_scores_local = torch.cat(obj_scores_list, dim=0)
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(0, *self._bb_feat_sizes[0], device=self.device)
            obj_scores_local = torch.zeros(0, device=self.device)

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: torch.Tensor,
        det_scores_np: np.ndarray,
        trk_masks: torch.Tensor,
        trk_obj_ids: np.ndarray,
    ):
        """Match detections on the current frame with the existing masklets.

        Args:
            det_masks: (N, H, W) tensor of predicted masks
            det_scores_np: (N,) array of detection scores
            trk_masks: (M, H, W) tensor of track masks
            trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

        Returns:
            new_det_fa_inds: array of new object indices.
            unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched to any detections on this
                frame (for unmatched, we only count masklets with >0 area)
            det_to_matched_trk_obj_ids: dict[int, np.ndarray]: mapping from detector's detection indices to the list of
                matched tracklet object IDs
            empty_trk_obj_ids: array of existing masklet object IDs with zero area in SAM2 prediction
        """
        iou_threshold = self.assoc_iou_thresh
        iou_threshold_trk = self.trk_assoc_iou_thresh
        new_det_thresh = self.new_det_thresh

        assert det_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.is_floating_point(), "float tensor expected (do not binarize)"
        assert trk_masks.size(0) == len(trk_obj_ids), (
            f"trk_masks and trk_obj_ids should have the same length, {trk_masks.size(0)} vs {len(trk_obj_ids)}"
        )
        if trk_masks.size(0) == 0:
            # all detections are new
            new_det_fa_inds = np.arange(det_masks.size(0))
            unmatched_trk_obj_ids = np.array([], np.int64)
            empty_trk_obj_ids = np.array([], np.int64)
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )
        elif det_masks.size(0) == 0:
            # all previous tracklets are unmatched if they have a non-zero area
            new_det_fa_inds = np.array([], np.int64)
            trk_is_nonempty = (trk_masks > 0).any(dim=(1, 2)).cpu().numpy()
            unmatched_trk_obj_ids = trk_obj_ids[trk_is_nonempty]
            empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]
            det_to_matched_trk_obj_ids = {}
            trk_id_to_max_iou_high_conf_det = {}
            return (
                new_det_fa_inds,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                trk_id_to_max_iou_high_conf_det,
                empty_trk_obj_ids,
            )

        if det_masks.shape[-2:] != trk_masks.shape[-2:]:
            # resize to the smaller size to save GPU memory
            if np.prod(det_masks.shape[-2:]) < np.prod(trk_masks.shape[-2:]):
                trk_masks = F.interpolate(
                    trk_masks.unsqueeze(1),
                    size=det_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                # resize detections to track size
                det_masks = F.interpolate(
                    det_masks.unsqueeze(1),
                    size=trk_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)

        det_masks_binary = det_masks > 0
        trk_masks_binary = trk_masks > 0
        ious = mask_iou(det_masks_binary.flatten(1).float(), trk_masks_binary.flatten(1).float())  # (N, M)

        ious_np = ious.cpu().numpy()
        if self.o2o_matching_masklets_enable:
            from scipy.optimize import linear_sum_assignment

            # Hungarian matching for tracks (one-to-one: each track matches at most one detection)
            cost_matrix = 1 - ious_np  # Hungarian solves for minimum cost
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            trk_is_matched = np.zeros(trk_masks.size(0), dtype=bool)
            for d, t in zip(row_ind, col_ind):
                if ious_np[d, t] >= iou_threshold_trk:
                    trk_is_matched[t] = True
        else:
            trk_is_matched = (ious_np >= iou_threshold_trk).any(axis=0)
        # Non-empty tracks not matched by Hungarian assignment above threshold are unmatched
        trk_is_nonempty = trk_masks_binary.any(dim=(1, 2)).cpu().numpy()
        trk_is_unmatched = np.logical_and(trk_is_nonempty, ~trk_is_matched)
        unmatched_trk_obj_ids = trk_obj_ids[trk_is_unmatched]
        # also record masklets that have zero area in SAM 2 prediction
        empty_trk_obj_ids = trk_obj_ids[~trk_is_nonempty]

        # For detections: allow many tracks to match to the same detection (many-to-one)
        # So, a detection is 'new' if it does not match any track above threshold
        is_new_det = np.logical_and(
            det_scores_np >= new_det_thresh,
            np.logical_not(np.any(ious_np >= iou_threshold, axis=1)),
        )
        new_det_fa_inds = np.nonzero(is_new_det)[0]

        # for each detection, which tracks it matched to (above threshold)
        det_to_matched_trk_obj_ids = {}
        trk_id_to_max_iou_high_conf_det = {}  # trk id --> exactly one detection idx
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= self.HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= self.HIGH_IOU_THRESH
        det_is_high_conf_and_iou = set(np.nonzero(det_is_high_conf & det_is_high_iou)[0])
        for d in range(det_masks.size(0)):
            det_to_matched_trk_obj_ids[d] = trk_obj_ids[ious_np[d, :] >= iou_threshold]
            if d in det_is_high_conf_and_iou:
                trk_obj_id = trk_obj_ids[det_to_max_iou_trk_idx[d]].item()
                trk_id_to_max_iou_high_conf_det[trk_obj_id] = d

        return (
            new_det_fa_inds,
            unmatched_trk_obj_ids,
            det_to_matched_trk_obj_ids,
            trk_id_to_max_iou_high_conf_det,
            empty_trk_obj_ids,
        )

    def _process_hotstart(
        self,
        frame_idx: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
        empty_trk_obj_ids: np.ndarray,
        unmatched_trk_obj_ids: np.ndarray,
        metadata: dict[str, Any],
    ):
        """Handle hotstart heuristics to remove unmatched or duplicated objects."""
        # obj_id --> first frame index where the object was detected
        obj_first_frame_idx = metadata["obj_first_frame_idx"]
        # obj_id --> [mismatched frame indices]
        unmatched_frame_inds = metadata["unmatched_frame_inds"]
        trk_keep_alive = metadata["trk_keep_alive"]
        # (first_appear_obj_id, obj_id) --> [overlap frame indices]
        overlap_pair_to_frame_inds = metadata["overlap_pair_to_frame_inds"]
        # removed_obj_ids: object IDs that are suppressed via hot-start
        removed_obj_ids = metadata["removed_obj_ids"]

        obj_ids_newly_removed = set()  # object IDs to be newly removed on this frame
        hotstart_diff = frame_idx - self.hotstart_delay if not reverse else frame_idx + self.hotstart_delay

        # Step 1: log the frame index where each object ID first appears
        for obj_id in new_det_obj_ids:
            if obj_id not in obj_first_frame_idx:
                obj_first_frame_idx[obj_id] = frame_idx
            assert obj_id not in trk_keep_alive
            trk_keep_alive[obj_id] = self.init_trk_keep_alive

        matched_trks = set()
        # We use the det-->tracks list to check for matched objects. Otherwise, we need to compute areas to decide whether they're occluded
        for matched_trks_per_det in det_to_matched_trk_obj_ids.values():
            matched_trks.update(matched_trks_per_det)
        for obj_id in matched_trks:
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the max value of trk_keep_alive
            trk_keep_alive[obj_id] = min(self.max_trk_keep_alive, trk_keep_alive[obj_id] + 1)
        for obj_id in unmatched_trk_obj_ids:
            unmatched_frame_inds[obj_id].append(frame_idx)
            # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
            # The max keep alive is 2x the min, means the model prefers to keep the prediction rather than suppress it if it was matched long enough.
            trk_keep_alive[obj_id] = max(self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1)
        if self.decrease_trk_keep_alive_for_empty_masklets:
            for obj_id in empty_trk_obj_ids:
                # NOTE: To minimize number of configurable params, we use the hotstart_unmatch_thresh to set the min value of trk_keep_alive
                trk_keep_alive[obj_id] = max(self.min_trk_keep_alive, trk_keep_alive[obj_id] - 1)

        # Step 2: removed tracks that has not matched with detections for `hotstart_unmatch_thresh` frames with hotstart period
        # a) add unmatched frame indices for each existing object ID
        # note that `unmatched_trk_obj_ids` contains those frames where the SAM2 output mask
        # doesn't match any detection; it excludes those frames where SAM2 gives an empty mask
        # b) remove a masklet if it first appears after `hotstart_diff` and is unmatched for more
        # than `self.hotstart_unmatch_thresh` frames
        for obj_id, frame_indices in unmatched_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if len(frame_indices) >= self.hotstart_unmatch_thresh:
                is_within_hotstart = (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                    obj_first_frame_idx[obj_id] < hotstart_diff and reverse
                )
                if is_within_hotstart:
                    obj_ids_newly_removed.add(obj_id)
                    LOGGER.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it is unmatched for frames: {frame_indices}"
                    )
            if (
                trk_keep_alive[obj_id] <= 0  # Object has not been matched for too long
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                LOGGER.debug(f"Removing object {obj_id} at frame {frame_idx}, due to being unmatched")
                # directly removed the object instead of suppressing it
                obj_ids_newly_removed.add(obj_id)

        # Step 3: removed tracks that overlaps with another track for `hotstart_dup_thresh` frames
        # a) find overlaps tracks -- we consider overlap if they match to the same detection
        for _, matched_trk_obj_ids in det_to_matched_trk_obj_ids.items():
            if len(matched_trk_obj_ids) < 2:
                continue  # only count detections that are matched to multiple (>=2) masklets
            # if there are multiple matched track ids, we need to find the one that appeared first;
            # these later appearing ids may be removed since they may be considered as duplicates
            first_appear_obj_id = (
                min(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
                if not reverse
                else max(matched_trk_obj_ids, key=lambda x: obj_first_frame_idx[x])
            )
            for obj_id in matched_trk_obj_ids:
                if obj_id != first_appear_obj_id:
                    key = (first_appear_obj_id, obj_id)
                    overlap_pair_to_frame_inds[key].append(frame_idx)

        # b) remove a masklet if it first appears after `hotstart_diff` and it overlaps with another
        # masklet (that appears earlier) for more than `self.hotstart_dup_thresh` frames
        for (first_obj_id, obj_id), frame_indices in overlap_pair_to_frame_inds.items():
            if obj_id in removed_obj_ids or obj_id in obj_ids_newly_removed:
                continue  # skip if the object is already removed
            if (obj_first_frame_idx[obj_id] > hotstart_diff and not reverse) or (
                obj_first_frame_idx[obj_id] < hotstart_diff and reverse
            ):
                if len(frame_indices) >= self.hotstart_dup_thresh:
                    obj_ids_newly_removed.add(obj_id)
                    LOGGER.debug(
                        f"Removing object {obj_id} at frame {frame_idx} "
                        f"since it overlaps with another track {first_obj_id} at frames: {frame_indices}"
                    )

        removed_obj_ids.update(obj_ids_newly_removed)
        return obj_ids_newly_removed, metadata

    def _tracker_update_memories(
        self, tracker_inference_states: list[Any], frame_idx: int, low_res_masks: torch.Tensor
    ):
        """Run Sam2 memory encoder, enforcing non-overlapping constraints globally."""
        if len(tracker_inference_states) == 0:
            return
        # NOTE: inspect this part if we observe OOMs in the demo
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        )
        # We first apply non-overlapping constraints before memory encoding. This may include some suppression heuristics.
        if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
            high_res_masks = self.tracker.model._suppress_object_pw_area_shrinkage(high_res_masks)
        # Instead of gathering the predicted object scores, we use mask areas as a proxy.
        object_score_logits = torch.where((high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        # Run the memory encoder on local slices for each GPU
        start_idx_gpu = 0
        start_idx_state = start_idx_gpu
        for tracker_state in tracker_inference_states:
            num_obj_per_state = len(tracker_state["obj_ids"])
            if num_obj_per_state == 0:
                continue
            # Get the local high-res masks and object score logits for this inference state
            end_idx_state = start_idx_state + num_obj_per_state
            local_high_res_masks = high_res_masks[start_idx_state:end_idx_state]
            local_object_score_logits = object_score_logits[start_idx_state:end_idx_state]
            local_batch_size = local_high_res_masks.size(0)
            # Run Sam2 memory encoder. Note that we do not re-enforce the non-overlapping constraint as it is turned off by default

            encoded_mem = self.tracker._run_memory_encoder(
                local_batch_size,
                local_high_res_masks,
                local_object_score_logits,
                is_mask_from_pts=False,
                inference_state=tracker_state,
            )
            local_maskmem_features, local_maskmem_pos_enc = encoded_mem
            # Store encoded memories in the local inference state
            output_dict = tracker_state["output_dict"]
            for storage_key in ["cond_frame_outputs", "non_cond_frame_outputs"]:
                if frame_idx not in output_dict[storage_key]:
                    continue
                output_dict[storage_key][frame_idx]["maskmem_features"] = local_maskmem_features
                output_dict[storage_key][frame_idx]["maskmem_pos_enc"] = [pos for pos in local_maskmem_pos_enc]
                # for batched inference state, we also need to add per-object
                # memory slides to support instance interactivity
                self.tracker._add_output_per_object(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    current_out=output_dict[storage_key][frame_idx],
                    storage_key=storage_key,
                )
            start_idx_state += num_obj_per_state

    def _tracker_add_new_objects(
        self,
        frame_idx: int,
        num_frames: int,
        new_obj_ids: list[int],
        new_obj_masks: torch.Tensor,
        tracker_states_local: list[Any],
    ):
        """Add a new object to SAM2 inference states."""
        prev_tracker_state = tracker_states_local[0] if len(tracker_states_local) > 0 else None

        # prepare inference_state
        # batch objects that first appear on the same frame together
        # Clear inference state. Keep the cached image features if available.
        new_tracker_state = self.tracker._init_state(num_frames=num_frames)
        # NOTE: adding image placeholder
        new_tracker_state["im"] = None
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None) if prev_tracker_state is not None else None
        )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(0),
            size=self.interpol_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        new_obj_masks = new_obj_masks > 0

        # add object one by one
        for new_obj_id, new_mask in zip(new_obj_ids, new_obj_masks):
            self.tracker.add_new_prompts(
                inference_state=new_tracker_state,
                frame_idx=frame_idx,
                obj_id=new_obj_id,
                masks=new_mask[None, None],  # add bs, channel
            )
        # NOTE: we skip enforcing the non-overlapping constraint **globally** when adding new objects.
        self.tracker.propagate_in_video_preflight(new_tracker_state)
        tracker_states_local.append(new_tracker_state)
        return tracker_states_local

    def _tracker_remove_objects(self, tracker_states_local: list[Any], obj_ids: list[int]):
        """Remove an object from SAM2 inference states. This would remove the object from all frames in the video."""
        if not obj_ids:
            return
        # Filter out states that become empty after removal
        active_states = []
        for state in tracker_states_local:
            for obj_id in obj_ids:
                # we try to remove `obj_id` on every inference state with `strict=False`
                # it will not do anything if an inference state doesn't contain `obj_id`
                self.tracker.remove_object(state, obj_id, strict=False)

            if len(state["obj_ids"]) > 0:
                active_states.append(state)

        # Update the list in-place
        tracker_states_local[:] = active_states

    def _initialize_metadata(self):
        """Initialize metadata for the masklets."""
        tracker_metadata = {
            "obj_ids": np.array([], np.int32),
            "num_obj": np.zeros(1, np.int32),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_cls": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        # "metadata" contains metadata that is only stored on (and accessible to) GPU 0
        # - obj_first_frame_idx: obj_id --> first frame index where the object was detected
        # - unmatched_frame_inds: obj_id --> [mismatched frame indices]
        # - overlap_pair_to_frame_inds: (first_appear_obj_id, obj_id) --> [overlap frame indices]
        # - removed_obj_ids: object IDs that are suppressed via hot-start
        metadata = {
            "obj_first_frame_idx": {},
            "unmatched_frame_inds": defaultdict(list),
            "trk_keep_alive": defaultdict(int),  # This is used only for object suppression not for removal
            "overlap_pair_to_frame_inds": defaultdict(list),
            "removed_obj_ids": set(),
        }
        if self.masklet_confirmation_enable:
            # all the following are np.ndarray with the same shape as `obj_ids_all_gpu`
            metadata["masklet_confirmation"] = {
                # "status" is the confirmation status of each masklet
                "status": np.array([], np.int64),
                # "consecutive_det_num" is the number of consecutive frames where the masklet is
                # detected by the detector (with a matched detection)
                "consecutive_det_num": np.array([], np.int64),
            }
        tracker_metadata["metadata"] = metadata

        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        metadata: dict[str, Any],
        obj_ids_all_gpu_prev: np.ndarray,
        obj_ids_all_gpu_updated: np.ndarray,
        det_to_matched_trk_obj_ids: dict[int, np.ndarray],
        new_det_obj_ids: np.ndarray,
    ):
        """Update the confirmation status of masklets based on the current frame's detection results."""
        confirmation_data = metadata["masklet_confirmation"]

        # a) first, expand "confirmation_data" to include new masklets added in this frame
        status_prev = confirmation_data["status"]
        consecutive_det_num_prev = confirmation_data["consecutive_det_num"]
        assert status_prev.shape == obj_ids_all_gpu_prev.shape, (
            f"Got {status_prev.shape} vs {obj_ids_all_gpu_prev.shape}"
        )

        obj_id_to_updated_idx = {obj_id: idx for idx, obj_id in enumerate(obj_ids_all_gpu_updated)}
        prev_elem_is_in_updated = np.isin(obj_ids_all_gpu_prev, obj_ids_all_gpu_updated)
        prev_elem_obj_ids_in_updated = obj_ids_all_gpu_prev[prev_elem_is_in_updated]
        prev_elem_inds_in_updated = np.array(
            [obj_id_to_updated_idx[obj_id] for obj_id in prev_elem_obj_ids_in_updated],
            dtype=np.int64,
        )
        # newly added masklets are initialized to "UNCONFIRMED" status
        unconfirmed_val = self.UNCONFIRMED
        status = np.full_like(obj_ids_all_gpu_updated, fill_value=unconfirmed_val)
        status[prev_elem_inds_in_updated] = status_prev[prev_elem_is_in_updated]
        consecutive_det_num = np.zeros_like(obj_ids_all_gpu_updated)
        consecutive_det_num[prev_elem_inds_in_updated] = consecutive_det_num_prev[prev_elem_is_in_updated]

        # b) update the confirmation status of all masklets based on the current frame
        # b.1) update "consecutive_det_num"
        # "is_matched": whether a masklet is matched to a detection on this frame
        is_matched = np.isin(obj_ids_all_gpu_updated, new_det_obj_ids)
        for matched_trk_obj_ids in det_to_matched_trk_obj_ids.values():
            is_matched |= np.isin(obj_ids_all_gpu_updated, matched_trk_obj_ids)
        consecutive_det_num = np.where(is_matched, consecutive_det_num + 1, 0)

        # b.2) update "status"
        change_to_confirmed = consecutive_det_num >= self.masklet_confirmation_consecutive_det_thresh
        status[change_to_confirmed] = self.CONFIRMED

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return metadata

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            LOGGER.warning(f"Loaded ckpt with {missing_keys=}, {unexpected_keys=}")
        else:
            LOGGER.info("Loaded ckpt successfully without missing or unexpected keys")

    def _encode_prompt(self, **kwargs):
        return self.model._encode_prompt(**kwargs)

    @staticmethod
    def _drop_new_det_with_obj_limit(new_det_fa_inds, det_scores_np, num_to_keep):
        """Drop a few new detections based on the maximum number of objects. We drop new objects based on their
        detection scores, keeping the high-scoring ones and dropping the low-scoring ones.
        """
        assert 0 <= num_to_keep <= len(new_det_fa_inds)
        if num_to_keep == 0:
            return np.array([], np.int64)  # keep none
        if num_to_keep == len(new_det_fa_inds):
            return new_det_fa_inds  # keep all

        # keep the top-scoring detections
        score_order = np.argsort(det_scores_np[new_det_fa_inds])[::-1]
        new_det_fa_inds = new_det_fa_inds[score_order[:num_to_keep]]
        return new_det_fa_inds
