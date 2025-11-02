# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Generate predictions using the Segment Anything Model (SAM).

SAM is an advanced image segmentation model offering features like promptable segmentation and zero-shot performance.
This module contains the implementation of the prediction logic and auxiliary utilities required to perform segmentation
using SAM. It forms an integral part of the Ultralytics framework and is designed for high-performance, real-time image
segmentation tasks.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
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

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize the Predictor with configuration, overrides, and callbacks.

        Sets up the Predictor object for SAM (Segment Anything Model) and applies any configuration overrides or
        callbacks provided. Initializes task-specific settings for SAM, such as retina_masks being set to True for
        optimal results.

        Args:
            cfg (dict): Configuration dictionary containing default settings.
            overrides (dict | None): Dictionary of values to override default configuration.
            _callbacks (dict | None): Dictionary of callback functions to customize behavior.

        Examples:
            >>> predictor_example = Predictor(cfg=DEFAULT_CFG)
            >>> predictor_example_with_imgsz = Predictor(overrides={"imgsz": 640})
            >>> predictor_example_with_callback = Predictor(_callbacks={"on_predict_start": custom_callback})
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
        torch.Tensor and list of np.ndarray as input formats.

        Args:
            im (torch.Tensor | list[np.ndarray]): Input image(s) in BCHW tensor format or list of HWC numpy arrays.

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
        letterbox = LetterBox(self.args.imgsz, auto=False, center=False)
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
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

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

    def setup_source(self, source):
        """Set up the data source for inference.

        This method configures the data source from which images will be fetched for inference. It supports various
        input types such as image files, directories, video files, and other compatible data sources.

        Args:
            source (str | Path | None): The path or identifier for the image data source. Can be a file path, directory
                path, URL, or other supported source types.

        Examples:
            >>> predictor = Predictor()
            >>> predictor.setup_source("path/to/images")
            >>> predictor.setup_source("video.mp4")
            >>> predictor.setup_source(None)  # Uses default source if available

        Notes:
            - If source is None, the method may use a default source if configured.
            - The method adapts to different source types and prepares them for subsequent inference steps.
            - Supported source types may include local files, directories, URLs, and video streams.
        """
        if source is not None:
            super().setup_source(source)

    def set_image(self, image):
        """Preprocess and set a single image for inference.

        This method prepares the model for inference on a single image by setting up the model if not already
        initialized, configuring the data source, and preprocessing the image for feature extraction. It ensures that
        only one image is set at a time and extracts image features for subsequent use.

        Args:
            image (str | np.ndarray): Path to the image file as a string, or a numpy array representing an image read by
                cv2.

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

    def get_im_features(self, im):
        """Extract image features using the SAM model's image encoder for subsequent mask prediction."""
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM models only support square image size, but got {self.imgsz}."
        )
        self.model.set_imgsz(self.imgsz)
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

    def set_image(self, image):
        """Preprocess and set a single image for inference using the SAM2 model.

        This method initializes the model if not already done, configures the data source to the specified image, and
        preprocesses the image for feature extraction. It supports setting only one image at a time.

        Args:
            image (str | np.ndarray): Path to the image file as a string, or a numpy array representing the image.

        Raises:
            AssertionError: If more than one image is attempted to be set.

        Examples:
            >>> predictor = SAM2Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(np.array([...]))  # Using a numpy array

        Notes:
            - This method must be called before performing any inference on a new image.
            - The method caches the extracted features for efficient subsequent inferences on the same image.
            - Only one image can be set at a time. To process multiple images, call this method for each new image.
        """
        if self.model is None:
            self.setup_model(model=None)
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def get_im_features(self, im):
        """Extract image features from the SAM image encoder for subsequent processing."""
        assert isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1], (
            f"SAM 2 models only support square image size, but got {self.imgsz}."
        )
        self.model.set_imgsz(self.imgsz)
        self._bb_feat_sizes = [[x // (4 * i) for x in self.imgsz] for i in [1, 2, 4]]

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

        Examples:
            >>> predictor = SAM2VideoPredictor(cfg=DEFAULT_CFG)
            >>> predictor_example_with_imgsz = SAM2VideoPredictor(overrides={"imgsz": 640})
            >>> predictor_example_with_callback = SAM2VideoPredictor(_callbacks={"on_predict_start": custom_callback})
        """
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}
        self.non_overlap_masks = True
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.callbacks["on_predict_start"].append(self.init_state)

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
        assert (masks is None) ^ (points is None), "'masks' and 'points' prompts are not compatible with each other."
        obj_idx = self._obj_id_to_idx(obj_id)

        point_inputs = None
        pop_key = "point_inputs_per_obj"
        if points is not None:
            point_inputs = {"point_coords": points, "point_labels": labels}
            self.inference_state["point_inputs_per_obj"][obj_idx][frame_idx] = point_inputs
            pop_key = "mask_inputs_per_obj"
        self.inference_state["mask_inputs_per_obj"][obj_idx][frame_idx] = masks
        self.inference_state[pop_key][obj_idx].pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in self.inference_state["frames_already_tracked"]
        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
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
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
        )
        pred_masks = consolidated_out["pred_masks"].flatten(0, 1)
        return pred_masks.flatten(0, 1), torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device)

    @smart_inference_mode()
    def propagate_in_video_preflight(self):
        """Prepare inference_state and consolidate temporary outputs before tracking.

        This method marks the start of tracking, disallowing the addition of new objects until the session is reset. It
        consolidates temporary outputs from `temp_output_dict_per_obj` and merges them into `output_dict`. Additionally,
        it clears non-conditioning memory around input frames and ensures that the state is consistent with the provided
        inputs.
        """
        # Tracking has started and we don't allow adding new objects until session is reset.
        self.inference_state["tracking_has_started"] = True
        batch_size = len(self.inference_state["obj_idx_to_id"])

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = self.inference_state["temp_output_dict_per_obj"]
        output_dict = self.inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
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
                    frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(frame_idx, consolidated_out, storage_key)
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
        for obj_output_dict in self.inference_state["output_dict_per_obj"].values():
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
        for point_inputs_per_frame in self.inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in self.inference_state["mask_inputs_per_obj"].values():
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

        inference_state = {
            "num_frames": predictor.dataset.frames,
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
        predictor.inference_state = inference_state

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
        self.model.set_imgsz(self.imgsz)
        backbone_out = self.model.forward_image(im)
        if batch > 1:  # expand features if there's more than one prompt
            for i, feat in enumerate(backbone_out["backbone_fpn"]):
                backbone_out["backbone_fpn"][i] = feat.expand(batch, -1, -1, -1)
            for i, pos in enumerate(backbone_out["vision_pos_enc"]):
                pos = pos.expand(batch, -1, -1, -1)
                backbone_out["vision_pos_enc"][i] = pos
        _, vis_feats, vis_pos_embed, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        return vis_feats, vis_pos_embed, feat_sizes

    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index.

        Args:
            obj_id (int): The unique identifier of the object provided by the client side.

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
        obj_idx = self.inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not self.inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(self.inference_state["obj_id_to_idx"])
            self.inference_state["obj_id_to_idx"][obj_id] = obj_idx
            self.inference_state["obj_idx_to_id"][obj_idx] = obj_id
            self.inference_state["obj_ids"] = list(self.inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            self.inference_state["point_inputs_per_obj"][obj_idx] = {}
            self.inference_state["mask_inputs_per_obj"][obj_idx] = {}
            self.inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            self.inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {self.inference_state['obj_ids']}. "
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
        # Retrieve correct image features
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(
            self.inference_state["im"], batch_size
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
            num_frames=self.inference_state["num_frames"],
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
        current_out["maskmem_pos_enc"] = self._get_maskmem_pos_enc(current_out["maskmem_pos_enc"])
        return current_out

    def _get_maskmem_pos_enc(self, out_maskmem_pos_enc):
        """Cache and manage the positional encoding for mask memory across frames and objects.

        This method optimizes storage by caching the positional encoding (`maskmem_pos_enc`) for mask memory, which is
        constant across frames and objects, thus reducing the amount of redundant information stored during an inference
        session. It checks if the positional encoding has already been cached; if not, it caches a slice of the provided
        encoding. If the batch size is greater than one, it expands the cached positional encoding to match the current
        batch size.

        Args:
            out_maskmem_pos_enc (list[torch.Tensor] | None): The positional encoding for mask memory. Should be a list
                of tensors or None.

        Returns:
            (list[torch.Tensor]): The positional encoding for mask memory, either cached or expanded.

        Notes:
            - The method assumes that `out_maskmem_pos_enc` is a list of tensors or None.
            - Only a single object's slice is cached since the encoding is the same across objects.
            - The method checks if the positional encoding has already been cached in the session's constants.
            - If the batch size is greater than one, the cached encoding is expanded to fit the batch size.
        """
        model_constants = self.inference_state["constants"]
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

        Returns:
            (dict): A consolidated output dictionary containing the combined results for all objects.

        Notes:
            - The method initializes the consolidated output with placeholder values for missing objects.
            - It searches for outputs in both the temporary and main output dictionaries.
            - If `run_mem_encoder` is True, it applies non-overlapping constraints and re-runs the memory encoder.
            - The `maskmem_features` and `maskmem_pos_enc` are only populated when `run_mem_encoder` is True.
        """
        batch_size = len(self.inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            "pred_masks": torch.full(
                size=(batch_size, 1, self.imgsz[0] // 4, self.imgsz[1] // 4),
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
            obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
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
            )

        return consolidated_out

    def _get_empty_mask_ptr(self, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame.

        Args:
            frame_idx (int): The index of the current frame for which to generate the dummy object pointer.

        Returns:
            (torch.Tensor): A tensor representing the dummy object pointer generated from the empty mask.
        """
        # Retrieve correct image features
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(self.inference_state["im"])

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
            num_frames=self.inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def _run_memory_encoder(self, batch_size, high_res_masks, object_score_logits, is_mask_from_pts):
        """Run the memory encoder on masks.

        This is usually after applying non-overlapping constraints to object scores. Since their scores changed, their
        memory also needs to be computed again with the memory encoder.

        Args:
            batch_size (int): The batch size for processing the frame.
            high_res_masks (torch.Tensor): High-resolution masks for which to compute the memory.
            object_score_logits (torch.Tensor): Logits representing the object scores.
            is_mask_from_pts (bool): Indicates if the mask is derived from point interactions.

        Returns:
            maskmem_features (torch.Tensor): The encoded mask features.
            maskmem_pos_enc (torch.Tensor): The positional encoding.
        """
        # Retrieve correct image features
        current_vision_feats, _, feat_sizes = self.get_im_features(self.inference_state["im"], batch_size)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
            object_score_logits=object_score_logits,
        )

        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(maskmem_pos_enc)
        return maskmem_features.to(
            dtype=torch.float16, device=self.device, non_blocking=self.device.type == "cuda"
        ), maskmem_pos_enc

    def _add_output_per_object(self, frame_idx, current_out, storage_key):
        """Split a multi-object output into per-object output slices and add them into Output_Dict_Per_Obj.

        The resulting slices share the same tensor storage.

        Args:
            frame_idx (int): The index of the current frame.
            current_out (dict): The current output dictionary containing multi-object outputs.
            storage_key (str): The key used to store the output in the per-object output dictionary.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        for obj_idx, obj_output_dict in self.inference_state["output_dict_per_obj"].items():
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

    def _clear_non_cond_mem_around_input(self, frame_idx):
        """Remove the non-conditioning memory around the input frame.

        When users provide correction clicks, the surrounding frames' non-conditioning memories can still contain
        outdated object appearance information and could confuse the model. This method clears those non-conditioning
        memories surrounding the interacted frame to avoid giving the model both old and new information about the
        object.

        Args:
            frame_idx (int): The index of the current frame where user interaction occurred.
        """
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        for t in range(frame_idx_begin, frame_idx_end + 1):
            self.inference_state["output_dict"]["non_cond_frame_outputs"].pop(t, None)
            for obj_output_dict in self.inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)


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

        Examples:
            >>> predictor = SAM2DynamicInteractivePredictor(cfg=DEFAULT_CFG)
            >>> predictor_example_with_imgsz = SAM2DynamicInteractivePredictor(overrides={"imgsz": 640})
            >>> predictor_example_with_callback = SAM2DynamicInteractivePredictor(
            ...     _callbacks={"on_predict_start": custom_callback}
            ... )
        """
        super().__init__(cfg, overrides, _callbacks)
        self.non_overlap_masks = True

        # Initialize the memory bank to store image states
        # NOTE: probably need to use dict for better query
        self.memory_bank = []

        # Initialize the object index set and mappings
        self.obj_idx_set = set()
        self.obj_id_to_idx = OrderedDict()
        self.obj_idx_to_id = OrderedDict()
        self._max_obj_num = max_obj_num
        for i in range(self._max_obj_num):
            self.obj_id_to_idx[i + 1] = i
            self.obj_idx_to_id[i] = i + 1

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
        """Prepare the memory-conditioned features for the current image state. If obj_idx is provided, it supposes to
        prepare features for a specific prompted object in the image. If obj_idx is None, it prepares features
        for all objects in the image. If there is no memory, it will directly add a no-memory embedding to the
        current vision features. If there is memory, it will use the memory features from previous frames to
        condition the current vision features using a transformer attention mechanism.

        Args:
            obj_idx (int | None): The index of the object for which to prepare the features.

        Returns:
            pix_feat_with_mem (torch.Tensor): The memory-conditioned pixel features.
        """
        if len(self.memory_bank) == 0 or isinstance(obj_idx, int):
            # for initial conditioning frames with, encode them without using any previous memory
            # directly add no-mem embedding (instead of using the transformer encoder)
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
        # reshape the output (HW)BC => BCHW
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
            # fused the visual feature with previous memory features in the memory bank
            pix_feat_with_mem = self._prepare_memory_conditioned_features(obj_idx)
            # calculate the first feature if adding obj_idx exists(means adding prompts)
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
