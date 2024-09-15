# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Generate predictions using the Segment Anything Model (SAM).

SAM is an advanced image segmentation model offering features like promptable segmentation and zero-shot performance.
This module contains the implementation of the prediction logic and auxiliary utilities required to perform segmentation
using SAM. It forms an integral part of the Ultralytics framework and is designed for high-performance, real-time image
segmentation tasks.
"""

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.torch_utils import select_device

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
from .build import build_sam


class Predictor(BasePredictor):
    """
    Predictor class for SAM, enabling real-time image segmentation with promptable capabilities.

    This class extends BasePredictor and implements the Segment Anything Model (SAM) for advanced image
    segmentation tasks. It supports various input prompts like points, bounding boxes, and masks for
    fine-grained control over segmentation results.

    Attributes:
        args (SimpleNamespace): Configuration arguments for the predictor.
        model (torch.nn.Module): The loaded SAM model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        im (torch.Tensor): The preprocessed input image.
        features (torch.Tensor): Extracted image features.
        prompts (Dict): Dictionary to store various types of prompts (e.g., bboxes, points, masks).
        segment_all (bool): Flag to indicate if full image segmentation should be performed.
        mean (torch.Tensor): Mean values for image normalization.
        std (torch.Tensor): Standard deviation values for image normalization.

    Methods:
        preprocess: Prepares input images for model inference.
        pre_transform: Performs initial transformations on the input image.
        inference: Performs segmentation inference based on input prompts.
        prompt_inference: Internal function for prompt-based segmentation inference.
        generate: Generates segmentation masks for an entire image.
        setup_model: Initializes the SAM model for inference.
        get_model: Builds and returns a SAM model.
        postprocess: Post-processes model outputs to generate final results.
        setup_source: Sets up the data source for inference.
        set_image: Sets and preprocesses a single image for inference.
        get_im_features: Extracts image features using the SAM image encoder.
        set_prompts: Sets prompts for subsequent inference.
        reset_image: Resets the current image and its features.
        remove_small_regions: Removes small disconnected regions and holes from masks.

    Examples:
        >>> predictor = Predictor()
        >>> predictor.setup_model(model_path="sam_model.pt")
        >>> predictor.set_image("image.jpg")
        >>> masks, scores, boxes = predictor.generate()
        >>> results = predictor.postprocess((masks, scores, boxes), im, orig_img)
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the Predictor with configuration, overrides, and callbacks.

        Sets up the Predictor object for SAM (Segment Anything Model) and applies any configuration overrides or
        callbacks provided. Initializes task-specific settings for SAM, such as retina_masks being set to True
        for optimal results.

        Args:
            cfg (Dict): Configuration dictionary containing default settings.
            overrides (Dict | None): Dictionary of values to override default configuration.
            _callbacks (Dict | None): Dictionary of callback functions to customize behavior.

        Examples:
            >>> predictor = Predictor(cfg=DEFAULT_CFG)
            >>> predictor = Predictor(overrides={"imgsz": 640})
            >>> predictor = Predictor(_callbacks={"on_predict_start": custom_callback})
        """
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict"))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess(self, im):
        """
        Preprocess the input image for model inference.

        This method prepares the input image by applying transformations and normalization. It supports both
        torch.Tensor and list of np.ndarray as input formats.

        Args:
            im (torch.Tensor | List[np.ndarray]): Input image(s) in BCHW tensor format or list of HWC numpy arrays.

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
        im = im.half() if self.model.fp16 else im.float()
        if not_tensor:
            im = (im - self.mean) / self.std
        return im

    def pre_transform(self, im):
        """
        Perform initial transformations on the input image for preprocessing.

        This method applies transformations such as resizing to prepare the image for further preprocessing.
        Currently, batched inference is not supported; hence the list length should be 1.

        Args:
            im (List[np.ndarray]): List containing a single image in HWC numpy array format.

        Returns:
            (List[np.ndarray]): List containing the transformed image.

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
        """
        Perform image segmentation inference based on the given input cues, using the currently loaded image.

        This method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt
        encoder, and mask decoder for real-time and promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | List | None): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List | None): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List | None): Labels for point prompts, shape (N,). 1 = foreground, 0 = background.
            masks (np.ndarray | None): Low-resolution masks from previous predictions, shape (N, H, W). For SAM H=W=256.
            multimask_output (bool): Flag to return multiple masks. Helpful for ambiguous prompts.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            (tuple): Contains the following three elements:
                - np.ndarray: The output masks in shape (C, H, W), where C is the number of generated masks.
                - np.ndarray: An array of length C containing quality scores predicted by the model for each mask.
                - np.ndarray: Low-resolution logits of shape (C, H, W) for subsequent inference, where H=W=256.

        Examples:
            >>> predictor = Predictor()
            >>> predictor.setup_model(model_path="sam_model.pt")
            >>> predictor.set_image("image.jpg")
            >>> masks, scores, logits = predictor.inference(im, bboxes=[[0, 0, 100, 100]])
        """
        # Override prompts if any stored in self.prompts
        bboxes = self.prompts.pop("bboxes", bboxes)
        points = self.prompts.pop("points", points)
        masks = self.prompts.pop("masks", masks)

        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(im, *args, **kwargs)

        return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """
        Performs image segmentation inference based on input cues using SAM's specialized architecture.

        This internal function leverages the Segment Anything Model (SAM) for prompt-based, real-time segmentation.
        It processes various input prompts such as bounding boxes, points, and masks to generate segmentation masks.

        Args:
            im (torch.Tensor): Preprocessed input image tensor with shape (N, C, H, W).
            bboxes (np.ndarray | List | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | List | None): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List | None): Point prompt labels with shape (N,). 1 for foreground, 0 for background.
            masks (np.ndarray | None): Low-res masks from previous predictions with shape (N, H, W). For SAM, H=W=256.
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.

        Returns:
            (tuple): Tuple containing:
                - np.ndarray: Output masks with shape (C, H, W), where C is the number of generated masks.
                - np.ndarray: Quality scores predicted by the model for each mask, with length C.
                - np.ndarray: Low-resolution logits with shape (C, H, W) for subsequent inference, where H=W=256.

        Examples:
            >>> predictor = Predictor()
            >>> im = torch.rand(1, 3, 1024, 1024)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> masks, scores, logits = predictor.prompt_inference(im, bboxes=bboxes)
        """
        features = self.get_im_features(im) if self.features is None else self.features

        src_shape, dst_shape = self.batch[1][0].shape[:2], im.shape[2:]
        r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        # Transform input prompts
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            points = points[None] if points.ndim == 1 else points
            # Assuming labels are all positive if users don't pass labels.
            if labels is None:
                labels = np.ones(points.shape[0])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            points *= r
            # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
            points, labels = points[:, None, :], labels[:, None]
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)

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
        """
        Perform image segmentation using the Segment Anything Model (SAM).

        This method segments an entire image into constituent parts by leveraging SAM's advanced architecture
        and real-time performance capabilities. It can optionally work on image crops for finer segmentation.

        Args:
            im (torch.Tensor): Input tensor representing the preprocessed image with shape (N, C, H, W).
            crop_n_layers (int): Number of layers for additional mask predictions on image crops.
            crop_overlap_ratio (float): Overlap between crops, scaled down in subsequent layers.
            crop_downscale_factor (int): Scaling factor for sampled points-per-side in each layer.
            point_grids (List[np.ndarray] | None): Custom grids for point sampling normalized to [0,1].
            points_stride (int): Number of points to sample along each side of the image.
            points_batch_size (int): Batch size for the number of points processed simultaneously.
            conf_thres (float): Confidence threshold [0,1] for filtering based on mask quality prediction.
            stability_score_thresh (float): Stability threshold [0,1] for mask filtering based on stability.
            stability_score_offset (float): Offset value for calculating stability score.
            crop_nms_thresh (float): IoU cutoff for NMS to remove duplicate masks between crops.

        Returns:
            (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A tuple containing:
                - pred_masks (torch.Tensor): Segmented masks with shape (N, H, W).
                - pred_scores (torch.Tensor): Confidence scores for each mask with shape (N,).
                - pred_bboxes (torch.Tensor): Bounding boxes for each mask with shape (N, 4).

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
            region_areas.append(area.expand(len(crop_masks)))

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

    def setup_model(self, model, verbose=True):
        """
        Initializes the Segment Anything Model (SAM) for inference.

        This method sets up the SAM model by allocating it to the appropriate device and initializing the necessary
        parameters for image normalization and other Ultralytics compatibility settings.

        Args:
            model (torch.nn.Module): A pre-trained SAM model. If None, a model will be built based on configuration.
            verbose (bool): If True, prints selected device information.

        Examples:
            >>> predictor = Predictor()
            >>> predictor.setup_model(model=sam_model, verbose=True)
        """
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = self.get_model()
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)

        # Ultralytics compatibility settings
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        self.done_warmup = True

    def get_model(self):
        """Retrieves or builds the Segment Anything Model (SAM) for image segmentation tasks."""
        return build_sam(self.args.model)

    def postprocess(self, preds, img, orig_imgs):
        """
        Post-processes SAM's inference outputs to generate object detection masks and bounding boxes.

        This method scales masks and boxes to the original image size and applies a threshold to the mask
        predictions. It leverages SAM's advanced architecture for real-time, promptable segmentation tasks.

        Args:
            preds (Tuple[torch.Tensor]): The output from SAM model inference, containing:
                - pred_masks (torch.Tensor): Predicted masks with shape (N, 1, H, W).
                - pred_scores (torch.Tensor): Confidence scores for each mask with shape (N, 1).
                - pred_bboxes (torch.Tensor, optional): Predicted bounding boxes if segment_all is True.
            img (torch.Tensor): The processed input image tensor with shape (C, H, W).
            orig_imgs (List[np.ndarray] | torch.Tensor): The original, unprocessed images.

        Returns:
            (List[Results]): List of Results objects containing detection masks, bounding boxes, and other
                metadata for each processed image.

        Examples:
            >>> predictor = Predictor()
            >>> preds = predictor.inference(img)
            >>> results = predictor.postprocess(preds, img, orig_imgs)
        """
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(len(pred_masks))))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if len(masks) == 0:
                masks = None
            else:
                masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
                masks = masks > self.model.mask_threshold  # to bool
                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                # NOTE: SAM models do not return cls info. This `cls` here is just a placeholder for consistency.
                cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # Reset segment-all mode.
        self.segment_all = False
        return results

    def setup_source(self, source):
        """
        Sets up the data source for inference.

        This method configures the data source from which images will be fetched for inference. It supports
        various input types such as image files, directories, video files, and other compatible data sources.

        Args:
            source (str | Path | None): The path or identifier for the image data source. Can be a file path,
                directory path, URL, or other supported source types.

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
        """
        Preprocesses and sets a single image for inference.

        This method prepares the model for inference on a single image by setting up the model if not already
        initialized, configuring the data source, and preprocessing the image for feature extraction. It
        ensures that only one image is set at a time and extracts image features for subsequent use.

        Args:
            image (str | np.ndarray): Path to the image file as a string, or a numpy array representing
                an image read by cv2.

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
            self.setup_model(model=None)
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.get_im_features(im)
            break

    def get_im_features(self, im):
        """Extracts image features using the SAM model's image encoder for subsequent mask prediction."""
        assert (
            isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1]
        ), f"SAM models only support square image size, but got {self.imgsz}."
        self.model.set_imgsz(self.imgsz)
        return self.model.image_encoder(im)

    def set_prompts(self, prompts):
        """Sets prompts for subsequent inference operations."""
        self.prompts = prompts

    def reset_image(self):
        """Resets the current image and its features, clearing them for subsequent inference."""
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """
        Remove small disconnected regions and holes from segmentation masks.

        This function performs post-processing on segmentation masks generated by the Segment Anything Model (SAM).
        It removes small disconnected regions and holes from the input masks, and then performs Non-Maximum
        Suppression (NMS) to eliminate any newly created duplicate boxes.

        Args:
            masks (torch.Tensor): Segmentation masks to be processed, with shape (N, H, W) where N is the number of
                masks, H is height, and W is width.
            min_area (int): Minimum area threshold for removing disconnected regions and holes. Regions smaller than
                this will be removed.
            nms_thresh (float): IoU threshold for the NMS algorithm to remove duplicate boxes.

        Returns:
            (tuple):
                - new_masks (torch.Tensor): Processed masks with small regions removed, shape (N, H, W).
                - keep (List[int]): Indices of remaining masks after NMS, for filtering corresponding boxes.

        Examples:
            >>> masks = torch.rand(5, 640, 640) > 0.5  # 5 random binary masks
            >>> new_masks, keep = remove_small_regions(masks, min_area=100, nms_thresh=0.7)
            >>> print(f"Original masks: {masks.shape}, Processed masks: {new_masks.shape}")
            >>> print(f"Indices of kept masks: {keep}")
        """
        import torchvision  # scope for faster 'import ultralytics'

        if len(masks) == 0:
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


class SAM2Predictor(Predictor):
    """
    SAM2Predictor class for advanced image segmentation using Segment Anything Model 2 architecture.

    This class extends the base Predictor class to implement SAM2-specific functionality for image
    segmentation tasks. It provides methods for model initialization, feature extraction, and
    prompt-based inference.

    Attributes:
        _bb_feat_sizes (List[Tuple[int, int]]): Feature sizes for different backbone levels.
        model (torch.nn.Module): The loaded SAM2 model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        features (Dict[str, torch.Tensor]): Cached image features for efficient inference.
        segment_all (bool): Flag to indicate if all segments should be predicted.
        prompts (Dict): Dictionary to store various types of prompts for inference.

    Methods:
        get_model: Retrieves and initializes the SAM2 model.
        prompt_inference: Performs image segmentation inference based on various prompts.
        set_image: Preprocesses and sets a single image for inference.
        get_im_features: Extracts and processes image features using SAM2's image encoder.

    Examples:
        >>> predictor = SAM2Predictor(cfg)
        >>> predictor.set_image("path/to/image.jpg")
        >>> bboxes = [[100, 100, 200, 200]]
        >>> masks, scores, _ = predictor.prompt_inference(predictor.im, bboxes=bboxes)
        >>> print(f"Predicted {len(masks)} masks with average score {scores.mean():.2f}")
    """

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]

    def get_model(self):
        """Retrieves and initializes the Segment Anything Model 2 (SAM2) for image segmentation tasks."""
        return build_sam(self.args.model)

    def prompt_inference(
        self,
        im,
        bboxes=None,
        points=None,
        labels=None,
        masks=None,
        multimask_output=False,
        img_idx=-1,
    ):
        """
        Performs image segmentation inference based on various prompts using SAM2 architecture.

        This method leverages the Segment Anything Model 2 (SAM2) to generate segmentation masks for input images
        based on provided prompts such as bounding boxes, points, or existing masks. It supports both single and
        multi-object prediction scenarios.

        Args:
            im (torch.Tensor): Preprocessed input image tensor with shape (N, C, H, W).
            bboxes (np.ndarray | List[List[float]] | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | List[List[float]] | None): Object location points with shape (N, 2), in pixels.
            labels (np.ndarray | List[int] | None): Point prompt labels with shape (N,). 1 = foreground, 0 = background.
            masks (np.ndarray | None): Low-resolution masks from previous predictions with shape (N, H, W).
            multimask_output (bool): Flag to return multiple masks for ambiguous prompts.
            img_idx (int): Index of the image in the batch to process.

        Returns:
            (tuple): Tuple containing:
                - np.ndarray: Output masks with shape (C, H, W), where C is the number of generated masks.
                - np.ndarray: Quality scores for each mask, with length C.
                - np.ndarray: Low-resolution logits with shape (C, 256, 256) for subsequent inference.

        Examples:
            >>> predictor = SAM2Predictor(cfg)
            >>> image = torch.rand(1, 3, 640, 640)
            >>> bboxes = [[100, 100, 200, 200]]
            >>> masks, scores, logits = predictor.prompt_inference(image, bboxes=bboxes)
            >>> print(f"Generated {masks.shape[0]} masks with average score {scores.mean():.2f}")

        Notes:
            - The method supports batched inference for multiple objects when points or bboxes are provided.
            - Input prompts (bboxes, points) are automatically scaled to match the input image dimensions.
            - When both bboxes and points are provided, they are merged into a single 'points' input for the model.

        References:
            - SAM2 Paper: [Add link to SAM2 paper when available]
        """
        features = self.get_im_features(im) if self.features is None else self.features

        src_shape, dst_shape = self.batch[1][0].shape[:2], im.shape[2:]
        r = 1.0 if self.segment_all else min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        # Transform input prompts
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            points = points[None] if points.ndim == 1 else points
            # Assuming labels are all positive if users don't pass labels.
            if labels is None:
                labels = torch.ones(points.shape[0])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            points *= r
            # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
            points, labels = points[:, None], labels[:, None]
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes = bboxes.view(-1, 2, 2) * r
            bbox_labels = torch.tensor([[2, 3]], dtype=torch.int32, device=bboxes.device).expand(len(bboxes), -1)
            # NOTE: merge "boxes" and "points" into a single "points" input
            # (where boxes are added at the beginning) to model.sam_prompt_encoder
            if points is not None:
                points = torch.cat([bboxes, points], dim=1)
                labels = torch.cat([bbox_labels, labels], dim=1)
            else:
                points, labels = bboxes, bbox_labels
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        points = (points, labels) if points is not None else None

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=None,
            masks=masks,
        )
        # Predict masks
        batched_mode = points is not None and points[0].shape[0] > 1  # multi object prediction
        high_res_features = [feat_level[img_idx].unsqueeze(0) for feat_level in features["high_res_feats"]]
        pred_masks, pred_scores, _, _ = self.model.sam_mask_decoder(
            image_embeddings=features["image_embed"][img_idx].unsqueeze(0),
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

    def set_image(self, image):
        """
        Preprocesses and sets a single image for inference using the SAM2 model.

        This method initializes the model if not already done, configures the data source to the specified image,
        and preprocesses the image for feature extraction. It supports setting only one image at a time.

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
        """Extracts image features from the SAM image encoder for subsequent processing."""
        assert (
            isinstance(self.imgsz, (tuple, list)) and self.imgsz[0] == self.imgsz[1]
        ), f"SAM 2 models only support square image size, but got {self.imgsz}."
        self.model.set_imgsz(self.imgsz)
        self._bb_feat_sizes = [[x // (4 * i) for x in self.imgsz] for i in [1, 2, 4]]

        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
