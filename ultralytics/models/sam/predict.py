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
import torchvision

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
    Predictor class for the Segment Anything Model (SAM), extending BasePredictor.

    The class provides an interface for model inference tailored to image segmentation tasks.
    With advanced architecture and promptable segmentation capabilities, it facilitates flexible and real-time
    mask generation. The class is capable of working with various types of prompts such as bounding boxes,
    points, and low-resolution masks.

    Attributes:
        cfg (dict): Configuration dictionary specifying model and task-related parameters.
        overrides (dict): Dictionary containing values that override the default configuration.
        _callbacks (dict): Dictionary of user-defined callback functions to augment behavior.
        args (namespace): Namespace to hold command-line arguments or other operational variables.
        im (torch.Tensor): Preprocessed input image tensor.
        features (torch.Tensor): Extracted image features used for inference.
        prompts (dict): Collection of various prompt types, such as bounding boxes and points.
        segment_all (bool): Flag to control whether to segment all objects in the image or only specified ones.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the Predictor with configuration, overrides, and callbacks.

        The method sets up the Predictor object and applies any configuration overrides or callbacks provided. It
        initializes task-specific settings for SAM, such as retina_masks being set to True for optimal results.

        Args:
            cfg (dict): Configuration dictionary.
            overrides (dict, optional): Dictionary of values to override default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to customize behavior.
        """
        if overrides is None:
            overrides = {}
        overrides.update(dict(task="segment", mode="predict", imgsz=1024))
        super().__init__(cfg, overrides, _callbacks)
        self.args.retina_masks = True
        self.im = None
        self.features = None
        self.prompts = {}
        self.segment_all = False

    def preprocess(self, im):
        """
        Preprocess the input image for model inference.

        The method prepares the input image by applying transformations and normalization.
        It supports both torch.Tensor and list of np.ndarray as input formats.

        Args:
            im (torch.Tensor | List[np.ndarray]): BCHW tensor format or list of HWC numpy arrays.

        Returns:
            (torch.Tensor): The preprocessed image tensor.
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

        The method applies transformations such as resizing to prepare the image for further preprocessing.
        Currently, batched inference is not supported; hence the list length should be 1.

        Args:
            im (List[np.ndarray]): List containing images in HWC numpy array format.

        Returns:
            (List[np.ndarray]): List of transformed images.
        """
        assert len(im) == 1, "SAM model does not currently support batched inference"
        letterbox = LetterBox(self.args.imgsz, auto=False, center=False)
        return [letterbox(image=x) for x in im]

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """
        Perform image segmentation inference based on the given input cues, using the currently loaded image. This
        method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt encoder, and
        mask decoder for real-time and promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixel coordinates.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 for foreground and 0 for background.
            masks (np.ndarray, optional): Low-resolution masks from previous predictions. Shape should be (N, H, W). For SAM, H=W=256.
            multimask_output (bool, optional): Flag to return multiple masks. Helpful for ambiguous prompts. Defaults to False.

        Returns:
            (tuple): Contains the following three elements.
                - np.ndarray: The output masks in shape CxHxW, where C is the number of generated masks.
                - np.ndarray: An array of length C containing quality scores predicted by the model for each mask.
                - np.ndarray: Low-resolution logits of shape CxHxW for subsequent inference, where H=W=256.
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
        Internal function for image segmentation inference based on cues like bounding boxes, points, and masks.
        Leverages SAM's specialized architecture for prompt-based, real-time segmentation.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixel coordinates.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 for foreground and 0 for background.
            masks (np.ndarray, optional): Low-resolution masks from previous predictions. Shape should be (N, H, W). For SAM, H=W=256.
            multimask_output (bool, optional): Flag to return multiple masks. Helpful for ambiguous prompts. Defaults to False.

        Returns:
            (tuple): Contains the following three elements.
                - np.ndarray: The output masks in shape CxHxW, where C is the number of generated masks.
                - np.ndarray: An array of length C containing quality scores predicted by the model for each mask.
                - np.ndarray: Low-resolution logits of shape CxHxW for subsequent inference, where H=W=256.
        """
        features = self.model.image_encoder(im) if self.features is None else self.features

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

        This function segments an entire image into constituent parts by leveraging SAM's advanced architecture
        and real-time performance capabilities. It can optionally work on image crops for finer segmentation.

        Args:
            im (torch.Tensor): Input tensor representing the preprocessed image with dimensions (N, C, H, W).
            crop_n_layers (int): Specifies the number of layers for additional mask predictions on image crops.
                                 Each layer produces 2**i_layer number of image crops.
            crop_overlap_ratio (float): Determines the extent of overlap between crops. Scaled down in subsequent layers.
            crop_downscale_factor (int): Scaling factor for the number of sampled points-per-side in each layer.
            point_grids (list[np.ndarray], optional): Custom grids for point sampling normalized to [0,1].
                                                      Used in the nth crop layer.
            points_stride (int, optional): Number of points to sample along each side of the image.
                                           Exclusive with 'point_grids'.
            points_batch_size (int): Batch size for the number of points processed simultaneously.
            conf_thres (float): Confidence threshold [0,1] for filtering based on the model's mask quality prediction.
            stability_score_thresh (float): Stability threshold [0,1] for mask filtering based on mask stability.
            stability_score_offset (float): Offset value for calculating stability score.
            crop_nms_thresh (float): IoU cutoff for Non-Maximum Suppression (NMS) to remove duplicate masks between crops.

        Returns:
            (tuple): A tuple containing segmented masks, confidence scores, and bounding boxes.
        """
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

        Attributes:
            model (torch.nn.Module): The SAM model allocated to the chosen device for inference.
            device (torch.device): The device to which the model and tensors are allocated.
            mean (torch.Tensor): The mean values for image normalization.
            std (torch.Tensor): The standard deviation values for image normalization.
        """
        device = select_device(self.args.device, verbose=verbose)
        if model is None:
            model = build_sam(self.args.model)
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

    def postprocess(self, preds, img, orig_imgs):
        """
        Post-processes SAM's inference outputs to generate object detection masks and bounding boxes.

        The method scales masks and boxes to the original image size and applies a threshold to the mask predictions. The
        SAM model uses advanced architecture and promptable segmentation tasks to achieve real-time performance.

        Args:
            preds (tuple): The output from SAM model inference, containing masks, scores, and optional bounding boxes.
            img (torch.Tensor): The processed input image tensor.
            orig_imgs (list | torch.Tensor): The original, unprocessed images.

        Returns:
            (list): List of Results objects containing detection masks, bounding boxes, and other metadata.
        """
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(len(pred_masks))))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for i, masks in enumerate([pred_masks]):
            orig_img = orig_imgs[i]
            if pred_bboxes is not None:
                pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)

            masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
            masks = masks > self.model.mask_threshold  # to bool
            img_path = self.batch[0][i]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # Reset segment-all mode.
        self.segment_all = False
        return results

    def setup_source(self, source):
        """
        Sets up the data source for inference.

        This method configures the data source from which images will be fetched for inference. The source could be a
        directory, a video file, or other types of image data sources.

        Args:
            source (str | Path): The path to the image data source for inference.
        """
        if source is not None:
            super().setup_source(source)

    def set_image(self, image):
        """
        Preprocesses and sets a single image for inference.

        This function sets up the model if not already initialized, configures the data source to the specified image,
        and preprocesses the image for feature extraction. Only one image can be set at a time.

        Args:
            image (str | np.ndarray): Image file path as a string, or a np.ndarray image read by cv2.

        Raises:
            AssertionError: If more than one image is set.
        """
        if self.model is None:
            model = build_sam(self.args.model)
            self.setup_model(model)
        self.setup_source(image)
        assert len(self.dataset) == 1, "`set_image` only supports setting one image!"
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.model.image_encoder(im)
            self.im = im
            break

    def set_prompts(self, prompts):
        """Set prompts in advance."""
        self.prompts = prompts

    def reset_image(self):
        """Resets the image and its features to None."""
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """
        Perform post-processing on segmentation masks generated by the Segment Anything Model (SAM). Specifically, this
        function removes small disconnected regions and holes from the input masks, and then performs Non-Maximum
        Suppression (NMS) to eliminate any newly created duplicate boxes.

        Args:
            masks (torch.Tensor): A tensor containing the masks to be processed. Shape should be (N, H, W), where N is
                                  the number of masks, H is height, and W is width.
            min_area (int): The minimum area below which disconnected regions and holes will be removed. Defaults to 0.
            nms_thresh (float): The IoU threshold for the NMS algorithm. Defaults to 0.7.

        Returns:
            (tuple([torch.Tensor, List[int]])):
                - new_masks (torch.Tensor): The processed masks with small regions removed. Shape is (N, H, W).
                - keep (List[int]): The indices of the remaining masks post-NMS, which can be used to filter the boxes.
        """
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
