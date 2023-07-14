# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.utils.torch_utils import select_device

from .amg import (batch_iterator, batched_mask_to_box, build_all_layer_point_grids, calculate_stability_score,
                  generate_crop_boxes, is_box_near_crop_edge, remove_small_regions, uncrop_boxes_xyxy, uncrop_masks)
from .build import build_sam


class Predictor(BasePredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        if overrides is None:
            overrides = {}
        overrides.update(dict(task='segment', mode='predict', imgsz=1024))
        super().__init__(cfg, overrides, _callbacks)
        # SAM needs retina_masks=True, or the results would be a mess.
        self.args.retina_masks = True
        # Args for set_image
        self.im = None
        self.features = None
        # Args for segment everything
        self.segment_all = False

    def preprocess(self, im):
        """Prepares input image before inference.

        Args:
            im (torch.Tensor | List(np.ndarray)): BCHW for tensor, [(HWC) x B] for list.
        """
        if self.im is not None:
            return self.im
        not_tensor = not isinstance(im, torch.Tensor)
        if not_tensor:
            im = np.stack(self.pre_transform(im))
            im = im[..., ::-1].transpose((0, 3, 1, 2))  # BGR to RGB, BHWC to BCHW, (n, 3, h, w)
            im = np.ascontiguousarray(im)  # contiguous
            im = torch.from_numpy(im)

        img = im.to(self.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        if not_tensor:
            img = (img - self.mean) / self.std
        return img

    def pre_transform(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        assert len(im) == 1, 'SAM model has not supported batch inference yet!'
        return [LetterBox(self.args.imgsz, auto=False, center=False)(image=x) for x in im]

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False, *args, **kwargs):
        """
        Predict masks for the given input prompts, using the currently set image.

        Args:
            im (torch.Tensor): The preprocessed image, (N, C, H, W).
            bboxes (np.ndarray | List, None): (N, 4), in XYXY format.
            points (np.ndarray | List, None): (N, 2), Each point is in (X,Y) in pixels.
            labels (np.ndarray | List, None): (N, ), labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            masks (np.ndarray, None): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form (N, H, W), where
                for SAM, H=W=256.
            multimask_output (bool): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.

        Returns:
            (np.ndarray): The output masks in CxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
            (np.ndarray): An array of length C containing the model's
                predictions for the quality of each mask.
            (np.ndarray): An array of shape CxHxW, where C is the number
                of masks and H=W=256. These low resolution logits can be passed to
                a subsequent iteration as mask input.
        """
        if all(i is None for i in [bboxes, points, masks]):
            return self.generate(im, *args, **kwargs)
        return self.prompt_inference(im, bboxes, points, labels, masks, multimask_output)

    def prompt_inference(self, im, bboxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """
        Predict masks for the given input prompts, using the currently set image.

        Args:
            im (torch.Tensor): The preprocessed image, (N, C, H, W).
            bboxes (np.ndarray | List, None): (N, 4), in XYXY format.
            points (np.ndarray | List, None): (N, 2), Each point is in (X,Y) in pixels.
            labels (np.ndarray | List, None): (N, ), labels for the point prompts.
                1 indicates a foreground point and 0 indicates a background point.
            masks (np.ndarray, None): A low resolution mask input to the model, typically
                coming from a previous prediction iteration. Has form (N, H, W), where
                for SAM, H=W=256.
            multimask_output (bool): If true, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will often
                produce better masks than a single prediction. If only a single
                mask is needed, the model's predicted quality score can be used
                to select the best mask. For non-ambiguous prompts, such as multiple
                input prompts, multimask_output=False can give better results.

        Returns:
            (np.ndarray): The output masks in CxHxW format, where C is the
                number of masks, and (H, W) is the original image size.
            (np.ndarray): An array of length C containing the model's
                predictions for the quality of each mask.
            (np.ndarray): An array of shape CxHxW, where C is the number
                of masks and H=W=256. These low resolution logits can be passed to
                a subsequent iteration as mask input.
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
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
            masks = masks[:, None, :, :]

        points = (points, labels) if points is not None else None
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=bboxes,
            masks=masks,
        )

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

    def generate(self,
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
                 crop_nms_thresh=0.7):
        """Segment the whole image.

        Args:
            im (torch.Tensor): The preprocessed image, (N, C, H, W).
            crop_n_layers (int): If >0, mask prediction will be run again on
                crops of the image. Sets the number of layers to run, where each
                layer has 2**i_layer number of image crops.
            crop_overlap_ratio (float): Sets the degree to which crops overlap.
                In the first crop layer, crops will overlap by this fraction of
                the image length. Later layers with more crops scale down this overlap.
            crop_downscale_factor (int): The number of points-per-side
                sampled in layer n is scaled down by crop_n_points_downscale_factor**n.
            point_grids (list(np.ndarray), None): A list over explicit grids
                of points used for sampling, normalized to [0,1]. The nth grid in the
                list is used in the nth crop layer. Exclusive with points_per_side.
            points_stride (int, None): The number of points to be sampled
                along one side of the image. The total number of points is
                points_per_side**2. If None, 'point_grids' must provide explicit
                point sampling.
            points_batch_size (int): Sets the number of points run simultaneously
                by the model. Higher numbers may be faster but use more GPU memory.
            conf_thres (float): A filtering threshold in [0,1], using the
                model's predicted mask quality.
            stability_score_thresh (float): A filtering threshold in [0,1], using
                the stability of the mask under changes to the cutoff used to binarize
                the model's mask predictions.
            stability_score_offset (float): The amount to shift the cutoff when
                calculated the stability score.
            crop_nms_thresh (float): The box IoU cutoff used by non-maximal
                suppression to filter duplicate masks between different crops.
        """
        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_regions, layer_idxs = generate_crop_boxes((ih, iw), crop_n_layers, crop_overlap_ratio)
        if point_grids is None:
            point_grids = build_all_layer_point_grids(
                points_stride,
                crop_n_layers,
                crop_downscale_factor,
            )
        pred_masks, pred_scores, pred_bboxes, region_areas = [], [], [], []
        for crop_region, layer_idx in zip(crop_regions, layer_idxs):
            x1, y1, x2, y2 = crop_region
            w, h = x2 - x1, y2 - y1
            area = torch.tensor(w * h, device=im.device)
            points_scale = np.array([[w, h]])  # w, h
            # Crop image and interpolate to input size
            crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode='bilinear', align_corners=False)
            # (num_points, 2)
            points_for_image = point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = [], [], []
            for (points, ) in batch_iterator(points_batch_size, points_for_image):
                pred_mask, pred_score = self.prompt_inference(crop_im, points=points, multimask_output=True)
                # Interpolate predicted masks to input size
                pred_mask = F.interpolate(pred_mask[None], (h, w), mode='bilinear', align_corners=False)[0]
                idx = pred_score > conf_thres
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]

                stability_score = calculate_stability_score(pred_mask, self.model.mask_threshold,
                                                            stability_score_offset)
                idx = stability_score > stability_score_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]
                # Bool type is much more memory-efficient.
                pred_mask = pred_mask > self.model.mask_threshold
                # (N, 4)
                pred_bbox = batched_mask_to_box(pred_mask).float()
                keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_region, [0, 0, iw, ih])
                if not torch.all(keep_mask):
                    pred_bbox = pred_bbox[keep_mask]
                    pred_mask = pred_mask[keep_mask]
                    pred_score = pred_score[keep_mask]

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
            pred_masks = pred_masks[keep]
            pred_bboxes = pred_bboxes[keep]
            pred_scores = pred_scores[keep]

        return pred_masks, pred_scores, pred_bboxes

    def setup_model(self, model, verbose=True):
        """Set up YOLO model with specified thresholds and device."""
        device = select_device(self.args.device)
        if model is None:
            model = build_sam(self.args.model)
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.mean = torch.tensor([123.675, 116.28, 103.53]).view(-1, 1, 1).to(device)
        self.std = torch.tensor([58.395, 57.12, 57.375]).view(-1, 1, 1).to(device)
        # TODO: Temporary settings for compatibility
        self.model.pt = False
        self.model.triton = False
        self.model.stride = 32
        self.model.fp16 = False
        self.done_warmup = True

    def postprocess(self, preds, img, orig_imgs):
        """Postprocesses inference output predictions to create detection masks for objects."""
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(len(pred_masks))))
        results = []
        for i, masks in enumerate([pred_masks]):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if pred_bboxes is not None:
                pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)

            masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
            masks = masks > self.model.mask_threshold  # to bool
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=names, masks=masks, boxes=pred_bboxes))
        # Reset segment-all mode.
        self.segment_all = False
        return results

    def setup_source(self, source):
        """Sets up source and inference mode."""
        if source is not None:
            super().setup_source(source)

    def set_image(self, image):
        """Set image in advance.
        Args:

            image (str | np.ndarray): image file path or np.ndarray image by cv2.
        """
        if self.model is None:
            model = build_sam(self.args.model)
            self.setup_model(model)
        self.setup_source(image)
        assert len(self.dataset) == 1, '`set_image` only supports setting one image!'
        for batch in self.dataset:
            im = self.preprocess(batch[1])
            self.features = self.model.image_encoder(im)
            self.im = im
            break

    def reset_image(self):
        self.im = None
        self.features = None

    @staticmethod
    def remove_small_regions(masks, min_area=0, nms_thresh=0.7):
        """
        Removes small disconnected regions and holes in masks, then reruns
        box NMS to remove any new duplicates. Requires open-cv as a dependency.

        Args:
            masks (torch.Tensor): Masks, (N, H, W).
            min_area (int): Minimum area threshold.
            nms_thresh (float): NMS threshold.
        """
        if len(masks) == 0:
            return masks

        # Filter small disconnected regions and holes
        new_masks = []
        scores = []
        for mask in masks:
            mask = mask.cpu().numpy()
            mask, changed = remove_small_regions(mask, min_area, mode='holes')
            unchanged = not changed
            mask, changed = remove_small_regions(mask, min_area, mode='islands')
            unchanged = unchanged and not changed

            new_masks.append(torch.as_tensor(mask).unsqueeze(0))
            # Give score=0 to changed masks and score=1 to unchanged masks
            # so NMS will prefer ones that didn't need postprocessing
            scores.append(float(unchanged))

        # Recalculate boxes and remove any new duplicates
        new_masks = torch.cat(new_masks, dim=0)
        boxes = batched_mask_to_box(new_masks)
        keep = torchvision.ops.nms(
            boxes.float(),
            torch.as_tensor(scores),
            nms_thresh,
        )

        # Only recalculate masks for masks that have changed
        for i in keep:
            if scores[i] == 0.0:
                masks[i] = new_masks[i]

        return masks[keep]
