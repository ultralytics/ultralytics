# Ultralytics YOLO 🚀, AGPL-3.0 license

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision.ops.boxes import box_area

from ultralytics.yolo.data.augment import LetterBox
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ops
from ultralytics.yolo.utils.torch_utils import select_device

from .amg import (batch_iterator, batched_mask_to_box, build_all_layer_point_grids, calculate_stability_score,
                  generate_crop_boxes, is_box_near_crop_edge, uncrop_boxes_xyxy, uncrop_masks)


class Predictor(BasePredictor):

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        # Args for set_image
        self.im = None
        self.features = None
        # Args for segment everything
        self.segment_all = False
        self.crop_n_layers = 0
        self.crop_overlap_ratio = 512 / 1500
        self.crop_n_points_downscale_factor = 1
        self.point_grids = None
        self.points_per_side = 32
        self.points_per_batch = 64
        self.pred_iou_thresh = 0.88
        self.stability_score_offset = 1.0
        self.stability_score_thresh = 0.95

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

    def pre_preprocess(self, im):
        """Pre-transform input image before inference.

        Args:
            im (List(np.ndarray)): (N, 3, h, w) for tensor, [(h, w, 3) x N] for list.

        Return: A list of transformed imgs.
        """
        assert len(im) == 1, 'SAM model has not supported batch inference yet!'
        return [LetterBox(self.imgsz, auto=False)(image=x) for x in im]

    def inference(self, im, boxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """
        Predict masks for the given input prompts, using the currently set image.

        Args:
            im (torch.Tensor): The preprocessed image, (N, C, H, W).
            boxes (np.ndarray | List, None): (N, 4), in XYXY format.
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
        if all([i is None for i in [boxes, points, masks]]):
            return self.generate(im)
        return self.prompt_inference(im, boxes, points, labels, masks, multimask_output)

    def prompt_inference(self, im, boxes=None, points=None, labels=None, masks=None, multimask_output=False):
        """
        Predict masks for the given input prompts, using the currently set image.

        Args:
            im (torch.Tensor): The preprocessed image, (N, C, H, W).
            boxes (np.ndarray | List, None): (N, 4), in XYXY format.
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
        # Transform input prompts
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            # Assuming labels are all positive if users don't pass labels.
            if labels is None:
                labels = np.ones(points.shape[0])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            # points = ops.scale_coords(src_shape, points, dst_shape)
            # (N, 2) --> (N, 1, 2), (N, ) --> (N, 1)
            points, labels = points[:, None, :], labels[:, None]
        if boxes is not None:
            boxes = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
            boxes = ops.scale_boxes(src_shape, boxes, dst_shape)
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device)
            masks = masks[:, None, :, :]

        points = (points, labels) if points is not None else None
        # Embed prompts
        sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
            points=points,
            boxes=boxes,
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

    # TODO: This function is WIP.
    def generate(self, im):
        """Segment the whole image.

        Args:
            im(torch.Tensor): The preprocessed image, (N, C, H, W).
        """
        self.segment_all = True
        ih, iw = im.shape[2:]
        crop_boxes, layer_idxs = generate_crop_boxes((ih, iw), self.crop_n_layers, self.crop_overlap_ratio)
        self.point_grids = build_all_layer_point_grids(
            self.points_per_side,
            self.crop_n_layers,
            self.crop_n_points_downscale_factor,
        )
        pred_masks, pred_scores, pred_bboxes = [], [], []
        for crop_box, layer_idx in zip(crop_boxes, layer_idxs):
            x1, y1, x2, y2 = crop_box
            w, h = x2 - x1, y2 - y1
            points_scale = np.array([[w, h]])  # w, h
            # Crop image and interpolate to input size
            crop_im = F.interpolate(im[..., y1:y2, x1:x2], (ih, iw), mode='bilinear', align_corners=False)
            # (num_points, 2)
            points_for_image = self.point_grids[layer_idx] * points_scale
            crop_masks, crop_scores, crop_bboxes = [], [], []
            for (points, ) in batch_iterator(self.points_per_batch, points_for_image):
                pred_mask, pred_score = self.prompt_inference(crop_im, points=points, multimask_output=True)
                # Interpolate predicted masks to input size
                pred_mask = F.interpolate(pred_mask[None], (h, w), mode='bilinear', align_corners=False)[0]
                idx = pred_score > self.pred_iou_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]

                stability_score = calculate_stability_score(pred_mask, self.model.mask_threshold,
                                                            self.stability_score_offset)
                idx = stability_score > self.stability_score_thresh
                pred_mask, pred_score = pred_mask[idx], pred_score[idx]
                # (N, 4)
                pred_bbox = batched_mask_to_box(pred_mask > self.model.mask_threshold)
                keep_mask = ~is_box_near_crop_edge(pred_bbox, crop_box, [0, 0, iw, ih])
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
            keep = torchvision.ops.nms(crop_bboxes.float(), crop_scores, self.args.iou)  # NMS
            crop_bboxes = uncrop_boxes_xyxy(crop_bboxes[keep], crop_box)
            crop_masks = uncrop_masks(crop_masks[keep], crop_box, ih, iw)
            crop_scores = crop_scores[keep]

            pred_masks.append(crop_masks)
            pred_bboxes.append(crop_bboxes)
            pred_scores.append(crop_scores)

        pred_masks = torch.cat(pred_masks)
        pred_bboxes = torch.cat(pred_bboxes)
        pred_scores = torch.cat(pred_scores)

        # Remove duplicate masks between crops
        # scores = 1 / box_area(data['crop_boxes'])
        # scores = scores.to(data['boxes'].device)
        # keep_by_nms = torchvision.ops.nms(
        #     data['boxes'].float(),
        #     scores,
        #     torch.zeros_like(data['boxes'][:, 0]),  # categories
        #     iou_threshold=self.crop_nms_thresh,
        # )

        return torch.cat(pred_masks), torch.cat(pred_scores), torch.cat(pred_bboxes)

    def setup_model(self, model):
        """Set up YOLO model with specified thresholds and device."""
        device = select_device(self.args.device)
        model.eval()
        # self.model = SamAutomaticMaskGenerator(model.to(device),
        #                                        pred_iou_thresh=self.args.conf,
        #                                        box_nms_thresh=self.args.iou)
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
        names = dict(enumerate([str(i) for i in range(len(pred_masks))]))
        results = []
        for i, masks in enumerate([pred_masks]):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if pred_bboxes is not None:
                pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape)

            masks = ops.scale_masks(masks[None], orig_img.shape[:2])[0]
            masks = masks > self.model.mask_threshold  # to bool
            path = self.batch[0]
            img_path = path[i] if isinstance(path, list) else path
            cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
            boxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
            results.append(Results(orig_img=orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        # Reset segment-all mode.
        self.segment_all = False
        return results

    def set_image(self, image):
        """Set image in advance.
        Args:

            im (torch.Tensor | np.ndarray): BCHW for tensor, HWC for ndarray.
        """
        im = self.preprocess(image)
        self.features = self.model.image_encoder(im)
        self.im = im

    def reset_image(self):
        self.im = None
        self.features = None
