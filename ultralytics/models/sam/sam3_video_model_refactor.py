from .predict import SAM3VideoPredictor, SAM3SemanticPredictor
from ultralytics.utils import DEFAULT_CFG, LOGGER
import math
import logging
import math
from collections import defaultdict
from copy import deepcopy
from enum import Enum
from typing import Any, Dict, List, Set

import numpy as np
import numpy.typing as npt
import torch
import torch.distributed as dist
import torch.nn.functional as F

from ultralytics.utils import LOGGER
from torch import Tensor

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ultralytics.utils.metrics import box_iou


class MaskletConfirmationStatus(Enum):
    UNCONFIRMED = 1  # newly added masklet, not confirmed by any detection yet
    CONFIRMED = 2  # confirmed by at least one detection


def mask_iou(pred_masks: torch.Tensor, gt_masks: torch.Tensor) -> torch.Tensor:
    """
    Compute the IoU (Intersection over Union) between predicted masks and ground truth masks.
    Args:
      - pred_masks: (N, H, W) bool Tensor, containing binary predicted segmentation masks
      - gt_masks: (M, H, W) bool Tensor, containing binary ground truth segmentation masks
    Returns:
      - ious: (N, M) float Tensor, containing IoUs for each pair of predicted and ground truth masks
    """
    assert pred_masks.dtype == gt_masks.dtype == torch.bool
    N, H, W = pred_masks.shape
    M, _, _ = gt_masks.shape

    # Flatten masks: (N, 1, H*W) and (1, M, H*W)
    pred_flat = pred_masks.view(N, 1, H * W)
    gt_flat = gt_masks.view(1, M, H * W)

    # Compute intersection and union: (N, M)
    intersection = (pred_flat & gt_flat).sum(dim=2).float()
    union = (pred_flat | gt_flat).sum(dim=2).float()
    ious = intersection / union.clamp(min=1)
    return ious  # shape: (N, M)


def mask_to_box(masks: torch.Tensor):
    """
    compute bounding box given an input mask

    Inputs:
    - masks: [B, 1, H, W] tensor

    Returns:
    - box_coords: [B, 1, 4], contains (x, y) coordinates of top left and bottom right box corners, dtype=torch.Tensor
    """
    B, _, h, w = masks.shape
    device = masks.device
    mask_area = masks.sum(dim=(-1, -2))
    xs = torch.arange(w, device=device, dtype=torch.int32)
    ys = torch.arange(h, device=device, dtype=torch.int32)
    grid_xs, grid_ys = torch.meshgrid(xs, ys, indexing="xy")
    grid_xs = grid_xs[None, None, ...].expand(B, 1, h, w)
    grid_ys = grid_ys[None, None, ...].expand(B, 1, h, w)
    min_xs, _ = torch.min(torch.where(masks, grid_xs, w).flatten(-2), dim=-1)
    max_xs, _ = torch.max(torch.where(masks, grid_xs, -1).flatten(-2), dim=-1)
    min_ys, _ = torch.min(torch.where(masks, grid_ys, h).flatten(-2), dim=-1)
    max_ys, _ = torch.max(torch.where(masks, grid_ys, -1).flatten(-2), dim=-1)
    bbox_coords = torch.stack((min_xs, min_ys, max_xs, max_ys), dim=-1)
    bbox_coords = torch.where(mask_area[..., None] > 0, bbox_coords, torch.zeros_like(bbox_coords))
    return bbox_coords


class SAM3VideoSemanticPredictor(SAM3SemanticPredictor):
    """Segment Anything Model 3 (SAM3) Video Semantic Predictor."""

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
        bpe_path="runs/bpe_simple_vocab_16e6.txt.gz",
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
        # Whether to suppress masks only within hotstart. If False, we can suppress masks even if they start before hotstart period.
        suppress_unmatched_only_within_hotstart=True,
        init_trk_keep_alive=0,
        max_trk_keep_alive=8,
        min_trk_keep_alive=-4,
        # Threshold for suppressing overlapping objects based on recent occlusion
        suppress_overlapping_based_on_recent_occlusion_threshold=0.0,
        decrease_trk_keep_alive_for_empty_masklets=False,
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
        overrides = overrides or dict(imgsz=1008, model="sam3.pt", half=True)
        super().__init__(cfg, overrides, _callbacks, bpe_path=bpe_path)
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
        self.suppress_unmatched_only_within_hotstart = suppress_unmatched_only_within_hotstart
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
        self.rank = 0
        self.world_size = 1
        self._dist_pg_cpu = None  # CPU process group (lazy-initialized on first use)

        # the maximum object number
        if max_num_objects > 0:
            num_obj_for_compile = math.ceil(max_num_objects / self.world_size)
        else:
            max_num_objects = 10000  # no limit
            num_obj_for_compile = 16
        LOGGER.info(f"setting {max_num_objects=} and {num_obj_for_compile=}")
        self.max_num_objects = max_num_objects
        self.num_obj_for_compile = num_obj_for_compile
        self.recondition_every_nth_frame = recondition_every_nth_frame
        self.masklet_confirmation_enable = masklet_confirmation_enable
        self.masklet_confirmation_consecutive_det_thresh = masklet_confirmation_consecutive_det_thresh
        self.reconstruction_bbox_iou_thresh = reconstruction_bbox_iou_thresh
        self.reconstruction_bbox_det_score = reconstruction_bbox_det_score

        # build SAM3 tracker
        self.tracker = SAM3VideoPredictor(overrides=overrides)
        self.tracker.setup_model()
        self.setup_model()
        # TODO
        self.imgsz = [1008, 1008]
        self.tracker.imgsz = [1008, 1008]

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
        images = None
        inference_state = {
            "num_frames": predictor.dataset.num_frames,
            "constants": {},  # values that don't change across frames (so we only need to hold one copy of them)
        }

        predictor._construct_initial_input_batch(inference_state, images)
        predictor.inference_state = inference_state

    def _det_track_one_frame(
        self,
        frame_idx: int,
        num_frames: int,
        reverse: bool,
        input_batch,
        geometric_prompt: Any,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, Any],
        feature_cache: Dict,
        orig_vid_height: int,
        orig_vid_width: int,
        allow_new_detections: bool = True,
    ):
        """
        This function handles one-step inference for the DenseTracking model in an SPMD manner.
        At a high-level, all GPUs execute the same function calls as if it's done on a single GPU,
        while under the hood, some function calls involve distributed computation based on sharded
        SAM2 states.

        - `input_batch` contains image and other inputs on the entire video; it should be identical across GPUs
        - `tracker_states_local` holds the local masklet information in this GPU shard
        - `tracker_metadata_prev` manages the metadata for SAM2 objects, such as which masklet is hold on which GPUs
          it contains both global and local masklet information
        """

        # Step 1: run backbone and detector in a distributed manner -- this is done via Sam3ImageOnVideoMultiGPU,
        # a MultiGPU model (assigned to `self.detector`) that shards frames in a round-robin manner.
        # It returns a "det_out" dict for `frame_idx` and fills SAM2 backbone features for `frame_idx`
        # into `feature_cache`. Despite its distributed inference under the hood, the results would be
        # the same as if it is running backbone and detector for every frame on a single GPU.
        det_out = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            allow_new_detections=allow_new_detections,
        )
        self.tracker.backbone_out = feature_cache[frame_idx][1]["tracker_backbone_out"]

        # TODO: update `im` for SAM2VideoPredictor, need to be fixed
        # for inference_state in tracker_states_local:
        #     inference_state["im"] = feature_cache[frame_idx][0].unsqueeze(0)

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
            feature_cache=feature_cache,
        )

        # Step 5: finally, build the outputs for this frame (it only needs to be done on GPU 0 since
        # only GPU 0 will send outputs to the server).
        if self.rank == 0:
            obj_id_to_mask = self.build_outputs(
                det_out=det_out,
                tracker_low_res_masks_global=tracker_low_res_masks_global,
                tracker_metadata_prev=tracker_metadata_prev,
                tracker_update_plan=tracker_update_plan,
                orig_vid_height=orig_vid_height,
                orig_vid_width=orig_vid_width,
                reconditioned_obj_ids=reconditioned_obj_ids,
            )
            obj_id_to_score = tracker_metadata_new["obj_id_to_score"]
        else:
            obj_id_to_mask, obj_id_to_score = {}, {}  # dummy outputs on other GPUs
        # a few statistics for the current frame as a part of the output
        frame_stats = {
            "num_obj_tracked": np.sum(tracker_metadata_new["num_obj_per_gpu"]),
            "num_obj_dropped": tracker_update_plan["num_obj_dropped_due_to_limit"],
        }
        # add tracker scores to metadata, it should be fired for frames except the first frame
        if tracker_obj_scores_global.shape[0] > 0:
            # Convert tracker_obj_scores_global to sigmoid scores before updating
            tracker_obj_scores_global = tracker_obj_scores_global.sigmoid().tolist()
            tracker_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                dict(zip(tracker_obj_ids, tracker_obj_scores_global))
            )
        return (
            obj_id_to_mask,  # a dict: obj_id --> output mask
            obj_id_to_score,  # a dict: obj_id --> output score (prob)
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            tracker_obj_scores_global,  # a dict: obj_id --> tracker frame-level scores
        )

    def _suppress_detections_close_to_boundary(self, boxes, margin=0.025):
        """
        Suppress detections too close to image edges (for normalized boxes).

        boxes: (N, 4) in xyxy format, normalized [0,1]
        margin: fraction of image
        """
        x_min, y_min, x_max, y_max = boxes.unbind(-1)
        x_c = (x_min + x_max) / 2
        y_c = (y_min + y_max) / 2
        keep = (x_c > margin) & (x_c < 1.0 - margin) & (y_c > margin) & (y_c < 1.0 - margin)

        return keep

    def run_backbone_and_detection(
        self,
        frame_idx: int,
        input_batch,
        geometric_prompt: Any,
        feature_cache: Dict,
        reverse: bool,
        allow_new_detections: bool,
    ):
        # Step 1: if text feature is not cached in `feature_cache`, compute and cache it
        text_batch_key = tuple(input_batch.find_text_batch)
        if "text" not in feature_cache or text_batch_key not in feature_cache["text"]:
            text_outputs = self.model.backbone.forward_text(input_batch.find_text_batch)
            # note: we only cache the text feature of the most recent prompt
            feature_cache["text"] = {text_batch_key: text_outputs}
        else:
            text_outputs = feature_cache["text"][text_batch_key]

        # Step 2: run backbone, detector, and post-processing with NMS
        if "multigpu_buffer" not in feature_cache:
            # "multigpu_buffer" is a buffer cache used by `self.detector` and it needs
            # to be passed to `forward_video_grounding_multigpu` for every call
            feature_cache["multigpu_buffer"] = {}

        sam3_image_out = self.model.forward_grounding(
            backbone_out={"img_batch_all_stages": input_batch.img_batch, **text_outputs},
            find_input=input_batch.find_inputs[frame_idx],
            geometric_prompt=geometric_prompt,
        )
        # TODO: probably apply nms on boxes
        pred_probs = sam3_image_out["pred_logits"].squeeze(-1).sigmoid()
        if not allow_new_detections:
            pred_probs = pred_probs - 1e8  # make sure no detections are kept
        pred_boxes_xyxy = sam3_image_out["pred_boxes_xyxy"]
        pred_masks = sam3_image_out["pred_masks"]
        # get the positive detection outputs above threshold
        pos_pred_idx = torch.where(pred_probs > self.score_threshold_detection)
        det_out = {
            "bbox": pred_boxes_xyxy[pos_pred_idx[0], pos_pred_idx[1]],
            "mask": pred_masks[pos_pred_idx[0], pos_pred_idx[1]],
            "scores": pred_probs[pos_pred_idx[0], pos_pred_idx[1]],
        }

        # Step 3: build SAM2 backbone features and store them in `feature_cache`
        backbone_cache = {}
        sam_mask_decoder = self.tracker.model.sam_mask_decoder
        feats = sam3_image_out["prev_encoder_out"]["backbone_out"]["sam2_backbone_out"]
        tracker_backbone_fpn = [
            sam_mask_decoder.conv_s0(feats["backbone_fpn"][0]),
            sam_mask_decoder.conv_s1(feats["backbone_fpn"][1]),
            feats["backbone_fpn"][2],  # fpn_2 doesn't need conv
        ]
        tracker_backbone_out = {
            "vision_features": tracker_backbone_fpn[-1],  # top-level feature
            "vision_pos_enc": feats["vision_pos_enc"],
            "backbone_fpn": tracker_backbone_fpn,
        }
        backbone_cache["tracker_backbone_out"] = tracker_backbone_out
        feature_cache[frame_idx] = (input_batch.img_batch[frame_idx], backbone_cache)
        # remove from `feature_cache` old features to save GPU memory
        feature_cache.pop(frame_idx - 1 if not reverse else frame_idx + 1, None)
        return det_out

    def run_tracker_propagation(
        self,
        frame_idx: int,
        tracker_states_local: List[Any],
        tracker_metadata_prev: Dict[str, npt.NDArray],
    ):
        # Step 1: propagate the local SAM2 states to get the current frame's prediction
        # `low_res_masks_local` of the existing masklets on this GPU
        # - obj_ids_local: List[int] -- list of object IDs
        # - low_res_masks_local: Tensor -- (num_local_obj, H_mask, W_mask)
        obj_ids_local, low_res_masks_local, obj_scores_local = self._propogate_tracker_one_frame_local_gpu(
            tracker_states_local, frame_idx=frame_idx
        )

        assert np.all(obj_ids_local == tracker_metadata_prev["obj_ids_per_gpu"][self.rank]), "{} != {}".format(
            obj_ids_local, tracker_metadata_prev["obj_ids_per_gpu"][self.rank]
        )

        # Step 2: all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        _, H_mask, W_mask = low_res_masks_local.shape
        if self.world_size > 1:
            # `low_res_masks_local` and `obj_scores_local` need to be contiguous and float32
            # (they could be non-contiguous due to slicing and/or bfloat16 due to autocast)
            low_res_masks_local = low_res_masks_local.float().contiguous()
            obj_scores_local = obj_scores_local.float().contiguous()
            num_obj_this_gpu = tracker_metadata_prev["num_obj_per_gpu"][self.rank]
            assert low_res_masks_local.size(0) == num_obj_this_gpu
            assert obj_scores_local.size(0) == num_obj_this_gpu
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            obj_scores_peers = [
                obj_scores_local.new_empty(num_obj) for num_obj in tracker_metadata_prev["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            dist.all_gather(obj_scores_peers, obj_scores_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
            obj_scores_global = torch.cat(obj_scores_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
            obj_scores_global = obj_scores_local
        return low_res_masks_global, obj_scores_global

    def _recondition_masklets(
        self,
        frame_idx,
        det_out: Dict[str, Tensor],
        trk_id_to_max_iou_high_conf_det: List[int],
        tracker_states_local: List[Any],
        tracker_metadata: Dict[str, npt.NDArray],
        tracker_obj_scores_global: Tensor,
    ):
        # Recondition the masklets based on the new detections
        for trk_obj_id, det_idx in trk_id_to_max_iou_high_conf_det.items():
            new_mask = det_out["mask"][det_idx : det_idx + 1]
            new_mask_binary = (
                F.interpolate(new_mask.unsqueeze(1), size=[1152, 1152], mode="bilinear", align_corners=False) > 0
            )
            HIGH_CONF_THRESH = 0.8
            reconditioned_states_idx = set()
            obj_idx = np.where(tracker_metadata["obj_ids_all_gpu"] == trk_obj_id)[0].item()
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
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_obj_scores_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_states_local: List[Any],
    ):
        # initialize new metadata from previous metadata (its values will be updated later)
        tracker_metadata_new = {
            "obj_ids_per_gpu": deepcopy(tracker_metadata_prev["obj_ids_per_gpu"]),
            "obj_ids_all_gpu": None,  # will be filled later
            "num_obj_per_gpu": deepcopy(tracker_metadata_prev["num_obj_per_gpu"]),
            "obj_id_to_score": deepcopy(tracker_metadata_prev["obj_id_to_score"]),
            "obj_id_to_tracker_score_frame_wise": deepcopy(tracker_metadata_prev["obj_id_to_tracker_score_frame_wise"]),
            "obj_id_to_last_occluded": {},  # will be filled later
            "max_obj_id": deepcopy(tracker_metadata_prev["max_obj_id"]),
        }

        # Initialize reconditioned_obj_ids early to avoid UnboundLocalError
        reconditioned_obj_ids = set()

        # Step 1: make the update plan and resolve heuristics on GPU 0
        det_mask_preds: Tensor = det_out["mask"]  # low-res mask logits
        det_scores_np: npt.NDArray = det_out["scores"].float().cpu().numpy()
        det_bbox_xyxy: Tensor = det_out["bbox"]
        if self.rank == 0:
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
                trk_obj_ids=tracker_metadata_prev["obj_ids_all_gpu"],
            )
            if self.suppress_det_close_to_boundary:
                keep = self._suppress_detections_close_to_boundary(det_bbox_xyxy[new_det_fa_inds])
                new_det_fa_inds = new_det_fa_inds[keep.cpu().numpy()]

            # check whether we've hit the maximum number of objects we can track (and if so, drop some detections)
            prev_obj_num = np.sum(tracker_metadata_prev["num_obj_per_gpu"])
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
            new_det_start_obj_id = tracker_metadata_prev["max_obj_id"] + 1
            new_det_obj_ids = new_det_start_obj_id + np.arange(new_det_num)
            prev_workload_per_gpu = tracker_metadata_prev["num_obj_per_gpu"]
            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=new_det_num,
                prev_workload_per_gpu=prev_workload_per_gpu,
            )

            # b) handle hotstart heuristics to remove objects
            # here `rank0_metadata` contains metadata stored on (and only accessible to) GPU 0;
            # we avoid broadcasting them to other GPUs to save communication cost, assuming
            # that `rank0_metadata` is not needed by other GPUs
            rank0_metadata_new = deepcopy(tracker_metadata_prev["rank0_metadata"])
            if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
                obj_ids_newly_removed, rank0_metadata_new = self._process_hotstart(
                    frame_idx=frame_idx,
                    reverse=reverse,
                    det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                    new_det_obj_ids=new_det_obj_ids,
                    empty_trk_obj_ids=empty_trk_obj_ids,
                    unmatched_trk_obj_ids=unmatched_trk_obj_ids,
                    rank0_metadata=rank0_metadata_new,
                )
            else:
                # if warm-up is not complete, we don't remove any objects
                obj_ids_newly_removed = set()
            tracker_metadata_new["rank0_metadata"] = rank0_metadata_new

        # Step 2: broadcast the update plan to other GPUs
        NUM_BROADCAST_ITEMS = 9
        if self.rank == 0 and self.world_size > 1:
            # `num_obj_per_gpu_on_rank0` is used for metadata consistency check on other GPUs
            # (it's a small array with length==self.world_size, so broadcasting it is cheap)
            num_obj_per_gpu_on_rank0 = tracker_metadata_prev["num_obj_per_gpu"]
            update_plan = [
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ]
            assert len(update_plan) == NUM_BROADCAST_ITEMS, (
                f"Manually update NUM_BROADCAST_ITEMS to be: {len(update_plan)}"
            )
        elif self.rank > 0 and self.world_size > 1:
            update_plan = [None] * NUM_BROADCAST_ITEMS  # other ranks receive the plan from rank 0
            (
                new_det_fa_inds,
                new_det_obj_ids,
                new_det_gpu_ids,
                num_obj_per_gpu_on_rank0,
                unmatched_trk_obj_ids,
                det_to_matched_trk_obj_ids,
                obj_ids_newly_removed,
                num_obj_dropped_due_to_limit,
                trk_id_to_max_iou_high_conf_det,
            ) = update_plan
            # metadata consistency check: verify that the received `num_obj_per_gpu_on_rank0` is consistent with the local metadata
            # it's critical that all GPUs agree on the previous number of objects (otherwise the inference might hang or fail silently)
            if not np.all(num_obj_per_gpu_on_rank0 == tracker_metadata_prev["num_obj_per_gpu"]):
                raise RuntimeError(
                    f"{self.rank=} received {num_obj_per_gpu_on_rank0=}, which is inconsistent with local record "
                    f"{tracker_metadata_prev['num_obj_per_gpu']=}. There's likely a bug in update planning or execution."
                )

        # `tracker_update_plan` should be identical on all GPUs after broadcasting
        tracker_update_plan = {
            "new_det_fa_inds": new_det_fa_inds,  # npt.NDArray
            "new_det_obj_ids": new_det_obj_ids,  # npt.NDArray
            "new_det_gpu_ids": new_det_gpu_ids,  # npt.NDArray
            "unmatched_trk_obj_ids": unmatched_trk_obj_ids,  # npt.NDArray
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
                    trk_idx = list(tracker_metadata_prev["obj_ids_all_gpu"]).index(trk_obj_id)
                except ValueError:
                    continue  # Skip if tracklet not found

                tracker_mask = tracker_low_res_masks_global[trk_idx]
                mask_binary = tracker_mask > 0
                mask_area = mask_binary.sum().item()

                if mask_area == 0:
                    continue  # Skip tracklets with zero mask area

                # Get bounding box from SAM2 mask and convert to normalized coordinates
                tracker_box_pixels = mask_to_box(mask_binary.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
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

            self._tracker_update_memories(
                tracker_states_local,
                frame_idx,
                tracker_metadata=tracker_metadata_prev,
                low_res_masks=tracker_low_res_masks_global,
            )

        # Step 4: update the SAM2 metadata based on the update plan
        # note: except for "rank0_metadata" (that is only available on GPU 0),
        # the updated `tracker_metadata_new` should be identical on all GPUs
        for rank in range(self.world_size):
            new_det_obj_ids_this_gpu = new_det_obj_ids[new_det_gpu_ids == rank]
            updated_obj_ids_this_gpu = tracker_metadata_new["obj_ids_per_gpu"][rank]
            if len(new_det_obj_ids_this_gpu) > 0:
                updated_obj_ids_this_gpu = np.concatenate([updated_obj_ids_this_gpu, new_det_obj_ids_this_gpu])
            if len(obj_ids_newly_removed) > 0:
                is_removed = np.isin(updated_obj_ids_this_gpu, list(obj_ids_newly_removed))
                updated_obj_ids_this_gpu = updated_obj_ids_this_gpu[~is_removed]
            tracker_metadata_new["obj_ids_per_gpu"][rank] = updated_obj_ids_this_gpu
            tracker_metadata_new["num_obj_per_gpu"][rank] = len(updated_obj_ids_this_gpu)
        tracker_metadata_new["obj_ids_all_gpu"] = np.concatenate(tracker_metadata_new["obj_ids_per_gpu"])
        # update object scores and the maximum object ID assigned so far
        if len(new_det_obj_ids) > 0:
            tracker_metadata_new["obj_id_to_score"].update(zip(new_det_obj_ids, det_scores_np[new_det_fa_inds]))
            # tracker scores are not available for new objects, use det score instead.
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx].update(
                zip(new_det_obj_ids, det_scores_np[new_det_fa_inds])
            )
            tracker_metadata_new["max_obj_id"] = max(
                tracker_metadata_new["max_obj_id"],
                np.max(new_det_obj_ids),
            )
        # for removed objects, we set their scores to a very low value (-1e4) but still
        # keep them in "obj_id_to_score" (it's easier to handle outputs this way)
        for obj_id in obj_ids_newly_removed:
            tracker_metadata_new["obj_id_to_score"][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = -1e4
            tracker_metadata_new["obj_id_to_last_occluded"].pop(obj_id, None)
        # check that "rank0_metadata" is in tracker_metadata_new if and only if it's GPU 0
        assert ("rank0_metadata" in tracker_metadata_new) == (self.rank == 0)
        if self.rank == 0 and self.masklet_confirmation_enable:
            rank0_metadata = self.update_masklet_confirmation_status(
                rank0_metadata=tracker_metadata_new["rank0_metadata"],
                obj_ids_all_gpu_prev=tracker_metadata_prev["obj_ids_all_gpu"],
                obj_ids_all_gpu_updated=tracker_metadata_new["obj_ids_all_gpu"],
                det_to_matched_trk_obj_ids=det_to_matched_trk_obj_ids,
                new_det_obj_ids=new_det_obj_ids,
            )
            tracker_metadata_new["rank0_metadata"] = rank0_metadata

        return tracker_update_plan, tracker_metadata_new

    def _suppress_overlapping_based_on_recent_occlusion(
        self,
        frame_idx: int,
        tracker_low_res_masks_global: Tensor,
        tracker_metadata_prev: Dict[str, Any],
        tracker_metadata_new: Dict[str, Any],
        obj_ids_newly_removed: Set[int],
        reverse: bool = False,
    ):
        """
        Suppress overlapping masks based on the most recent occlusion information. If an object is removed by hotstart, we always suppress it if it overlaps with any other object.
        Args:
            frame_idx (int): The current frame index.
            tracker_low_res_masks_global (Tensor): The low-resolution masks for the current frame.
            tracker_metadata_prev (Dict[str, Any]): The metadata from the previous frame.
            tracker_metadata_new (Dict[str, Any]): The metadata for the current frame.
            obj_ids_newly_removed (Set[int]): The object IDs that have been removed.
        Return:
            Tensor: The updated low-resolution masks with some objects suppressed.
        """
        obj_ids_global = tracker_metadata_prev["obj_ids_all_gpu"]
        binary_tracker_low_res_masks_global = tracker_low_res_masks_global > 0
        batch_size = tracker_low_res_masks_global.size(0)
        if batch_size > 0:
            assert len(obj_ids_global) == batch_size, (
                f"Mismatch in number of objects: {len(obj_ids_global)} vs {batch_size}"
            )
            NEVER_OCCLUDED = -1
            ALWAYS_OCCLUDED = 100000  # This value should be larger than any possible frame index, indicates that the object was removed by hotstart logic
            last_occluded_prev = torch.cat(
                [
                    tracker_metadata_prev["obj_id_to_last_occluded"].get(
                        obj_id,
                        torch.full(
                            (1,),
                            fill_value=(NEVER_OCCLUDED if obj_id not in obj_ids_newly_removed else ALWAYS_OCCLUDED),
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
            NO_OBJ_LOGIT = -10
            tracker_low_res_masks_global[to_suppress] = NO_OBJ_LOGIT

        return tracker_low_res_masks_global

    def run_tracker_update_execution_phase(
        self,
        frame_idx: int,
        num_frames: int,
        det_out: Dict[str, Tensor],
        tracker_states_local: List[Any],
        tracker_update_plan: Dict[str, npt.NDArray],
        feature_cache: Dict,
    ):
        # initialize tracking scores with detection scores
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        new_det_gpu_ids: npt.NDArray = tracker_update_plan["new_det_gpu_ids"]
        is_on_this_gpu: npt.NDArray = new_det_gpu_ids == self.rank
        new_det_obj_ids_local: npt.NDArray = new_det_obj_ids[is_on_this_gpu]
        new_det_fa_inds_local: npt.NDArray = new_det_fa_inds[is_on_this_gpu]
        obj_ids_newly_removed: Set[int] = tracker_update_plan["obj_ids_newly_removed"]

        # Step 1: add new objects from the detector to SAM2 inference states
        if len(new_det_fa_inds_local) > 0:
            new_det_fa_inds_local_t = torch.from_numpy(new_det_fa_inds_local)
            new_det_masks: Tensor = det_out["mask"][new_det_fa_inds_local_t]
            # initialize SAM2 with new object masks
            tracker_states_local = self._tracker_add_new_objects(
                frame_idx=frame_idx,
                num_frames=num_frames,
                new_obj_ids=new_det_obj_ids_local,
                new_obj_masks=new_det_masks,
                tracker_states_local=tracker_states_local,
                feature_cache=feature_cache,
            )

        # Step 2: remove from SAM2 inference states those objects removed by heuristics
        if len(obj_ids_newly_removed) > 0:
            self._tracker_remove_objects(tracker_states_local, obj_ids_newly_removed)

        return tracker_states_local

    def build_outputs(
        self,
        det_out: Dict[str, Tensor],
        tracker_low_res_masks_global: Tensor,
        tracker_metadata_prev: Dict[str, npt.NDArray],
        tracker_update_plan: Dict[str, npt.NDArray],
        orig_vid_height: int,
        orig_vid_width: int,
        reconditioned_obj_ids: set = None,
    ):
        new_det_fa_inds: npt.NDArray = tracker_update_plan["new_det_fa_inds"]
        new_det_obj_ids: npt.NDArray = tracker_update_plan["new_det_obj_ids"]
        obj_id_to_mask = {}  # obj_id --> output mask tensor

        # Part 1: masks from previous SAM2 propagation
        existing_masklet_obj_ids = tracker_metadata_prev["obj_ids_all_gpu"]
        existing_masklet_video_res_masks = F.interpolate(
            tracker_low_res_masks_global.unsqueeze(1),
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )  # (num_obj, 1, H_video, W_video)
        existing_masklet_binary = existing_masklet_video_res_masks > 0
        assert len(existing_masklet_obj_ids) == len(existing_masklet_binary)
        for obj_id, mask in zip(existing_masklet_obj_ids, existing_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # Part 2: masks from new detections
        new_det_fa_inds_t = torch.from_numpy(new_det_fa_inds)
        new_det_low_res_masks = det_out["mask"][new_det_fa_inds_t].unsqueeze(1)
        # TODO
        # new_det_low_res_masks = fill_holes_in_mask_scores(
        #     new_det_low_res_masks,
        #     max_area=self.fill_hole_area,
        #     fill_holes=True,
        #     remove_sprinkles=True,
        # )
        new_masklet_video_res_masks = F.interpolate(
            new_det_low_res_masks,
            size=(orig_vid_height, orig_vid_width),
            mode="bilinear",
            align_corners=False,
        )  # (num_obj, 1, H_video, W_video)

        new_masklet_binary = new_masklet_video_res_masks > 0
        assert len(new_det_obj_ids) == len(new_masklet_video_res_masks)
        for obj_id, mask in zip(new_det_obj_ids, new_masklet_binary):
            obj_id_to_mask[obj_id] = mask  # (1, H_video, W_video)

        # Part 3: Override masks for reconditioned objects using detection masks
        if reconditioned_obj_ids is not None and len(reconditioned_obj_ids) > 0:
            trk_id_to_max_iou_high_conf_det = tracker_update_plan.get("trk_id_to_max_iou_high_conf_det", {})

            for obj_id in reconditioned_obj_ids:
                det_idx = trk_id_to_max_iou_high_conf_det.get(obj_id)

                if det_idx is not None:
                    det_mask = det_out["mask"][det_idx]
                    det_mask = det_mask.unsqueeze(0).unsqueeze(0)
                    det_mask_resized = (
                        F.interpolate(
                            det_mask.float(),
                            size=(orig_vid_height, orig_vid_width),
                            mode="bilinear",
                            align_corners=False,
                        )
                        > 0
                    )

                    det_mask_final = det_mask_resized.squeeze(0)
                    obj_id_to_mask[obj_id] = det_mask_final

        return obj_id_to_mask

    def _get_objects_to_suppress_based_on_most_recently_occluded(
        self,
        binary_low_res_masks: Tensor,
        last_occluded: List[int],
        obj_ids: List[int],
        frame_idx: int = None,
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

        iou = mask_iou(binary_low_res_masks, binary_low_res_masks)  # [N,N]

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
        if self.rank == 0 and LOGGER.isEnabledFor(logging.DEBUG) and frame_idx is not None:
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

    def _propogate_tracker_one_frame_local_gpu(self, inference_states: List[Any], frame_idx: int):
        """
        inference_states: List of inference states, each state corresponds to a different set of objects.
        """
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

            # Apply hole filling to the masks
            # NOTE
            # low_res_masks_local = fill_holes_in_mask_scores(
            #     low_res_masks_local.unsqueeze(1),
            #     max_area=self.fill_hole_area,
            #     fill_holes=True,
            #     remove_sprinkles=True,
            # )
            low_res_masks_local = low_res_masks_local.squeeze(1)
        else:
            low_res_masks_local = torch.zeros(0, *self._bb_feat_sizes[0], device=self.device)
            obj_scores_local = torch.zeros(0, device=self.device)

        return obj_ids_local, low_res_masks_local, obj_scores_local

    def _associate_det_trk(
        self,
        det_masks: Tensor,
        det_scores_np: npt.NDArray,
        trk_masks: Tensor,
        trk_obj_ids: npt.NDArray,
    ):
        """
        Match detections on the current frame with the existing masklets.

        Args:
          - det_masks: (N, H, W) tensor of predicted masks
          - det_scores_np: (N,) array of detection scores
          - trk_masks: (M, H, W) tensor of track masks
          - trk_obj_ids: (M,) array of object IDs corresponding to trk_masks

        Returns:
          - new_det_fa_inds: array of new object indices.
          - unmatched_trk_obj_ids: array of existing masklet object IDs that are not matched
            to any detections on this frame (for unmatched, we only count masklets with >0 area)
          - det_to_matched_trk_obj_ids: dict[int, npt.NDArray]: mapping from detector's detection indices
            to the list of matched tracklet object IDs
          - empty_trk_obj_ids: array of existing masklet object IDs with zero area in SAM2 prediction
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
        ious = mask_iou(det_masks_binary, trk_masks_binary)  # (N, M)

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
        HIGH_CONF_THRESH = 0.8
        HIGH_IOU_THRESH = 0.8
        det_to_max_iou_trk_idx = np.argmax(ious_np, axis=1)
        det_is_high_conf = (det_scores_np >= HIGH_CONF_THRESH) & ~is_new_det
        det_is_high_iou = np.max(ious_np, axis=1) >= HIGH_IOU_THRESH
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

    def _assign_new_det_to_gpus(self, new_det_num, prev_workload_per_gpu):
        """Distribute the new objects to the GPUs with the least workload."""
        workload_per_gpu: npt.NDArray = prev_workload_per_gpu.copy()
        new_det_gpu_ids = np.zeros(new_det_num, np.int64)

        # assign the objects one by one
        for i in range(len(new_det_gpu_ids)):
            # find the GPU with the least workload
            min_gpu = np.argmin(workload_per_gpu)
            new_det_gpu_ids[i] = min_gpu
            workload_per_gpu[min_gpu] += 1
        return new_det_gpu_ids

    def _process_hotstart(
        self,
        frame_idx: int,
        reverse: bool,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
        empty_trk_obj_ids: npt.NDArray,
        unmatched_trk_obj_ids: npt.NDArray,
        rank0_metadata: Dict[str, Any],
    ):
        """Handle hotstart heuristics to remove unmatched or duplicated objects."""
        # obj_id --> first frame index where the object was detected
        obj_first_frame_idx = rank0_metadata["obj_first_frame_idx"]
        # obj_id --> [mismatched frame indices]
        unmatched_frame_inds = rank0_metadata["unmatched_frame_inds"]
        trk_keep_alive = rank0_metadata["trk_keep_alive"]
        # (first_appear_obj_id, obj_id) --> [overlap frame indices]
        overlap_pair_to_frame_inds = rank0_metadata["overlap_pair_to_frame_inds"]
        # removed_obj_ids: object IDs that are suppressed via hot-start
        removed_obj_ids = rank0_metadata["removed_obj_ids"]
        suppressed_obj_ids = rank0_metadata["suppressed_obj_ids"][frame_idx]

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
                and not self.suppress_unmatched_only_within_hotstart
                and obj_id not in removed_obj_ids
                and obj_id not in obj_ids_newly_removed
            ):
                LOGGER.debug(f"Suppressing object {obj_id} at frame {frame_idx}, due to being unmatched")
                suppressed_obj_ids.add(obj_id)

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
        return obj_ids_newly_removed, rank0_metadata

    def _tracker_update_memories(
        self,
        tracker_inference_states: List[Any],
        frame_idx: int,
        tracker_metadata: Dict[str, Any],
        low_res_masks: Tensor,
    ):
        """
        Run Sam2 memory encoder, enforcing non-overlapping constraints globally.
        """
        if len(tracker_inference_states) == 0:
            return
        # Avoid an extra interpolation step by directly interpolating to `interpol_size`
        high_res_H, high_res_W = self.tracker.model.memory_encoder.mask_downsampler.interpol_size
        # NOTE: inspect this part if we observe OOMs in the demo
        high_res_masks = F.interpolate(
            low_res_masks.unsqueeze(1),
            size=(high_res_H, high_res_W),
            mode="bilinear",
            align_corners=False,
        )
        # We first apply non-overlapping constraints before memory encoding. This may include some suppression heuristics.
        # TODO
        # if not hasattr(self, "_warm_up_complete") or self._warm_up_complete:
        #     high_res_masks = self.tracker._suppress_object_pw_area_shrinkage(high_res_masks)
        # Instead of gathering the predicted object scores, we use mask areas as a proxy.
        object_score_logits = torch.where((high_res_masks > 0).any(dim=(-1, -2)), 10.0, -10.0)

        # Run the memory encoder on local slices for each GPU
        start_idx_gpu = sum(tracker_metadata["num_obj_per_gpu"][: self.rank])
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
        new_obj_ids: List[int],
        new_obj_masks: Tensor,
        tracker_states_local: List[Any],
        feature_cache: Dict,
    ):
        """Add a new object to SAM2 inference states."""
        prev_tracker_state = tracker_states_local[0] if len(tracker_states_local) > 0 else None

        # prepare inference_state
        # batch objects that first appear on the same frame together
        # Clear inference state. Keep the cached image features if available.
        new_tracker_state = self.tracker._init_state(num_frames=num_frames)
        # TODO: adding image placeholder
        new_tracker_state["im"] = feature_cache[frame_idx][0].unsqueeze(0)
        new_tracker_state["backbone_out"] = (
            prev_tracker_state.get("backbone_out", None) if prev_tracker_state is not None else None
        )

        assert len(new_obj_ids) == new_obj_masks.size(0)
        assert new_obj_masks.is_floating_point()
        new_obj_masks = F.interpolate(
            new_obj_masks.unsqueeze(1),
            size=[1152, 1152],
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)
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

    def _tracker_remove_object(self, tracker_states_local: List[Any], obj_id: int):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        tracker_states_local_before_removal = tracker_states_local.copy()
        tracker_states_local.clear()
        for tracker_inference_state in tracker_states_local_before_removal:
            # we try to remove `obj_id` on every inference state with `strict=False`
            # it will not do anything if an inference state doesn't contain `obj_id`
            new_obj_ids = self.tracker.remove_object(tracker_inference_state, obj_id, strict=False)
            # only keep an inference state if it's non-empty after object removal
            if len(new_obj_ids) > 0:
                tracker_states_local.append(tracker_inference_state)

    def _tracker_remove_objects(self, tracker_states_local: List[Any], obj_ids: list[int]):
        """
        Remove an object from SAM2 inference states. This would remove the object from
        all frames in the video.
        """
        for obj_id in obj_ids:
            self._tracker_remove_object(tracker_states_local, obj_id)

    def _initialize_metadata(self):
        """Initialize metadata for the masklets."""
        tracker_metadata = {
            "obj_ids_per_gpu": [np.array([], np.int64) for _ in range(self.world_size)],
            "obj_ids_all_gpu": np.array([], np.int64),
            "num_obj_per_gpu": np.zeros(self.world_size, np.int64),
            "max_obj_id": -1,
            "obj_id_to_score": {},
            "obj_id_to_tracker_score_frame_wise": defaultdict(dict),
            "obj_id_to_last_occluded": {},
        }
        if self.rank == 0:
            # "rank0_metadata" contains metadata that is only stored on (and accessible to) GPU 0
            # - obj_first_frame_idx: obj_id --> first frame index where the object was detected
            # - unmatched_frame_inds: obj_id --> [mismatched frame indices]
            # - overlap_pair_to_frame_inds: (first_appear_obj_id, obj_id) --> [overlap frame indices]
            # - removed_obj_ids: object IDs that are suppressed via hot-start
            rank0_metadata = {
                "obj_first_frame_idx": {},
                "unmatched_frame_inds": defaultdict(list),
                "trk_keep_alive": defaultdict(int),  # This is used only for object suppression not for removal
                "overlap_pair_to_frame_inds": defaultdict(list),
                "removed_obj_ids": set(),
                "suppressed_obj_ids": defaultdict(
                    set
                ),  # frame_idx --> set of objects with suppressed outputs, but still continue to be tracked
            }
            if self.masklet_confirmation_enable:
                # all the following are npt.NDArray with the same shape as `obj_ids_all_gpu`
                rank0_metadata["masklet_confirmation"] = {
                    # "status" is the confirmation status of each masklet (in `MaskletConfirmationStatus`)
                    "status": np.array([], np.int64),
                    # "consecutive_det_num" is the number of consecutive frames where the masklet is
                    # detected by the detector (with a matched detection)
                    "consecutive_det_num": np.array([], np.int64),
                }
            tracker_metadata["rank0_metadata"] = rank0_metadata

        return tracker_metadata

    def update_masklet_confirmation_status(
        self,
        rank0_metadata: Dict[str, Any],
        obj_ids_all_gpu_prev: npt.NDArray,
        obj_ids_all_gpu_updated: npt.NDArray,
        det_to_matched_trk_obj_ids: Dict[int, npt.NDArray],
        new_det_obj_ids: npt.NDArray,
    ):
        confirmation_data = rank0_metadata["masklet_confirmation"]

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
        unconfirmed_val = MaskletConfirmationStatus.UNCONFIRMED.value
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
        status[change_to_confirmed] = MaskletConfirmationStatus.CONFIRMED.value

        confirmation_data["status"] = status
        confirmation_data["consecutive_det_num"] = consecutive_det_num
        return rank0_metadata

    def _load_checkpoint(self, ckpt_path: str, strict: bool = True):
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        missing_keys, unexpected_keys = self.load_state_dict(sd, strict=strict)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            LOGGER.warning(f"Loaded ckpt with {missing_keys=}, {unexpected_keys=}")
        else:
            LOGGER.info("Loaded ckpt successfully without missing or unexpected keys")

    def _encode_prompt(self, **kwargs):
        return self.model._encode_prompt(**kwargs)

    def _drop_new_det_with_obj_limit(self, new_det_fa_inds, det_scores_np, num_to_keep):
        """
        Drop a few new detections based on the maximum number of objects. We drop new objects based
        on their detection scores, keeping the high-scoring ones and dropping the low-scoring ones.
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
