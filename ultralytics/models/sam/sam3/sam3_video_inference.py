# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
from collections import defaultdict

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from sam3 import perflib
from sam3.logger import get_logger
from sam3.model.act_ckpt_utils import clone_output_wrapper
from sam3.model.box_ops import box_xywh_to_cxcywh, box_xyxy_to_xywh
from sam3.model.data_misc import BatchedDatapoint, convert_my_tensors, FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model.io_utils import IMAGE_EXTS, load_resource_as_video_frames
from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.sam3_video_base import MaskletConfirmationStatus, Sam3VideoBase
from sam3.model.utils.misc import copy_data_to_device
from sam3.perflib.compile import compile_wrapper, shape_logging_wrapper
from sam3.perflib.masks_ops import masks_to_boxes as perf_masks_to_boxes
from torchvision.ops import masks_to_boxes
from tqdm.auto import tqdm

logger = get_logger(__name__)


class Sam3VideoInference(Sam3VideoBase):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1

    def __init__(
        self,
        image_size=1008,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
        compile_model=False,
        **kwargs,
    ):
        """
        hotstart_delay: int, the delay (in #frames) before the model starts to yield output, 0 to disable hotstart delay.
        hotstart_unmatch_thresh: int, remove the object if it has this many unmatched frames within its hotstart_delay period.
            If `hotstart_delay` is set to 0, this parameter is ignored.
        hotstart_dup_thresh: int, remove the object if it has overlapped with another object this many frames within its hotstart_delay period.
        """
        super().__init__(**kwargs)
        self.image_size = image_size
        self.image_mean = image_mean
        self.image_std = image_std
        self.compile_model = compile_model

    @torch.inference_mode()
    def init_state(
        self,
        resource_path,
        offload_video_to_cpu=False,
        async_loading_frames=False,
        video_loader_type="cv2",
    ):
        """Initialize an inference state from `resource_path` (an image or a video)."""
        images, orig_height, orig_width = load_resource_as_video_frames(
            resource_path=resource_path,
            image_size=self.image_size,
            offload_video_to_cpu=offload_video_to_cpu,
            img_mean=self.image_mean,
            img_std=self.image_std,
            async_loading_frames=async_loading_frames,
            video_loader_type=video_loader_type,
        )
        inference_state = {}
        inference_state["image_size"] = self.image_size
        inference_state["num_frames"] = len(images)
        # the original video height and width, used for resizing final output scores
        inference_state["orig_height"] = orig_height
        inference_state["orig_width"] = orig_width
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # inputs on each frame
        self._construct_initial_input_batch(inference_state, images)
        # initialize extra states
        inference_state["tracker_inference_states"] = []
        inference_state["tracker_metadata"] = {}
        inference_state["feature_cache"] = {}
        inference_state["cached_frame_outputs"] = {}
        inference_state["action_history"] = []  # for logging user actions
        inference_state["is_image_only"] = is_image_type(resource_path)
        return inference_state

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Revert `inference_state` to what it was right after initialization."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            # constructing an output list in inference state (we start with an empty list)
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_raw_point_input"][t] = None
            inference_state["per_frame_raw_box_input"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None
            inference_state["per_frame_cur_step"][t] = 0

        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None
        inference_state["tracker_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()
        inference_state["cached_frame_outputs"].clear()
        inference_state["action_history"].clear()  # for logging user actions

    def _construct_initial_input_batch(self, inference_state, images):
        """Construct an initial `BatchedDatapoint` instance as input."""
        # 1) img_batch
        num_frames = len(images)
        device = self.device

        # 2) find_text_batch
        # "<text placeholder>" will be replaced by the actual text prompt when adding prompts
        find_text_batch = ["<text placeholder>", "visual"]

        # 3) find_inputs
        input_box_embedding_dim = 258  # historical default
        input_points_embedding_dim = 257  # historical default
        stages = [
            FindStage(
                img_ids=[stage_id],
                text_ids=[0],
                input_boxes=[torch.zeros(input_box_embedding_dim)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool)],
                input_boxes_label=[torch.empty(0, dtype=torch.long)],
                input_points=[torch.empty(0, input_points_embedding_dim)],
                input_points_mask=[torch.empty(0)],
                object_ids=[],
            )
            for stage_id in range(num_frames)
        ]
        for i in range(len(stages)):
            stages[i] = convert_my_tensors(stages[i])

        # construct the final `BatchedDatapoint` and cast to GPU
        input_batch = BatchedDatapoint(
            img_batch=images,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * num_frames,
            find_metadatas=[None] * num_frames,
        )
        input_batch = copy_data_to_device(input_batch, device, non_blocking=True)
        inference_state["input_batch"] = input_batch

        # construct the placeholder interactive prompts and tracking queries
        bs = 1
        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, bs, 4, device=device),
            box_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            box_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
            point_embeddings=torch.zeros(0, bs, 2, device=device),
            point_mask=torch.zeros(bs, 0, device=device, dtype=torch.bool),
            point_labels=torch.zeros(0, bs, device=device, dtype=torch.long),
        )

        # constructing an output list in inference state (we start with an empty list)
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_raw_point_input"] = [None] * num_frames
        inference_state["per_frame_raw_box_input"] = [None] * num_frames
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames
        inference_state["per_frame_cur_step"] = [0] * num_frames

        # placeholders for cached outputs
        # (note: currently, a single visual prompt embedding is shared for all frames)
        inference_state["visual_prompt_embed"] = None
        inference_state["visual_prompt_mask"] = None

    def _get_visual_prompt(self, inference_state, frame_idx, boxes_cxcywh, box_labels):
        """
        Handle the case of visual prompt. Currently, in the inference API we do not
        explicitly distinguish between initial box as visual prompt vs subsequent boxes
        or boxes after inference for refinement.
        """
        # If the frame hasn't had any inference results before (prompting or propagation),
        # we treat the first added box prompt as a visual prompt; otherwise, we treat
        # the first box just as a refinement prompt.
        is_new_visual_prompt = (
            inference_state["per_frame_visual_prompt"][frame_idx] is None
            and inference_state["previous_stages_out"][frame_idx] is None
        )
        if is_new_visual_prompt:
            if boxes_cxcywh.size(0) != 1:
                raise RuntimeError(
                    "visual prompts (box as an initial prompt) should only have one box, "
                    f"but got {boxes_cxcywh.shape=}"
                )
            if not box_labels.item():
                logging.warning("A negative box is added as a visual prompt.")
            # take the first box prompt as a visual prompt
            device = self.device
            new_visual_prompt = Prompt(
                box_embeddings=boxes_cxcywh[None, 0:1, :].to(device),  # (seq, bs, 4)
                box_mask=None,
                box_labels=box_labels[None, 0:1].to(device),  # (seq, bs)
                point_embeddings=None,
                point_mask=None,
                point_labels=None,
            )
            inference_state["per_frame_visual_prompt"][frame_idx] = new_visual_prompt
        else:
            new_visual_prompt = None

        # `boxes_cxcywh` and `box_labels` contains all the raw box inputs added so far
        # strip any visual prompt from the input boxes (for geometric prompt encoding)
        if inference_state["per_frame_visual_prompt"][frame_idx] is not None:
            boxes_cxcywh = boxes_cxcywh[1:]
            box_labels = box_labels[1:]

        return boxes_cxcywh, box_labels, new_visual_prompt

    def _get_processing_order(
        self, inference_state, start_frame_idx, max_frame_num_to_track, reverse
    ):
        num_frames = inference_state["num_frames"]
        previous_stages_out = inference_state["previous_stages_out"]
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at least one frame before propagation."
            )
        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(
                t for t, out in enumerate(previous_stages_out) if out is not None
            )
        if max_frame_num_to_track is None:
            # default: track all the frames in the video
            max_frame_num_to_track = num_frames
        if reverse:
            end_frame_idx = start_frame_idx - max_frame_num_to_track
            end_frame_idx = max(end_frame_idx, 0)
            processing_order = range(start_frame_idx - 1, end_frame_idx - 1, -1)
        else:
            end_frame_idx = start_frame_idx + max_frame_num_to_track
            end_frame_idx = min(end_frame_idx, num_frames - 1)
            processing_order = range(start_frame_idx, end_frame_idx + 1)
        return processing_order, end_frame_idx

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        """
        Propagate the prompts to get grounding results for the entire video. This method
        is a generator and yields inference outputs for all frames in the range specified
        by `start_frame_idx`, `max_frame_num_to_track`, and `reverse`.
        """
        # compile the model (it's a no-op if the model is already compiled)
        # note that it's intentionally added to `self.propagate_in_video`, so that the first
        # `self.add_prompt` call will be done in eager mode to fill in the decoder buffers
        # such as positional encoding cache)
        self._compile_model()

        processing_order, end_frame_idx = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )

        # Store max_frame_num_to_track in feature_cache for downstream methods
        inference_state["feature_cache"]["tracking_bounds"] = {
            "max_frame_num_to_track": max_frame_num_to_track,
            "propagate_in_video_start_frame_idx": start_frame_idx,
        }

        hotstart_buffer = []
        hotstart_removed_obj_ids = set()
        # when deciding whether to output a masklet on `yield_frame_idx`, we check whether the object is confirmed
        # in a future frame (`unconfirmed_frame_delay` frames after the current frame). For example, if we require
        # an object to be detected in 3 consecutive frames to be confirmed, then we look 2 frames in the future --
        # e.g., we output an object on frame 4 only if it becomes confirmed on frame 6.
        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1
        unconfirmed_obj_ids_per_frame = {}  # frame_idx -> hidden_obj_ids
        for frame_idx in tqdm(
            processing_order, desc="propagate_in_video", disable=self.rank > 0
        ):
            out = self._run_single_frame_inference(inference_state, frame_idx, reverse)

            if self.hotstart_delay > 0:
                # accumulate the outputs for the first `hotstart_delay` frames
                hotstart_buffer.append([frame_idx, out])
                # update the object IDs removed by hotstart so that we don't output them
                if self.rank == 0:
                    hotstart_removed_obj_ids.update(out["removed_obj_ids"])
                    unconfirmed_obj_ids = out.get("unconfirmed_obj_ids", None)
                    if unconfirmed_obj_ids is not None:
                        unconfirmed_obj_ids_per_frame[frame_idx] = unconfirmed_obj_ids

                if frame_idx == end_frame_idx:
                    # we reached the end of propagation -- yield all frames in the buffer
                    yield_list = hotstart_buffer
                    hotstart_buffer = []
                elif len(hotstart_buffer) >= self.hotstart_delay:
                    # we have enough frames -- yield and remove the first (oldest) frame from the buffer
                    yield_list = hotstart_buffer[:1]
                    hotstart_buffer = hotstart_buffer[1:]
                else:
                    # not enough frames yet -- skip yielding
                    yield_list = []
            else:
                yield_list = [(frame_idx, out)]  # output the current frame

            for yield_frame_idx, yield_out in yield_list:
                # post-process the output and yield it
                if self.rank == 0:
                    suppressed_obj_ids = yield_out["suppressed_obj_ids"]
                    unconfirmed_status_frame_idx = (
                        yield_frame_idx + unconfirmed_status_delay
                        if not reverse
                        else yield_frame_idx - unconfirmed_status_delay
                    )

                    # Clamp the frame index to stay within video bounds
                    num_frames = inference_state["num_frames"]
                    unconfirmed_status_frame_idx = max(
                        0, min(unconfirmed_status_frame_idx, num_frames - 1)
                    )

                    unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(
                        unconfirmed_status_frame_idx, None
                    )
                    postprocessed_out = self._postprocess_output(
                        inference_state,
                        yield_out,
                        hotstart_removed_obj_ids,
                        suppressed_obj_ids,
                        unconfirmed_obj_ids,
                    )

                    self._cache_frame_outputs(
                        inference_state,
                        yield_frame_idx,
                        yield_out["obj_id_to_mask"],
                        suppressed_obj_ids=suppressed_obj_ids,
                        removed_obj_ids=hotstart_removed_obj_ids,
                        unconfirmed_obj_ids=unconfirmed_obj_ids,
                    )
                else:
                    postprocessed_out = None  # no output on other GPUs
                yield yield_frame_idx, postprocessed_out

    def _run_single_frame_inference(self, inference_state, frame_idx, reverse):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        # prepare inputs
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = (
            inference_state["per_frame_geometric_prompt"][frame_idx] is not None
        )
        # run inference for the current frame
        (
            obj_id_to_mask,
            obj_id_to_score,
            tracker_states_local_new,
            tracker_metadata_new,
            frame_stats,
            _,
        ) = self._det_track_one_frame(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            reverse=reverse,
            input_batch=input_batch,
            geometric_prompt=(
                inference_state["constants"]["empty_geometric_prompt"]
                if not has_geometric_prompt
                else inference_state["per_frame_geometric_prompt"][frame_idx]
            ),
            tracker_states_local=tracker_states_local,
            tracker_metadata_prev=inference_state["tracker_metadata"],
            feature_cache=inference_state["feature_cache"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            is_image_only=inference_state["is_image_only"],
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        # update inference state
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        # use a dummy string in "previous_stages_out" to indicate this frame has outputs
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        if self.rank == 0:
            self._cache_frame_outputs(inference_state, frame_idx, obj_id_to_mask)

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # first frame detection score
            "obj_id_to_tracker_score": tracker_metadata_new[
                "obj_id_to_tracker_score_frame_wise"
            ][frame_idx],
        }
        # removed_obj_ids is only needed on rank 0 to handle hotstart delay buffer
        if self.rank == 0:
            rank0_metadata = tracker_metadata_new["rank0_metadata"]
            removed_obj_ids = rank0_metadata["removed_obj_ids"]
            out["removed_obj_ids"] = removed_obj_ids
            out["suppressed_obj_ids"] = rank0_metadata["suppressed_obj_ids"][frame_idx]
            out["frame_stats"] = frame_stats
            if self.masklet_confirmation_enable:
                status = rank0_metadata["masklet_confirmation"]["status"]
                is_unconfirmed = status == MaskletConfirmationStatus.UNCONFIRMED.value
                out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][
                    is_unconfirmed
                ].tolist()
            else:
                out["unconfirmed_obj_ids"] = []

        return out

    def _postprocess_output(
        self,
        inference_state,
        out,
        removed_obj_ids=None,
        suppressed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        obj_id_to_mask = out["obj_id_to_mask"]  # low res masks
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        H_video, W_video = inference_state["orig_height"], inference_state["orig_width"]
        if len(curr_obj_ids) == 0:
            out_obj_ids = torch.zeros(0, dtype=torch.int64)
            out_probs = torch.zeros(0, dtype=torch.float32)
            out_binary_masks = torch.zeros(0, H_video, W_video, dtype=torch.bool)
            out_boxes_xywh = torch.zeros(0, 4, dtype=torch.float32)
        else:
            out_obj_ids = torch.tensor(curr_obj_ids, dtype=torch.int64)
            out_probs = torch.tensor(
                [out["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids]
            )
            out_tracker_probs = torch.tensor(
                [
                    (
                        out["obj_id_to_tracker_score"][obj_id]
                        if obj_id in out["obj_id_to_tracker_score"]
                        else 0.0
                    )
                    for obj_id in curr_obj_ids
                ]
            )
            out_binary_masks = torch.cat(
                [obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0
            )

            assert out_binary_masks.dtype == torch.bool
            keep = out_binary_masks.any(dim=(1, 2)).cpu()  # remove masks with 0 areas
            # hide outputs for those object IDs in `obj_ids_to_hide`
            obj_ids_to_hide = []
            if suppressed_obj_ids is not None:
                obj_ids_to_hide.extend(suppressed_obj_ids)
            if removed_obj_ids is not None:
                obj_ids_to_hide.extend(removed_obj_ids)
            if unconfirmed_obj_ids is not None:
                obj_ids_to_hide.extend(unconfirmed_obj_ids)
            if len(obj_ids_to_hide) > 0:
                obj_ids_to_hide_t = torch.tensor(obj_ids_to_hide, dtype=torch.int64)
                keep &= ~torch.isin(out_obj_ids, obj_ids_to_hide_t)

            # slice those valid entries from the original outputs
            keep_idx = torch.nonzero(keep, as_tuple=True)[0]
            keep_idx_gpu = keep_idx.pin_memory().to(
                device=out_binary_masks.device, non_blocking=True
            )

            out_obj_ids = torch.index_select(out_obj_ids, 0, keep_idx)
            out_probs = torch.index_select(out_probs, 0, keep_idx)
            out_tracker_probs = torch.index_select(out_tracker_probs, 0, keep_idx)
            out_binary_masks = torch.index_select(out_binary_masks, 0, keep_idx_gpu)

            if perflib.is_enabled:
                out_boxes_xyxy = perf_masks_to_boxes(
                    out_binary_masks, out_obj_ids.tolist()
                )
            else:
                out_boxes_xyxy = masks_to_boxes(out_binary_masks)

            out_boxes_xywh = box_xyxy_to_xywh(out_boxes_xyxy)  # convert to xywh format
            # normalize boxes
            out_boxes_xywh[..., 0] /= W_video
            out_boxes_xywh[..., 1] /= H_video
            out_boxes_xywh[..., 2] /= W_video
            out_boxes_xywh[..., 3] /= H_video

        # apply non-overlapping constraints on the existing masklets
        if out_binary_masks.shape[0] > 1:
            assert len(out_binary_masks) == len(out_tracker_probs)
            out_binary_masks = (
                self.tracker._apply_object_wise_non_overlapping_constraints(
                    out_binary_masks.unsqueeze(1),
                    out_tracker_probs.unsqueeze(1).to(out_binary_masks.device),
                    background_value=0,
                ).squeeze(1)
            ) > 0

        outputs = {
            "out_obj_ids": out_obj_ids.cpu().numpy(),
            "out_probs": out_probs.cpu().numpy(),
            "out_boxes_xywh": out_boxes_xywh.cpu().numpy(),
            "out_binary_masks": out_binary_masks.cpu().numpy(),
            "frame_stats": out.get("frame_stats", None),
        }
        return outputs

    def _cache_frame_outputs(
        self,
        inference_state,
        frame_idx,
        obj_id_to_mask,
        suppressed_obj_ids=None,
        removed_obj_ids=None,
        unconfirmed_obj_ids=None,
    ):
        # Filter out suppressed, removed, and unconfirmed objects from the cache
        filtered_obj_id_to_mask = obj_id_to_mask.copy()

        objects_to_exclude = set()
        if suppressed_obj_ids is not None:
            objects_to_exclude.update(suppressed_obj_ids)
        if removed_obj_ids is not None:
            objects_to_exclude.update(removed_obj_ids)
        if unconfirmed_obj_ids is not None:
            objects_to_exclude.update(unconfirmed_obj_ids)

        if objects_to_exclude:
            for obj_id in objects_to_exclude:
                if obj_id in filtered_obj_id_to_mask:
                    del filtered_obj_id_to_mask[obj_id]

        inference_state["cached_frame_outputs"][frame_idx] = filtered_obj_id_to_mask

    def _build_tracker_output(
        self, inference_state, frame_idx, refined_obj_id_to_mask=None
    ):
        assert (
            "cached_frame_outputs" in inference_state
            and frame_idx in inference_state["cached_frame_outputs"]
        ), "No cached outputs found. Ensure normal propagation has run first to populate the cache."
        cached_outputs = inference_state["cached_frame_outputs"][frame_idx]

        obj_id_to_mask = cached_outputs.copy()

        # Update with refined masks if provided
        if refined_obj_id_to_mask is not None:
            for obj_id, refined_mask in refined_obj_id_to_mask.items():
                assert (
                    refined_mask is not None
                ), f"Refined mask data must be provided for obj_id {obj_id}"
                obj_id_to_mask[obj_id] = refined_mask

        return obj_id_to_mask

    def _compile_model(self):
        """Compile the SAM model with torch.compile for speedup."""
        is_compiled = getattr(self, "_model_is_compiled", False)
        if is_compiled or not self.compile_model:
            return

        import torch._dynamo

        # a larger cache size to hold varying number of shapes for torch.compile
        # see https://github.com/pytorch/pytorch/blob/v2.5.1/torch/_dynamo/config.py#L42-L49
        torch._dynamo.config.cache_size_limit = 128
        torch._dynamo.config.accumulated_cache_size_limit = 2048
        torch._dynamo.config.capture_scalar_outputs = True
        torch._dynamo.config.suppress_errors = True

        # Compile module components
        # skip compilation of `_encode_prompt` since it sometimes tiggger SymInt errors
        # self._encode_prompt = clone_output_wrapper(
        #     torch.compile(self._encode_prompt, fullgraph=True, mode="max-autotune")
        # )

        ## Compile SAM3 model components
        self.detector.backbone.vision_backbone.forward = clone_output_wrapper(
            torch.compile(
                self.detector.backbone.vision_backbone.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.detector.transformer.encoder.forward = clone_output_wrapper(
            torch.compile(
                self.detector.transformer.encoder.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )
        self.detector.transformer.decoder.forward = clone_output_wrapper(
            torch.compile(
                self.detector.transformer.decoder.forward,
                fullgraph=True,
                mode="max-autotune",
                dynamic=False,
            )
        )

        self.detector.segmentation_head.forward = clone_output_wrapper(
            torch.compile(
                self.detector.segmentation_head.forward,
                fullgraph=True,
                mode="max-autotune",
            )
        )

        ## Compile Tracker model components
        self.tracker.maskmem_backbone.forward = compile_wrapper(
            self.tracker.maskmem_backbone.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,
        )

        self.tracker.transformer.encoder.forward = shape_logging_wrapper(
            compile_wrapper(
                self.tracker.transformer.encoder.forward,
                mode="max-autotune-no-cudagraphs",
                fullgraph=True,
                dynamic=True,
            ),
            keep_kwargs=["src", "src_pos", "prompt", "prompt_pos"],
        )

        self.tracker.sam_mask_decoder.forward = compile_wrapper(
            self.tracker.sam_mask_decoder.forward,
            mode="max-autotune",
            fullgraph=True,
            dynamic=False,  # Accuracy regression on True
        )

        self._model_is_compiled = True

    def _warm_up_vg_propagation(self, inference_state, start_frame_idx=0):
        # use different tracking score thresholds for each round to simulate different number of output objects
        num_objects_list = range(self.num_obj_for_compile + 1)
        new_det_score_thresh_list = [0.3, 0.5, 0.7]
        num_rounds = len(new_det_score_thresh_list)
        orig_new_det_thresh = self.new_det_thresh

        for i, thresh in enumerate(new_det_score_thresh_list):
            self.new_det_thresh = thresh
            for num_objects in num_objects_list:
                logger.info(f"{i+1}/{num_rounds} warming up model compilation")
                self.add_prompt(
                    inference_state, frame_idx=start_frame_idx, text_str="cat"
                )
                logger.info(
                    f"{i+1}/{num_rounds} warming up model compilation -- simulating {num_objects}/{self.num_obj_for_compile} objects"
                )
                inference_state = self.add_fake_objects_to_inference_state(
                    inference_state, num_objects, frame_idx=start_frame_idx
                )
                inference_state["tracker_metadata"]["rank0_metadata"].update(
                    {
                        "masklet_confirmation": {
                            "status": np.zeros(num_objects, dtype=np.int64),
                            "consecutive_det_num": np.zeros(
                                num_objects, dtype=np.int64
                            ),
                        }
                    }
                )
                for _ in self.propagate_in_video(
                    inference_state, start_frame_idx, reverse=False
                ):
                    pass
                for _ in self.propagate_in_video(
                    inference_state, start_frame_idx, reverse=True
                ):
                    pass
                self.reset_state(inference_state)
                logger.info(
                    f"{i+1}/{num_rounds} warming up model compilation -- completed round {i+1} out of {num_rounds}"
                )

        # Warm up Tracker memory encoder with varying input shapes
        num_iters = 3
        feat_size = self.tracker.sam_image_embedding_size**2  # 72 * 72 = 5184
        hidden_dim = self.tracker.hidden_dim  # 256
        mem_dim = self.tracker.mem_dim  # 64
        for _ in tqdm(range(num_iters)):
            for b in range(1, self.num_obj_for_compile + 1):
                for i in range(
                    1,
                    self.tracker.max_cond_frames_in_attn + self.tracker.num_maskmem,
                ):
                    for j in range(
                        self.tracker.max_cond_frames_in_attn
                        + self.tracker.max_obj_ptrs_in_encoder
                    ):
                        num_obj_ptr_tokens = (hidden_dim // mem_dim) * j
                        src = torch.randn(feat_size, b, hidden_dim, device=self.device)
                        src_pos = torch.randn(
                            feat_size, b, hidden_dim, device=self.device
                        )
                        prompt = torch.randn(
                            feat_size * i + num_obj_ptr_tokens,
                            b,
                            mem_dim,
                            device=self.device,
                        )
                        prompt_pos = torch.randn(
                            feat_size * i + num_obj_ptr_tokens,
                            b,
                            mem_dim,
                            device=self.device,
                        )

                        self.tracker.transformer.encoder.forward(
                            src=src,
                            src_pos=src_pos,
                            prompt=prompt,
                            prompt_pos=prompt_pos,
                            num_obj_ptr_tokens=num_obj_ptr_tokens,
                        )

        self.new_det_thresh = orig_new_det_thresh
        return inference_state

    def add_fake_objects_to_inference_state(
        self, inference_state, num_objects, frame_idx
    ):
        new_det_obj_ids_local = np.arange(num_objects)
        high_res_H, high_res_W = (
            self.tracker.maskmem_backbone.mask_downsampler.interpol_size
        )
        new_det_masks = torch.ones(
            len(new_det_obj_ids_local), high_res_H, high_res_W
        ).to(self.device)

        inference_state["tracker_inference_states"] = self._tracker_add_new_objects(
            frame_idx=frame_idx,
            num_frames=inference_state["num_frames"],
            new_obj_ids=new_det_obj_ids_local,
            new_obj_masks=new_det_masks,
            tracker_states_local=inference_state["tracker_inference_states"],
            orig_vid_height=inference_state["orig_height"],
            orig_vid_width=inference_state["orig_width"],
            feature_cache=inference_state["feature_cache"],
        )

        # Synthesize obj_id_to_mask data for cached_frame_outputs to support _build_tracker_output during warmup
        obj_id_to_mask = {}
        if num_objects > 0:
            H_video = inference_state["orig_height"]
            W_video = inference_state["orig_width"]

            video_res_masks = F.interpolate(
                new_det_masks.unsqueeze(1),  # Add channel dimension for interpolation
                size=(H_video, W_video),
                mode="bilinear",
                align_corners=False,
            )  # (num_objects, 1, H_video, W_video)
            for i, obj_id in enumerate(new_det_obj_ids_local):
                obj_id_to_mask[obj_id] = (video_res_masks[i] > 0.0).to(torch.bool)
        if self.rank == 0:
            for fidx in range(inference_state["num_frames"]):
                self._cache_frame_outputs(inference_state, fidx, obj_id_to_mask)

        inference_state["tracker_metadata"].update(
            {
                "obj_ids_per_gpu": [np.arange(num_objects)],
                "obj_ids_all_gpu": np.arange(num_objects),  # Same as 1 GPU
                "num_obj_per_gpu": [num_objects],
                "obj_id_to_score": {i: 1.0 for i in range(num_objects)},
                "max_obj_id": num_objects,
                "rank0_metadata": {
                    "masklet_confirmation": {
                        "status": np.zeros(num_objects, dtype=np.int64),
                        "consecutive_det_num": np.zeros(num_objects, dtype=np.int64),
                    },
                    "removed_obj_ids": set(),
                    "suppressed_obj_ids": defaultdict(set),
                },
            }
        )
        return inference_state

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def warm_up_compilation(self):
        """
        Warm up the model by running a dummy inference to compile the model. This is
        useful to avoid the compilation overhead in the first inference call.
        """
        if not self.compile_model:
            return
        self._warm_up_complete = False
        if self.device.type != "cuda":
            raise RuntimeError(
                f"The model must be on CUDA for warm-up compilation, got {self.device=}."
            )

        # temporally set to single GPU temporarily for warm-up compilation
        orig_rank = self.rank
        orig_world_size = self.world_size
        self.rank = self.detector.rank = 0
        self.world_size = self.detector.world_size = 1
        orig_recondition_every_nth_frame = self.recondition_every_nth_frame
        # self.recondition_every_nth_frame = 2

        # Get a random video
        inference_state = self.init_state(resource_path="<load-dummy-video-30>")
        start_frame_idx = 0

        # Run basic propagation warm-up
        inference_state = self._warm_up_vg_propagation(inference_state, start_frame_idx)

        logger.info("Warm-up compilation completed.")

        # revert to the original GPU and rank
        self.rank = self.detector.rank = orig_rank
        self.world_size = self.detector.world_size = orig_world_size
        self.recondition_every_nth_frame = orig_recondition_every_nth_frame
        self._warm_up_complete = True
        self.tracker.transformer.encoder.forward.set_logging(True)

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        boxes_xywh=None,
        box_labels=None,
    ):
        """
        Add text, point or box prompts on a single frame. This method returns the inference
        outputs only on the prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.
        """
        logger.debug("Running add_prompt on frame %d", frame_idx)

        num_frames = inference_state["num_frames"]
        assert (
            text_str is not None or boxes_xywh is not None
        ), "at least one type of prompt (text, boxes) must be provided"
        assert (
            0 <= frame_idx < num_frames
        ), f"{frame_idx=} is out of range for a total of {num_frames} frames"

        # since it's a semantic prompt, we start over
        self.reset_state(inference_state)

        # 1) add text prompt
        if text_str is not None and text_str != "visual":
            inference_state["text_prompt"] = text_str
            inference_state["input_batch"].find_text_batch[0] = text_str
            text_id = self.TEXT_ID_FOR_TEXT
        else:
            inference_state["text_prompt"] = None
            inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
            text_id = self.TEXT_ID_FOR_VISUAL
        for t in range(inference_state["num_frames"]):
            inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # 2) handle box prompt
        assert (boxes_xywh is not None) == (box_labels is not None)
        if boxes_xywh is not None:
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=torch.float32)
            box_labels = torch.as_tensor(box_labels, dtype=torch.long)
            # input boxes are expected to be [xmin, ymin, width, height] format
            # in normalized coordinates of range 0~1, similar to FA
            assert boxes_xywh.dim() == 2
            assert boxes_xywh.size(0) > 0 and boxes_xywh.size(-1) == 4
            assert box_labels.dim() == 1 and box_labels.size(0) == boxes_xywh.size(0)
            boxes_cxcywh = box_xywh_to_cxcywh(boxes_xywh)
            assert (boxes_xywh >= 0).all().item() and (boxes_xywh <= 1).all().item()
            assert (boxes_cxcywh >= 0).all().item() and (boxes_cxcywh <= 1).all().item()

            new_box_input = boxes_cxcywh, box_labels
            inference_state["per_frame_raw_box_input"][frame_idx] = new_box_input

            # handle the case of visual prompt (also added as an input box from the UI)
            boxes_cxcywh, box_labels, geometric_prompt = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )

            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        out = self._run_single_frame_inference(
            inference_state, frame_idx, reverse=False
        )
        return frame_idx, self._postprocess_output(inference_state, out)

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def forward(self, input: BatchedDatapoint, is_inference: bool = False):
        """This method is only used for benchmark eval (not used in the demo)."""
        # set the model to single GPU for benchmark evaluation (to be compatible with trainer)
        orig_rank = self.rank
        orig_world_size = self.world_size
        self.rank = self.detector.rank = 0
        self.world_size = self.detector.world_size = 1

        # get data
        text_prompt_ids = input.find_metadatas[0].original_category_id
        text_prompt_list = input.find_text_batch

        # loop over txt prompts
        tracking_res = defaultdict(dict)  # frame_idx --> {obj_id: mask}
        scores_labels = defaultdict(tuple)  # obj_id --> (score, text_prompt_id)
        inference_state = self.init_state(resource_path=input.raw_images)
        for prompt_id, prompt in zip(text_prompt_ids, text_prompt_list):
            self.add_prompt(inference_state, frame_idx=0, text_str=prompt)
            start_obj_id = max(scores_labels.keys(), default=-1) + 1  # prev max + 1

            # propagate the prompts
            obj_ids_this_prompt = set()
            for frame_idx, out in self.propagate_in_video(
                inference_state,
                start_frame_idx=0,
                max_frame_num_to_track=inference_state["num_frames"],
                reverse=False,
            ):
                current_frame_res = tracking_res[frame_idx]
                for obj_id, mask in zip(out["out_obj_ids"], out["out_binary_masks"]):
                    mask_tensor = torch.tensor(mask[None], dtype=torch.bool)
                    current_frame_res[obj_id + start_obj_id] = mask_tensor
                obj_ids_this_prompt.update(current_frame_res.keys())

            obj_id_to_score = inference_state["tracker_metadata"]["obj_id_to_score"]
            for obj_id, score in obj_id_to_score.items():
                if obj_id + start_obj_id in obj_ids_this_prompt:
                    score_tensor = torch.tensor(score, dtype=torch.float32)
                    scores_labels[obj_id + start_obj_id] = (score_tensor, prompt_id)

            self.reset_state(inference_state)

        video_id = input.find_metadatas[0].original_image_id[0].cpu().item()
        preds = self.prep_for_evaluator(input.raw_images, tracking_res, scores_labels)

        # revert the model to the original GPU and rank
        self.rank = self.detector.rank = orig_rank
        self.world_size = self.detector.world_size = orig_world_size
        return {video_id: preds}

    def back_convert(self, targets):
        # Needed for retraining compatibility with trainer
        return targets


class Sam3VideoInferenceWithInstanceInteractivity(Sam3VideoInference):
    def __init__(
        self,
        use_prev_mem_frame=False,
        use_stateless_refinement=False,
        refinement_detector_cond_frame_removal_window=16,
        **kwargs,
    ):
        """
        use_prev_mem_frame: bool, whether to condition on previous memory frames for adding points
        use_stateless_refinement: bool, whether to enable stateless refinement behavior
        refinement_detector_cond_frame_removal_window: int, we remove a detector conditioning frame if it
            is within this many frames of a user refined frame. Set to a large value (e.g. 10000) to
            always remove detector conditioning frames if there is any user refinement in the video.
        """
        super().__init__(**kwargs)
        self.use_prev_mem_frame = use_prev_mem_frame
        self.use_stateless_refinement = use_stateless_refinement
        self.refinement_detector_cond_frame_removal_window = (
            refinement_detector_cond_frame_removal_window
        )

    def _init_new_tracker_state(self, inference_state):
        return self.tracker.init_state(
            cached_features=inference_state["feature_cache"],
            video_height=inference_state["orig_height"],
            video_width=inference_state["orig_width"],
            num_frames=inference_state["num_frames"],
        )

    @torch.inference_mode()
    def propagate_in_video(
        self,
        inference_state,
        start_frame_idx=None,
        max_frame_num_to_track=None,
        reverse=False,
    ):
        # step 1: check which type of propagation to run, should be the same for all GPUs.
        propagation_type, obj_ids = self.parse_action_history_for_propagation(
            inference_state
        )
        self.add_action_history(
            inference_state,
            action_type=propagation_type,
            obj_ids=obj_ids,
            frame_idx=start_frame_idx,
        )

        # step 2: run full VG propagation
        if propagation_type == "propagation_full":
            logger.debug(f"Running full VG propagation (reverse={reverse}).")
            yield from super().propagate_in_video(
                inference_state,
                start_frame_idx=start_frame_idx,
                max_frame_num_to_track=max_frame_num_to_track,
                reverse=reverse,
            )
            return

        # step 3: run Tracker partial propagation or direct fetch existing predictions
        assert propagation_type in ["propagation_partial", "propagation_fetch"]
        logger.debug(
            f"Running Tracker propagation for objects {obj_ids} and merging it with existing VG predictions (reverse={reverse})."
            if propagation_type == "propagation_partial"
            else f"Fetching existing VG predictions without running any propagation (reverse={reverse})."
        )
        processing_order, _ = self._get_processing_order(
            inference_state,
            start_frame_idx=start_frame_idx,
            max_frame_num_to_track=max_frame_num_to_track,
            reverse=reverse,
        )

        tracker_metadata = inference_state["tracker_metadata"]

        # if fetch just return from output
        if propagation_type == "propagation_fetch":
            for frame_idx in tqdm(processing_order):
                if self.rank == 0:
                    obj_id_to_mask = inference_state["cached_frame_outputs"].get(
                        frame_idx, {}
                    )
                    # post processing - remove suppressed obj_ids
                    obj_id_to_score = tracker_metadata["obj_id_to_score"]
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    obj_id_to_tracker_score = tracker_metadata[
                        "obj_id_to_tracker_score_frame_wise"
                    ][frame_idx]

                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": obj_id_to_score,
                        "obj_id_to_tracker_score": obj_id_to_tracker_score,
                    }
                    yield (
                        frame_idx,
                        self._postprocess_output(
                            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
                        ),
                    )
                else:
                    yield frame_idx, None

            return

        # get Tracker inference states containing selected obj_ids
        if propagation_type == "propagation_partial":
            # can be empty for GPUs where objects are not in their inference states
            tracker_states_local = self._get_tracker_inference_states_by_obj_ids(
                inference_state, obj_ids
            )
            for tracker_state in tracker_states_local:
                self.tracker.propagate_in_video_preflight(
                    tracker_state, run_mem_encoder=True
                )

        for frame_idx in tqdm(processing_order):
            # run Tracker propagation
            if propagation_type == "propagation_partial":
                self._prepare_backbone_feats(inference_state, frame_idx, reverse)
                obj_ids_local, low_res_masks_local, tracker_scores_local = (
                    self._propogate_tracker_one_frame_local_gpu(
                        tracker_states_local,
                        frame_idx=frame_idx,
                        reverse=reverse,
                        run_mem_encoder=True,
                    )
                )

                # broadcast refined object tracker scores and masks to all GPUs
                # handle multiple objects that can be located on different GPUs
                refined_obj_data = {}  # obj_id -> (score, mask_video_res)

                # Collect data for objects on this GPU
                local_obj_data = {}
                for obj_id in obj_ids:
                    obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                    if self.rank == obj_rank and obj_id in obj_ids_local:
                        refined_obj_idx = obj_ids_local.index(obj_id)
                        refined_mask_low_res = low_res_masks_local[
                            refined_obj_idx
                        ]  # (H_low_res, W_low_res)
                        refined_score = tracker_scores_local[refined_obj_idx]

                        # Keep low resolution for broadcasting to reduce communication cost
                        local_obj_data[obj_id] = (refined_score, refined_mask_low_res)

                # Broadcast data from each GPU that has refined objects
                if self.world_size > 1:
                    for obj_id in obj_ids:
                        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
                        if self.rank == obj_rank:
                            # This GPU has the object, broadcast its data
                            data_to_broadcast = local_obj_data.get(obj_id, None)
                            data_list = [
                                (data_to_broadcast[0].cpu(), data_to_broadcast[1].cpu())
                            ]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            if data_to_broadcast is not None:
                                refined_obj_data[obj_id] = data_to_broadcast
                        elif self.rank != obj_rank:
                            # This GPU doesn't have the object, receive data
                            data_list = [None]
                            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
                            refined_obj_data[obj_id] = (
                                data_list[0][0].to(self.device),
                                data_list[0][1].to(self.device),
                            )
                else:
                    # Single GPU case
                    refined_obj_data = local_obj_data

                # Update Tracker scores for all refined objects
                for obj_id, (refined_score, _) in refined_obj_data.items():
                    tracker_metadata["obj_id_to_tracker_score_frame_wise"][
                        frame_idx
                    ].update({obj_id: refined_score.item()})

                if self.rank == 0:
                    # get predictions from Tracker inference states, it includes the original
                    # VG predictions and the refined predictions from interactivity.

                    # Prepare refined masks dictionary - upscale to video resolution after broadcast
                    refined_obj_id_to_mask = {}
                    for obj_id, (_, refined_mask_low_res) in refined_obj_data.items():
                        refined_mask_video_res = (
                            self._convert_low_res_mask_to_video_res(
                                refined_mask_low_res, inference_state
                            )
                        )  # (1, H_video, W_video) bool
                        refined_obj_id_to_mask[obj_id] = refined_mask_video_res

                    obj_id_to_mask = self._build_tracker_output(
                        inference_state, frame_idx, refined_obj_id_to_mask
                    )
                    out = {
                        "obj_id_to_mask": obj_id_to_mask,
                        "obj_id_to_score": tracker_metadata["obj_id_to_score"],
                        "obj_id_to_tracker_score": tracker_metadata[
                            "obj_id_to_tracker_score_frame_wise"
                        ][frame_idx],
                    }
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    self._cache_frame_outputs(
                        inference_state,
                        frame_idx,
                        obj_id_to_mask,
                        suppressed_obj_ids=suppressed_obj_ids,
                    )
                    suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                        "suppressed_obj_ids"
                    ][frame_idx]
                    yield (
                        frame_idx,
                        self._postprocess_output(
                            inference_state, out, suppressed_obj_ids=suppressed_obj_ids
                        ),
                    )
                else:
                    yield frame_idx, None

    def add_action_history(
        self, inference_state, action_type, frame_idx=None, obj_ids=None
    ):
        """
        action_history is used to automatically decide what to do during propagation.
        action_type: one of ["add", "remove", "refine"] + ["propagation_full", "propagation_partial", "propagation_fetch"]
        """
        instance_actions = ["add", "remove", "refine"]
        propagation_actions = [
            "propagation_full",
            "propagation_partial",
            "propagation_fetch",
        ]
        assert (
            action_type in instance_actions + propagation_actions
        ), f"Invalid action type: {action_type}, must be one of {instance_actions + propagation_actions}"
        action = {
            "type": action_type,
            "frame_idx": frame_idx,
            "obj_ids": obj_ids,
        }
        inference_state["action_history"].append(action)

    def _has_object_been_refined(self, inference_state, obj_id):
        action_history = inference_state["action_history"]
        for action in action_history:
            if action["type"] in ["add", "refine"] and action.get("obj_ids"):
                if obj_id in action["obj_ids"]:
                    return True
        return False

    def parse_action_history_for_propagation(self, inference_state):
        """
        Parse the actions in history before the last propagation and prepare for the next propagation.
        We support multiple actions (add/remove/refine) between two propagations. If we had an action
        history similar to this ["propagate", "add", "refine", "remove", "add"], the next propagation
        would remove the removed object, and also propagate the two added/refined objects.

        Returns:
            propagation_type: one of ["propagation_full", "propagation_partial", "propagation_fetch"]
                - "propagation_full": run VG propagation for all objects
                - "propagation_partial": run Tracker propagation for selected objects, useful for add/refine actions
                - "propagation_fetch": fetch existing VG predictions without running any propagation
            obj_ids: list of object ids to run Tracker propagation on if propagation_type is "propagation_partial".
        """
        action_history = inference_state["action_history"]
        if len(action_history) == 0:
            # we run propagation for the first time
            return "propagation_full", None

        if "propagation" in action_history[-1]["type"]:
            if action_history[-1]["type"] in ["propagation_fetch"]:
                # last propagation is direct fetch, we fetch existing predictions
                return "propagation_fetch", None
            elif action_history[-1]["type"] in [
                "propagation_partial",
                "propagation_full",
            ]:
                # we do fetch prediction if we have already run propagation twice or we have run
                # propagation once and it is from the first frame or last frame.
                if (
                    len(action_history) > 1
                    and action_history[-2]["type"]
                    in ["propagation_partial", "propagation_full"]
                ) or action_history[-1]["frame_idx"] in [
                    0,
                    inference_state["num_frames"] - 1,
                ]:
                    # we have run both forward and backward partial/full propagation
                    return "propagation_fetch", None
                else:
                    # we have run partial/full forward or backward propagation once, need run it for the rest of the frames
                    return action_history[-1]["type"], action_history[-1]["obj_ids"]

        # parse actions since last propagation
        obj_ids = []
        for action in action_history[::-1]:
            if "propagation" in action["type"]:
                # we reached the last propagation action, stop parsing
                break
            if action["type"] in ["add", "refine"]:
                obj_ids.extend(action["obj_ids"])
            # else action["type"] == "remove": noop
        obj_ids = list(set(obj_ids)) if len(obj_ids) > 0 else None
        propagation_type = (
            "propagation_partial" if obj_ids is not None else "propagation_fetch"
        )
        return propagation_type, obj_ids

    def remove_object(self, inference_state, obj_id, is_user_action=False):
        """
        We try to remove object from tracker states on every GPU, it will do nothing
        for states without this object.
        """
        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)
        assert obj_rank is not None, f"Object {obj_id} not found in any GPU."

        tracker_states_local = inference_state["tracker_inference_states"]
        if self.rank == obj_rank:
            self._tracker_remove_object(tracker_states_local, obj_id)

        if is_user_action:
            self.add_action_history(
                inference_state, action_type="remove", obj_ids=[obj_id]
            )

        # update metadata
        tracker_metadata = inference_state["tracker_metadata"]
        _obj_ids = tracker_metadata["obj_ids_per_gpu"][obj_rank]
        tracker_metadata["obj_ids_per_gpu"][obj_rank] = _obj_ids[_obj_ids != obj_id]
        tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
            tracker_metadata["obj_ids_per_gpu"][obj_rank]
        )
        tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
            tracker_metadata["obj_ids_per_gpu"]
        )
        tracker_metadata["obj_id_to_score"].pop(obj_id, None)
        # tracker_metadata["max_obj_id"] # we do not reuse the object id, so we do not update it here

        # Clean up cached frame outputs to remove references to the deleted object
        if "cached_frame_outputs" in inference_state:
            for frame_idx in inference_state["cached_frame_outputs"]:
                frame_cache = inference_state["cached_frame_outputs"][frame_idx]
                if obj_id in frame_cache:
                    del frame_cache[obj_id]

    def _get_gpu_id_by_obj_id(self, inference_state, obj_id):
        """
        Locate GPU ID for a given object.
        """
        obj_ids_per_gpu = inference_state["tracker_metadata"]["obj_ids_per_gpu"]
        for rank, obj_ids in enumerate(obj_ids_per_gpu):
            if obj_id in obj_ids:
                return rank
        return None  # object not found in any GPU

    def _get_tracker_inference_states_by_obj_ids(self, inference_state, obj_ids):
        """
        Get the Tracker inference states that contain the given object ids.
        This is used to run partial Tracker propagation on a single object/bucket.
        Possibly multiple or zero states can be returned.
        """
        states = [
            state
            for state in inference_state["tracker_inference_states"]
            if set(obj_ids) & set(state["obj_ids"])
        ]
        return states

    def _prepare_backbone_feats(self, inference_state, frame_idx, reverse):
        input_batch = inference_state["input_batch"]
        feature_cache = inference_state["feature_cache"]
        num_frames = inference_state["num_frames"]
        geometric_prompt = (
            inference_state["constants"]["empty_geometric_prompt"]
            if inference_state["per_frame_geometric_prompt"][frame_idx] is None
            else inference_state["per_frame_geometric_prompt"][frame_idx]
        )
        _ = self.run_backbone_and_detection(
            frame_idx=frame_idx,
            num_frames=num_frames,
            input_batch=input_batch,
            geometric_prompt=geometric_prompt,
            feature_cache=feature_cache,
            reverse=reverse,
            allow_new_detections=True,
        )

    @torch.inference_mode()
    def add_prompt(
        self,
        inference_state,
        frame_idx,
        text_str=None,
        boxes_xywh=None,
        box_labels=None,
        points=None,
        point_labels=None,
        obj_id=None,
        rel_coordinates=True,
    ):
        if points is not None:
            # Tracker instance prompts
            assert (
                text_str is None and boxes_xywh is None
            ), "When points are provided, text_str and boxes_xywh must be None."
            assert (
                obj_id is not None
            ), "When points are provided, obj_id must be provided."
            return self.add_tracker_new_points(
                inference_state,
                frame_idx,
                obj_id=obj_id,
                points=points,
                labels=point_labels,
                rel_coordinates=rel_coordinates,
                use_prev_mem_frame=self.use_prev_mem_frame,
            )
        else:
            # SAM3 prompts
            return super().add_prompt(
                inference_state,
                frame_idx,
                text_str=text_str,
                boxes_xywh=boxes_xywh,
                box_labels=box_labels,
            )

    @torch.inference_mode()
    def add_tracker_new_points(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points,
        labels,
        rel_coordinates=True,
        use_prev_mem_frame=False,
    ):
        """Add a new point prompt to Tracker. Suppporting instance refinement to existing
        objects by passing existing obj_id or adding a new object by passing a new obj_id.
        use_prev_mem_frame=False to disable cross attention to previous memory frames.
        Every GPU returns the same results, and results should contain all masks including
        these masks not refined or not added by the current user points.
        """
        assert obj_id is not None, "obj_id must be provided to add new points"
        tracker_metadata = inference_state["tracker_metadata"]
        if tracker_metadata == {}:
            # initialize masklet metadata if it's uninitialized (empty dict)
            tracker_metadata.update(self._initialize_metadata())

        obj_rank = self._get_gpu_id_by_obj_id(inference_state, obj_id)

        # prepare feature
        self._prepare_backbone_feats(inference_state, frame_idx, reverse=False)

        object_has_been_refined = self._has_object_been_refined(inference_state, obj_id)
        if (
            obj_rank is not None
            and self.use_stateless_refinement
            and not object_has_been_refined
        ):
            # The first time we start refinement on the object, we remove it.
            logger.debug(
                f"[rank={self.rank}] Removing object {obj_id} before refinement."
            )
            self.remove_object(inference_state, obj_id, is_user_action=False)
            obj_rank = None

        if obj_rank is None:
            # new object, we assign it a GPU and create a new inference state if limit allows
            num_prev_obj = np.sum(tracker_metadata["num_obj_per_gpu"])
            if num_prev_obj >= self.max_num_objects:
                logger.warning(
                    f"add_tracker_new_points: cannot add a new object as we are already tracking {num_prev_obj=} "
                    f"masklets (under {self.max_num_objects=})"
                )
                obj_ids = []
                H_low_res = W_low_res = self.tracker.low_res_mask_size
                H_video_res = inference_state["orig_height"]
                W_video_res = inference_state["orig_width"]
                low_res_masks = torch.zeros(0, 1, H_low_res, W_low_res)
                video_res_masks = torch.zeros(0, 1, H_video_res, W_video_res)
                return frame_idx, obj_ids, low_res_masks, video_res_masks

            new_det_gpu_ids = self._assign_new_det_to_gpus(
                new_det_num=1,
                prev_workload_per_gpu=tracker_metadata["num_obj_per_gpu"],
            )
            obj_rank = new_det_gpu_ids[0]

            # get tracker inference state for the new object
            if self.rank == obj_rank:
                # for batched inference, we create a new inference state
                tracker_state = self._init_new_tracker_state(inference_state)
                inference_state["tracker_inference_states"].append(tracker_state)

            # update metadata
            tracker_metadata["obj_ids_per_gpu"][obj_rank] = np.concatenate(
                [
                    tracker_metadata["obj_ids_per_gpu"][obj_rank],
                    np.array([obj_id], dtype=np.int64),
                ]
            )
            tracker_metadata["num_obj_per_gpu"][obj_rank] = len(
                tracker_metadata["obj_ids_per_gpu"][obj_rank]
            )
            tracker_metadata["obj_ids_all_gpu"] = np.concatenate(
                tracker_metadata["obj_ids_per_gpu"]
            )
            tracker_metadata["max_obj_id"] = max(tracker_metadata["max_obj_id"], obj_id)

            logger.debug(
                f"[rank={self.rank}] Adding new object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "add", frame_idx=frame_idx, obj_ids=[obj_id]
            )
        else:
            # existing object, for refinement
            if self.rank == obj_rank:
                tracker_states = self._get_tracker_inference_states_by_obj_ids(
                    inference_state, [obj_id]
                )
                assert (
                    len(tracker_states) == 1
                ), f"[rank={self.rank}] Multiple Tracker inference states found for the same object id."
                tracker_state = tracker_states[0]

            # log
            logger.debug(
                f"[rank={self.rank}] Refining existing object with id {obj_id} at frame {frame_idx}."
            )
            self.add_action_history(
                inference_state, "refine", frame_idx=frame_idx, obj_ids=[obj_id]
            )

        # assign higher score to added/refined object
        tracker_metadata["obj_id_to_score"][obj_id] = 1.0
        tracker_metadata["obj_id_to_tracker_score_frame_wise"][frame_idx][obj_id] = 1.0

        if self.rank == 0:
            rank0_metadata = tracker_metadata.get("rank0_metadata", {})

            if "removed_obj_ids" in rank0_metadata:
                rank0_metadata["removed_obj_ids"].discard(obj_id)

            if "suppressed_obj_ids" in rank0_metadata:
                for frame_id in rank0_metadata["suppressed_obj_ids"]:
                    rank0_metadata["suppressed_obj_ids"][frame_id].discard(obj_id)

            if "masklet_confirmation" in rank0_metadata:
                obj_ids_all_gpu = tracker_metadata["obj_ids_all_gpu"]
                obj_indices = np.where(obj_ids_all_gpu == obj_id)[0]
                if len(obj_indices) > 0:
                    obj_idx = obj_indices[0]
                    if obj_idx < len(rank0_metadata["masklet_confirmation"]["status"]):
                        rank0_metadata["masklet_confirmation"]["status"][obj_idx] = 1
                        rank0_metadata["masklet_confirmation"]["consecutive_det_num"][
                            obj_idx
                        ] = self.masklet_confirmation_consecutive_det_thresh

        if self.rank == obj_rank:
            frame_idx, obj_ids, low_res_masks, video_res_masks = (
                self.tracker.add_new_points(
                    inference_state=tracker_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=points,
                    labels=labels,
                    clear_old_points=True,
                    rel_coordinates=rel_coordinates,
                    use_prev_mem_frame=use_prev_mem_frame,
                )
            )

            if video_res_masks is not None and len(video_res_masks) > 0:
                video_res_masks = fill_holes_in_mask_scores(
                    video_res_masks,  # shape (N, 1, H_video, W_video)
                    max_area=self.fill_hole_area,
                    fill_holes=True,
                    remove_sprinkles=True,
                )

            # Since the mem encoder has already run for the current input points?
            self.tracker.propagate_in_video_preflight(
                tracker_state, run_mem_encoder=True
            )
            # Clear detector conditioning frames when user clicks are received to allow
            # model updating masks on these frames. It is a noop if user is refining on the
            # detector conditioning frames or adding new objects.
            self.clear_detector_added_cond_frame_in_tracker(
                tracker_state, obj_id, frame_idx
            )

        # fetch results from states and gather across GPUs
        # Use optimized caching approach to avoid reprocessing unmodified objects
        if self.rank == obj_rank and len(obj_ids) > 0:
            new_mask_data = (video_res_masks[obj_ids.index(obj_id)] > 0.0).to(
                torch.bool
            )
        else:
            new_mask_data = None
        # Broadcast the new mask data across all ranks for consistency
        if self.world_size > 1:
            data_list = [new_mask_data.cpu() if new_mask_data is not None else None]
            self.broadcast_python_obj_cpu(data_list, src=obj_rank)
            new_mask_data = data_list[0].to(self.device)

        if self.rank == 0:
            obj_id_to_mask = self._build_tracker_output(
                inference_state,
                frame_idx,
                {obj_id: new_mask_data} if new_mask_data is not None else None,
            )
            # post processing - remove suppressed obj_ids
            obj_id_to_score = tracker_metadata["obj_id_to_score"]
            suppressed_obj_ids = tracker_metadata["rank0_metadata"][
                "suppressed_obj_ids"
            ][frame_idx]
            obj_id_to_tracker_score = tracker_metadata[
                "obj_id_to_tracker_score_frame_wise"
            ][frame_idx]

            out = {
                "obj_id_to_mask": obj_id_to_mask,
                "obj_id_to_score": obj_id_to_score,
                "obj_id_to_tracker_score": obj_id_to_tracker_score,
            }
            self._cache_frame_outputs(
                inference_state,
                frame_idx,
                obj_id_to_mask,
                suppressed_obj_ids=suppressed_obj_ids,
            )
            return frame_idx, self._postprocess_output(
                inference_state, out, suppressed_obj_ids=suppressed_obj_ids
            )
        else:
            return frame_idx, None  # no output on other GPUs

    def _gather_obj_id_to_mask_across_gpus(self, inference_state, obj_id_to_mask_local):
        """Gather obj_id_to_mask from all GPUs. Optionally resize the masks to the video resolution."""
        tracker_metadata = inference_state["tracker_metadata"]

        # concatenate the output masklets from all local inference states
        H_mask = W_mask = self.tracker.low_res_mask_size
        obj_ids_local = tracker_metadata["obj_ids_per_gpu"][self.rank]
        low_res_masks_local = []
        for obj_id in obj_ids_local:
            if obj_id in obj_id_to_mask_local:
                low_res_masks_local.append(obj_id_to_mask_local[obj_id])
            else:
                low_res_masks_local.append(
                    torch.full((H_mask, W_mask), -1024.0, device=self.device)
                )
        if len(low_res_masks_local) > 0:
            low_res_masks_local = torch.stack(low_res_masks_local, dim=0)  # (N, H, W)
            assert low_res_masks_local.shape[1:] == (H_mask, W_mask)
        else:
            low_res_masks_local = torch.zeros(0, H_mask, W_mask, device=self.device)

        # all-gather `low_res_masks_local` into `low_res_masks_global`
        # - low_res_masks_global: Tensor -- (num_global_obj, H_mask, W_mask)
        if self.world_size > 1:
            low_res_masks_local = low_res_masks_local.float().contiguous()
            low_res_masks_peers = [
                low_res_masks_local.new_empty(num_obj, H_mask, W_mask)
                for num_obj in tracker_metadata["num_obj_per_gpu"]
            ]
            dist.all_gather(low_res_masks_peers, low_res_masks_local)
            low_res_masks_global = torch.cat(low_res_masks_peers, dim=0)
        else:
            low_res_masks_global = low_res_masks_local
        return low_res_masks_global

    def _convert_low_res_mask_to_video_res(self, low_res_mask, inference_state):
        """
        Convert a low-res mask to video resolution, matching the format expected by _build_tracker_output.

        Args:
            low_res_mask: Tensor of shape (H_low_res, W_low_res)
            inference_state: Contains video dimensions

        Returns:
            video_res_mask: Tensor of shape (1, H_video, W_video) bool
        """
        if low_res_mask is None:
            return None

        # Convert to 3D for interpolation: (H_low_res, W_low_res) -> (1, H_low_res, W_low_res)
        low_res_mask_3d = low_res_mask.unsqueeze(0).unsqueeze(0)

        # Get video dimensions
        H_video = inference_state["orig_height"]
        W_video = inference_state["orig_width"]

        video_res_mask = F.interpolate(
            low_res_mask_3d.float(),
            size=(H_video, W_video),
            mode="bilinear",
            align_corners=False,
        )  # (1, H_video, W_video)

        # Convert to boolean - already in the right shape!
        return (video_res_mask.squeeze(0) > 0.0).to(torch.bool)

    def clear_detector_added_cond_frame_in_tracker(
        self, tracker_state, obj_id, refined_frame_idx
    ):
        """Clear detector added conditioning frame if it is within a predefined window
        of the refined frame. This allow model to update masks on these frames."""
        obj_idx = self.tracker._obj_id_to_idx(tracker_state, obj_id)

        mask_only_cond_frame_indices = []
        window = self.refinement_detector_cond_frame_removal_window
        for frame_idx in tracker_state["mask_inputs_per_obj"][obj_idx]:
            if frame_idx not in tracker_state["point_inputs_per_obj"][obj_idx]:
                # clear conditioning frames within a window of the refined frame
                if abs(frame_idx - refined_frame_idx) <= window:
                    mask_only_cond_frame_indices.append(frame_idx)

        # clear
        if len(mask_only_cond_frame_indices) > 0:
            for frame_idx in mask_only_cond_frame_indices:
                # obj_ids_on_this_frame is essentially all obj_ids in the state
                # since they are bucket batched
                obj_ids_on_this_frame = tracker_state["obj_id_to_idx"].keys()
                for obj_id2 in obj_ids_on_this_frame:
                    self.tracker.clear_all_points_in_frame(
                        tracker_state, frame_idx, obj_id2, need_output=False
                    )
            logger.debug(
                f"Cleared detector mask only conditioning frames ({mask_only_cond_frame_indices}) in Tracker."
            )
        return


def is_image_type(resource_path: str) -> bool:
    if isinstance(resource_path, list):
        return len(resource_path) == 1
    return resource_path.lower().endswith(tuple(IMAGE_EXTS))
