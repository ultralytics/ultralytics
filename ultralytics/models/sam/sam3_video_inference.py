import torch
from .sam3_video_model import SAM3VideoSemanticPredictor, MaskletConfirmationStatus
from ultralytics.utils.ops import xyxy2xywhn, xyxy2ltwh
from .sam3.data_misc import BatchedDatapoint, convert_my_tensors, FindStage
from .sam3.geometry_encoders import Prompt
from ultralytics.utils import LOGGER, ops
from ultralytics.engine.results import Results
from torchvision.ops import masks_to_boxes
from tqdm.auto import tqdm


class Sam3VideoInference(SAM3VideoSemanticPredictor):
    TEXT_ID_FOR_TEXT = 0
    TEXT_ID_FOR_VISUAL = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_state = {}
        self.callbacks["on_predict_start"].append(self.init_state)

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
        num_frames = predictor.dataset.frames
        inference_state = {
            "num_frames": num_frames,
            "constants": {},  # values that don't change across frames (so we only need to hold one copy of them)
            "tracker_inference_states": [],
            "tracker_metadata": {},
            "feature_cache": {},
        }
        inference_state["previous_stages_out"] = [None] * num_frames
        inference_state["text_prompt"] = None
        inference_state["per_frame_visual_prompt"] = [None] * num_frames
        inference_state["per_frame_geometric_prompt"] = [None] * num_frames

        inference_state["constants"]["empty_geometric_prompt"] = Prompt(
            box_embeddings=torch.zeros(0, 1, 4, device=predictor.device, dtype=predictor.torch_dtype),
            box_mask=torch.zeros(1, 0, device=predictor.device, dtype=torch.bool),
            box_labels=torch.zeros(0, 1, device=predictor.device, dtype=torch.long),
            point_embeddings=torch.zeros(0, 1, 2, device=predictor.device, dtype=predictor.torch_dtype),
            point_mask=torch.zeros(1, 0, device=predictor.device, dtype=torch.bool),
            point_labels=torch.zeros(0, 1, device=predictor.device, dtype=torch.long),
        )
        predictor.inference_state = inference_state

    def inference(self, im, bboxes=None, labels=None, text: list[str] = None, *args, **kwargs):
        frame = self.dataset.frame
        height, width = self.batch[1][0].shape[:2]
        self.inference_state["orig_height"] = height
        self.inference_state["orig_width"] = width

        find_text_batch = ["<text placeholder>", "visual"]
        stages = [
            FindStage(
                img_ids=[stage_id],
                text_ids=[0],
                input_boxes=[torch.zeros(258, device=self.device)],
                input_boxes_mask=[torch.empty(0, dtype=torch.bool, device=self.device)],
                input_boxes_label=[torch.empty(0, dtype=torch.long, device=self.device)],
                input_points=[torch.empty(0, 257, device=self.device)],
                input_points_mask=[torch.empty(0, device=self.device)],
                object_ids=[],
            )
            for stage_id in range(1)
        ]
        for i in range(len(stages)):
            stages[i] = convert_my_tensors(stages[i])  # TODO

        # construct the final `BatchedDatapoint` and cast to GPU
        input_batch = BatchedDatapoint(
            img_batch=im,
            find_text_batch=find_text_batch,
            find_inputs=stages,
            find_targets=[None] * 1,
            find_metadatas=[None] * 1,
        )
        self.inference_state["input_batch"] = input_batch
        frame = frame - 1
        if frame == 0:  # TODO: more stable check
            self.add_prompt(frame_idx=frame, text_str=text, boxes_xywh=bboxes, box_labels=labels)
        out = self._run_single_frame_inference(frame, reverse=False)
        # TODO: add hotstart_delay
        unconfirmed_obj_ids_per_frame = {}  # frame_idx -> hidden_obj_ids
        hotstart_removed_obj_ids = set()

        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1
        suppressed_obj_ids = out["suppressed_obj_ids"]
        unconfirmed_status_frame_idx = frame + unconfirmed_status_delay

        # Clamp the frame index to stay within video bounds
        num_frames = self.inference_state["num_frames"]
        unconfirmed_status_frame_idx = max(0, min(unconfirmed_status_frame_idx, num_frames - 1))

        unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(unconfirmed_status_frame_idx, None)
        return self._postprocess_output(
            self.inference_state,
            out,
            hotstart_removed_obj_ids,
            suppressed_obj_ids,
            unconfirmed_obj_ids,
        )

    def postprocess(self, preds, img, orig_imgs):
        """Post-process the predictions to apply non-overlapping constraints if required."""
        pred_boxes = preds["out_boxes_xywh"]  # (nc, num_query, 4)
        pred_masks = preds["out_binary_masks"]
        pred_scores = preds["out_probs"]
        # pred_cls = preds["out_obj_ids"]
        pred_cls = torch.tensor(list(range(pred_scores.shape[0])), dtype=pred_scores.dtype, device=pred_scores.device)
        pred_boxes = torch.cat([pred_boxes, pred_scores[..., None], pred_cls[..., None]], dim=-1)

        keep = pred_scores > self.args.conf
        pred_masks = pred_masks[keep]
        pred_boxes = pred_boxes[keep]
        pred_boxes[:, :4] = ops.ltwh2xyxy(pred_boxes[:, :4])

        # names = getattr(self.model, "names", [str(i) for i in range(pred_scores.shape[0])])
        names = dict(enumerate(str(i) for i in range(pred_masks.shape[0])))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            if masks.shape[0] == 0:
                masks, boxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                boxes[..., 0] *= orig_img.shape[1]
                boxes[..., 1] *= orig_img.shape[0]
                boxes[..., 2] *= orig_img.shape[1]
                boxes[..., 3] *= orig_img.shape[0]
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

    @torch.inference_mode()
    def reset_state(self, inference_state):
        """Revert `inference_state` to what it was right after initialization."""
        inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
        inference_state["text_prompt"] = None
        for t in range(0):  # TODO
            inference_state["input_batch"].find_inputs[t].text_ids[...] = 0
            # constructing an output list in inference state (we start with an empty list)
            inference_state["previous_stages_out"][t] = None
            inference_state["per_frame_visual_prompt"][t] = None
            inference_state["per_frame_geometric_prompt"][t] = None

        inference_state["tracker_inference_states"].clear()
        inference_state["tracker_metadata"].clear()
        inference_state["feature_cache"].clear()

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
                    f"visual prompts (box as an initial prompt) should only have one box, but got {boxes_cxcywh.shape=}"
                )
            if not box_labels.item():
                LOGGER.warning("A negative box is added as a visual prompt.")
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

    def _get_processing_order(self, inference_state, start_frame_idx, max_frame_num_to_track, reverse):
        num_frames = inference_state["num_frames"]
        previous_stages_out = inference_state["previous_stages_out"]
        if all(out is None for out in previous_stages_out) and start_frame_idx is None:
            raise RuntimeError(
                "No prompts are received on any frames. Please add prompt on at least one frame before propagation."
            )
        # set start index, end index, and processing order
        if start_frame_idx is None:
            # default: start from the earliest frame with input points
            start_frame_idx = min(t for t, out in enumerate(previous_stages_out) if out is not None)
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

        processing_order, end_frame_idx = self._get_processing_order(
            inference_state,
            start_frame_idx,
            max_frame_num_to_track,
            reverse=reverse,
        )

        hotstart_buffer = []
        hotstart_removed_obj_ids = set()
        # when deciding whether to output a masklet on `yield_frame_idx`, we check whether the object is confirmed
        # in a future frame (`unconfirmed_frame_delay` frames after the current frame). For example, if we require
        # an object to be detected in 3 consecutive frames to be confirmed, then we look 2 frames in the future --
        # e.g., we output an object on frame 4 only if it becomes confirmed on frame 6.
        unconfirmed_status_delay = self.masklet_confirmation_consecutive_det_thresh - 1
        unconfirmed_obj_ids_per_frame = {}  # frame_idx -> hidden_obj_ids
        for frame_idx in tqdm(processing_order, desc="propagate_in_video", disable=self.rank > 0):
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
                    unconfirmed_status_frame_idx = max(0, min(unconfirmed_status_frame_idx, num_frames - 1))

                    unconfirmed_obj_ids = unconfirmed_obj_ids_per_frame.get(unconfirmed_status_frame_idx, None)
                    postprocessed_out = self._postprocess_output(
                        inference_state,
                        yield_out,
                        hotstart_removed_obj_ids,
                        suppressed_obj_ids,
                        unconfirmed_obj_ids,
                    )

                else:
                    postprocessed_out = None  # no output on other GPUs
                yield yield_frame_idx, postprocessed_out

    def _run_single_frame_inference(self, frame_idx, reverse, inference_state=None):
        """
        Perform inference on a single frame and get its inference results. This would
        also update `inference_state`.
        """
        inference_state = inference_state or self.inference_state
        # prepare inputs
        input_batch = inference_state["input_batch"]
        tracker_states_local = inference_state["tracker_inference_states"]
        has_text_prompt = inference_state["text_prompt"] is not None
        has_geometric_prompt = inference_state["per_frame_geometric_prompt"][frame_idx] is not None
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
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        # update inference state
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new
        # use a dummy string in "previous_stages_out" to indicate this frame has outputs
        inference_state["previous_stages_out"][frame_idx] = "_THIS_FRAME_HAS_OUTPUTS_"

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # first frame detection score
            "obj_id_to_tracker_score": tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
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
                out["unconfirmed_obj_ids"] = tracker_metadata_new["obj_ids_all_gpu"][is_unconfirmed].tolist()
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
            out_probs = torch.tensor([out["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids])
            out_tracker_probs = torch.tensor(
                [
                    (out["obj_id_to_tracker_score"][obj_id] if obj_id in out["obj_id_to_tracker_score"] else 0.0)
                    for obj_id in curr_obj_ids
                ]
            )
            out_binary_masks = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)

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
            keep_idx_gpu = keep_idx.pin_memory().to(device=out_binary_masks.device, non_blocking=True)

            out_obj_ids = torch.index_select(out_obj_ids, 0, keep_idx)
            out_probs = torch.index_select(out_probs, 0, keep_idx)
            out_tracker_probs = torch.index_select(out_tracker_probs, 0, keep_idx)
            out_binary_masks = torch.index_select(out_binary_masks, 0, keep_idx_gpu)

            out_boxes_xyxy = masks_to_boxes(out_binary_masks)
            out_boxes_xywh = xyxy2ltwh(out_boxes_xyxy)  # convert to xywh format
            # normalize boxes
            out_boxes_xywh[..., 0] /= W_video
            out_boxes_xywh[..., 1] /= H_video
            out_boxes_xywh[..., 2] /= W_video
            out_boxes_xywh[..., 3] /= H_video

        # apply non-overlapping constraints on the existing masklets
        if out_binary_masks.shape[0] > 1:
            assert len(out_binary_masks) == len(out_tracker_probs)
            out_binary_masks = (
                self._apply_object_wise_non_overlapping_constraints(
                    out_binary_masks.unsqueeze(1),
                    out_tracker_probs.unsqueeze(1).to(out_binary_masks.device),
                    background_value=0,
                ).squeeze(1)
            ) > 0

        outputs = {
            "out_obj_ids": out_obj_ids.to(out_boxes_xywh.device),
            "out_probs": out_probs.to(out_boxes_xywh.device),
            "out_boxes_xywh": out_boxes_xywh,
            "out_binary_masks": out_binary_masks,
            "frame_stats": out.get("frame_stats", None),
        }
        # for k, v in outputs.items():
        #     if isinstance(v, torch.Tensor):
        #         print(k, v.shape, v.device)
        return outputs

    @torch.inference_mode()
    def add_prompt(
        self,
        frame_idx,
        text_str=None,
        boxes_xywh=None,
        box_labels=None,
        inference_state=None,
    ):
        """
        Add text, point or box prompts on a single frame. This method returns the inference
        outputs only on the prompted frame.

        Note that text prompts are NOT associated with a particular frame (i.e. they apply
        to all frames). However, we only run inference on the frame specified in `frame_idx`.
        """
        LOGGER.debug("Running add_prompt on frame %d", frame_idx)

        inference_state = inference_state or self.inference_state
        num_frames = inference_state["num_frames"]
        assert text_str is not None or boxes_xywh is not None, (
            "at least one type of prompt (text, boxes) must be provided"
        )
        assert 0 <= frame_idx < num_frames, f"{frame_idx=} is out of range for a total of {num_frames} frames"

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
        for t in range(1):  # TODO
            inference_state["input_batch"].find_inputs[t].text_ids[...] = text_id

        # 2) handle box prompt
        assert (boxes_xywh is not None) == (box_labels is not None)
        if boxes_xywh is not None:
            boxes_xywh = torch.as_tensor(boxes_xywh, dtype=self.torch_dtype)
            box_labels = torch.as_tensor(box_labels, dtype=torch.long)
            # input boxes are expected to be [xmin, ymin, width, height] format
            # in normalized coordinates of range 0~1, similar to FA
            assert boxes_xywh.dim() == 2
            assert boxes_xywh.size(0) > 0 and boxes_xywh.size(-1) == 4
            assert box_labels.dim() == 1 and box_labels.size(0) == boxes_xywh.size(0)
            boxes_cxcywh = xyxy2xywhn(boxes_xywh, h=inference_state["orig_height"], w=inference_state["orig_width"])
            # assert (boxes_xywh >= 0).all().item() and (boxes_xywh <= 1).all().item()
            assert (boxes_cxcywh >= 0).all().item() and (boxes_cxcywh <= 1).all().item()

            new_box_input = boxes_cxcywh, box_labels

            # handle the case of visual prompt (also added as an input box from the UI)
            boxes_cxcywh, box_labels, geometric_prompt = self._get_visual_prompt(
                inference_state, frame_idx, boxes_cxcywh, box_labels
            )

            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt

        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, self._postprocess_output(inference_state, out)

    # TODO: handle this
    def _apply_object_wise_non_overlapping_constraints(self, pred_masks, obj_scores, background_value=-10.0):
        """
        Applies non-overlapping constraints object wise (i.e. only one object can claim the overlapping region)
        """
        # Replace pixel scores with object scores
        pred_masks_single_score = torch.where(pred_masks > 0, obj_scores[..., None, None], background_value)
        # Apply pixel-wise non-overlapping constraint based on mask scores
        pixel_level_non_overlapping_masks = self._apply_non_overlapping_constraints(pred_masks_single_score)
        # Replace object scores with pixel scores. Note, that now only one object can claim the overlapping region
        pred_masks = torch.where(
            pixel_level_non_overlapping_masks > 0,
            pred_masks,
            torch.clamp(pred_masks, max=background_value),
        )
        return pred_masks

    # TODO: remove it once this class inherits from SAM2Predictor
    @staticmethod
    def _apply_non_overlapping_constraints(pred_masks):
        """Apply non-overlapping constraints to masks, keeping the highest scoring object per location."""
        batch_size = pred_masks.shape[0]
        if batch_size == 1:
            return pred_masks

        device = pred_masks.device
        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = torch.argmax(pred_masks, dim=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = torch.arange(batch_size, device=device)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = torch.where(keep, pred_masks, torch.clamp(pred_masks, max=-10.0))
        return pred_masks
