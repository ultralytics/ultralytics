import torch
from .sam3_video_model import SAM3VideoSemanticPredictor, MaskletConfirmationStatus
from ultralytics.utils.ops import xyxy2ltwh
from .sam3.data_misc import Datapoint, FindStage
from .sam3.geometry_encoders import Prompt
from ultralytics.utils import LOGGER, ops
from ultralytics.engine.results import Results
from torchvision.ops import masks_to_boxes


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
        inference_state["text_prompt"] = None
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
        find_text_batch = ["<text placeholder>", "visual"]
        stages = FindStage(
            img_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=self.device, dtype=torch.long),
            input_boxes=[torch.zeros(258, device=self.device)],
            input_boxes_mask=[torch.empty(0, dtype=torch.bool, device=self.device)],
            input_boxes_label=[torch.empty(0, dtype=torch.long, device=self.device)],
            input_points=[torch.empty(0, 257, device=self.device)],
            input_points_mask=[torch.empty(0, device=self.device)],
            object_ids=[],
        )

        # construct the final `BatchedDatapoint` and cast to GPU
        input_batch = Datapoint(img_batch=im, find_text_batch=find_text_batch, find_inputs=stages)
        self.inference_state["input_batch"] = input_batch
        frame = frame - 1
        if frame == 0:  # TODO: more stable check
            self.add_prompt(frame_idx=frame, text_str=text, bboxes=bboxes, labels=labels)
        return self._run_single_frame_inference(frame, reverse=False)

    def postprocess(self, preds, img, orig_imgs):
        """Post-process the predictions to apply non-overlapping constraints if required."""
        obj_id_to_mask = preds["obj_id_to_mask"]  # low res masks
        curr_obj_ids = sorted(obj_id_to_mask.keys())
        if len(curr_obj_ids) == 0:
            pred_masks, pred_boxes = None, torch.zeros((0, 6), device=pred_masks.device)
        else:
            pred_masks = torch.cat([obj_id_to_mask[obj_id] for obj_id in curr_obj_ids], dim=0)
            pred_ids = torch.tensor(curr_obj_ids, dtype=torch.int32, device=pred_masks.device)
            pred_scores = torch.tensor(
                [preds["obj_id_to_score"][obj_id] for obj_id in curr_obj_ids], device=pred_masks.device
            )
            pred_cls = torch.zeros(pred_scores.shape[0], dtype=pred_scores.dtype, device=pred_scores.device)
            keep = (pred_scores > self.args.conf) & pred_masks.any(dim=(1, 2))
            pred_masks = pred_masks[keep]
            pred_boxes = masks_to_boxes(pred_masks)
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

        # names = getattr(self.model, "names", [str(i) for i in range(pred_scores.shape[0])])
        names = dict(enumerate(str(i) for i in range(pred_masks.shape[0])))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)
        results = []
        for masks, boxes, orig_img, img_path in zip([pred_masks], [pred_boxes], orig_imgs, self.batch[0]):
            results.append(Results(orig_img, path=img_path, names=names, masks=masks, boxes=boxes))
        return results

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
            allow_new_detections=has_text_prompt or has_geometric_prompt,
        )
        # update inference state
        inference_state["tracker_inference_states"] = tracker_states_local_new
        inference_state["tracker_metadata"] = tracker_metadata_new

        out = {
            "obj_id_to_mask": obj_id_to_mask,
            "obj_id_to_score": obj_id_to_score,  # first frame detection score
            "obj_id_to_tracker_score": tracker_metadata_new["obj_id_to_tracker_score_frame_wise"][frame_idx],
        }
        # removed_obj_ids is only needed on rank 0 to handle hotstart delay buffer
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

    @torch.inference_mode()
    def add_prompt(
        self,
        frame_idx,
        text_str=None,
        bboxes=None,
        labels=None,
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
        assert text_str is not None or bboxes is not None, "at least one type of prompt (text, boxes) must be provided"
        assert 0 <= frame_idx < num_frames, f"{frame_idx=} is out of range for a total of {num_frames} frames"

        # 1) add text prompt
        if text_str is not None and text_str != "visual":
            inference_state["text_prompt"] = text_str
            inference_state["input_batch"].find_text_batch[0] = text_str
            text_id = self.TEXT_ID_FOR_TEXT
        else:
            inference_state["text_prompt"] = None
            inference_state["input_batch"].find_text_batch[0] = "<text placeholder>"
            text_id = self.TEXT_ID_FOR_VISUAL
        inference_state["input_batch"].find_inputs.text_ids[...] = text_id

        # 2) handle box prompt
        bboxes, labels = self._prepare_geometric_prompts(self.batch[1][0].shape[:2], bboxes, labels)
        assert (bboxes is not None) == (labels is not None)
        if bboxes is not None:
            geometric_prompt = Prompt(
                box_embeddings=bboxes,  # (seq, bs, 4)
                box_mask=None,
                box_labels=labels,  # (seq, bs)
                point_embeddings=None,
                point_mask=None,
                point_labels=None,
            )

            inference_state["per_frame_geometric_prompt"][frame_idx] = geometric_prompt
        out = self._run_single_frame_inference(frame_idx, reverse=False, inference_state=inference_state)
        return frame_idx, out

    def _apply_object_wise_non_overlapping_constraints(self, pred_masks, obj_scores, background_value=-10.0):
        """
        Applies non-overlapping constraints object wise (i.e. only one object can claim the overlapping region)
        """
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
