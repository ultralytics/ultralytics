# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import smart_inference_mode
from collections import OrderedDict
from ..sam.predict import Predictor
from .build import build_sam2


def concat_points(old_point_inputs, new_points, new_labels):
    """Add new points and labels to previous point inputs (add at the end)."""
    if old_point_inputs is None:
        points, labels = new_points, new_labels
    else:
        points = torch.cat([old_point_inputs["point_coords"], new_points], dim=1)
        labels = torch.cat([old_point_inputs["point_labels"], new_labels], dim=1)

    return {"point_coords": points, "point_labels": labels}


class SAM2Predictor(Predictor):
    """
    A predictor class for the Segment Anything Model 2 (SAM2), extending the base Predictor class.

    This class provides an interface for model inference tailored to image segmentation tasks, leveraging SAM2's
    advanced architecture and promptable segmentation capabilities. It facilitates flexible and real-time mask
    generation, working with various types of prompts such as bounding boxes, points, and low-resolution masks.

    Attributes:
        cfg (Dict): Configuration dictionary specifying model and task-related parameters.
        overrides (Dict): Dictionary containing values that override the default configuration.
        _callbacks (Dict): Dictionary of user-defined callback functions to augment behavior.
        args (namespace): Namespace to hold command-line arguments or other operational variables.
        im (torch.Tensor): Preprocessed input image tensor.
        features (torch.Tensor): Extracted image features used for inference.
        prompts (Dict): Collection of various prompt types, such as bounding boxes and points.
        segment_all (bool): Flag to control whether to segment all objects in the image or only specified ones.
        model (torch.nn.Module): The loaded SAM2 model.
        device (torch.device): The device (CPU or GPU) on which the model is loaded.
        _bb_feat_sizes (List[Tuple[int, int]]): List of feature sizes for different backbone levels.

    Methods:
        get_model: Builds and returns the SAM2 model.
        prompt_inference: Performs image segmentation inference based on various prompts.
        set_image: Preprocesses and sets a single image for inference.
        get_im_features: Extracts image features from the SAM2 image encoder.

    Examples:
        >>> predictor = SAM2Predictor(model='sam2_l.pt')
        >>> predictor.set_image('path/to/image.jpg')
        >>> masks, scores = predictor.prompt_inference(im=predictor.im, points=[[500, 375]], labels=[1])
        >>> print(f"Generated {len(masks)} mask(s) with scores: {scores}")
    """

    _bb_feat_sizes = [
        (256, 256),
        (128, 128),
        (64, 64),
    ]

    def get_model(self):
        """Retrieves and initializes the Segment Anything Model (SAM) for image segmentation tasks."""
        return build_sam2(self.args.model)

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

        Args:
            im (torch.Tensor): Preprocessed input image tensor with shape (N, C, H, W).
            bboxes (np.ndarray | List | None): Bounding boxes in XYXY format with shape (N, 4).
            points (np.ndarray | List | None): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List | None): Labels for point prompts with shape (N,). 1 = foreground, 0 = background.
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
            bboxes *= r
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        points = (points, labels) if points is not None else None
        # TODO: Embed prompts
        # if bboxes is not None:
        #     box_coords = bboxes.reshape(-1, 2, 2)
        #     box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=bboxes.device)
        #     box_labels = box_labels.repeat(bboxes.size(0), 1)
        #     # we merge "boxes" and "points" into a single "concat_points" input (where
        #     # boxes are added at the beginning) to sam_prompt_encoder
        #     if concat_points is not None:
        #         concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
        #         concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
        #         concat_points = (concat_coords, concat_labels)
        #     else:
        #         concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=points,
            boxes=bboxes,
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
        Preprocesses and sets a single image for inference.

        This function sets up the model if not already initialized, configures the data source to the specified image,
        and preprocesses the image for feature extraction. Only one image can be set at a time.

        Args:
            image (str | np.ndarray): Image file path as a string, or a numpy array image read by cv2.

        Raises:
            AssertionError: If more than one image is set.

        Examples:
            >>> predictor = SAM2Predictor()
            >>> predictor.set_image("path/to/image.jpg")
            >>> predictor.set_image(np.array([...]))  # Using a numpy array
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
        """Extracts and processes image features using SAM2's image encoder for subsequent segmentation tasks."""
        backbone_out = self.model.forward_image(im)
        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)
        if self.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        feats = [
            feat.permute(1, 2, 0).view(1, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]
        return {"image_embed": feats[-1], "high_res_feats": feats[:-1]}


class SAM2VideoPredictor(SAM2Predictor):
    """The predictor class to handle user interactions with videos and manage inference states."""

    fill_hole_area = 8  # not used

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}

    def get_model(self):
        model = super().get_model()
        model.set_binarize(True)
        return model

    @smart_inference_mode()
    def add_new_points(
        self,
        frame_idx,
        obj_id,
        points,
        labels,
        clear_old_points=True,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(self.inference_state, obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        if not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        points = points.to(self.inference_state["device"])
        labels = labels.to(self.inference_state["device"])

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in self.inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = self.inference_state["frames_already_tracked"][frame_idx]["reverse"]
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
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
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
        obj_ids = self.inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            self.inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            self.inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def init_state(self, offload_video_to_cpu=False, offload_state_to_cpu=False):
        """Initialize a inference state."""
        assert self.dataset is not None
        assert self.dataset.mode == "video"

        inference_state = {}
        inference_state["num_frames"] = self.dataset.frames
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        # inference_state["video_height"] = video_height
        # inference_state["video_width"] = video_width

        inference_state["device"] = torch.device("cuda")
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = torch.device("cuda")

        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        self.inference_state = inference_state

    def get_im_features(self, im):
        """Extracts and processes image features using SAM2's image encoder for subsequent segmentation tasks."""
        backbone_out = self.model.forward_image(im)
        _, vis_feats, vis_pos_embed, feat_sizes = self.model._prepare_backbone_features(backbone_out)
        return vis_feats, vis_pos_embed, feat_sizes

    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index."""
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

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _run_single_frame_inference(
        self,
        inference_state,
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
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(
            inference_state, frame_idx, batch_size
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

        # optionally offload the output to CPU memory to save GPU space
        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # NOTE: Do not support the `fill_holes_in_mask_scores` function since it needs cuda extensions
        # potentially fill holes in the predicted masks
        # if self.fill_hole_area > 0:
        #     pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
        }
        return compact_current_out, pred_masks_gpu

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].clone() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].size(0)
            expanded_maskmem_pos_enc = [x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc
