# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import OrderedDict

import torch

from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.torch_utils import smart_inference_mode

from ..sam.predict import Predictor
from .build import build_sam2


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

    # fill_hole_area = 8  # not used

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.inference_state = {}
        self.non_overlap_masks = True
        self.clear_non_cond_mem_around_input = False
        self.clear_non_cond_mem_for_multi_obj = False
        self.callbacks["on_predict_start"].append(self.init_state)

    def get_model(self):
        model = super().get_model()
        model.set_binarize(True)
        return model

    def inference(self, im, bboxes=None, points=None, labels=None, masks=None):
        """
        Perform image segmentation inference based on the given input cues, using the currently loaded image. This
        method leverages SAM's (Segment Anything Model) architecture consisting of image encoder, prompt encoder, and
        mask decoder for real-time and promptable segmentation tasks.

        Args:
            im (torch.Tensor): The preprocessed input image in tensor format, with shape (N, C, H, W).
            bboxes (np.ndarray | List, optional): Bounding boxes with shape (N, 4), in XYXY format.
            points (np.ndarray | List, optional): Points indicating object locations with shape (N, 2), in pixels.
            labels (np.ndarray | List, optional): Labels for point prompts, shape (N, ). 1 = foreground, 0 = background.
            masks (np.ndarray, optional): Low-resolution masks from previous predictions shape (N,H,W). For SAM H=W=256.
            multimask_output (bool, optional): Flag to return multiple masks. Helpful for ambiguous prompts.

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
        assert bboxes is None and masks is None, "Bounding boxes and masks as prompts has not been supported yet."

        src_shape, dst_shape = self.batch[1][0].shape[:2], im.shape[2:]
        r = min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
        self.inference_state["im"] = im
        if points is not None:
            points = torch.as_tensor(points, dtype=torch.float32, device=self.device)
            points = points[None] if points.ndim == 1 else points
            # Assuming labels are all positive if users don't pass labels.
            if labels is None:
                labels = torch.ones(points.shape[0])
            labels = torch.as_tensor(labels, dtype=torch.int32, device=self.device)
            points *= r
            # (N, 2) --> (1, N, 2), (N, ) --> (1, N)
            points, labels = points[None], labels[None]
        if bboxes is not None:
            bboxes = torch.as_tensor(bboxes, dtype=torch.float32, device=self.device)
            bboxes = bboxes[None] if bboxes.ndim == 1 else bboxes
            bboxes *= r
        if masks is not None:
            masks = torch.as_tensor(masks, dtype=torch.float32, device=self.device).unsqueeze(1)

        frame = self.seen  # NOTE: fix this
        if frame == 0:
            self.add_new_points(obj_id=0, points=points, labels=labels)
            self.inference_state["video_height"] = src_shape[0]
            self.inference_state["video_width"] = src_shape[1]
        self.propagate_in_video_preflight()

        output_dict = self.inference_state["output_dict"]
        consolidated_frame_inds = self.inference_state["consolidated_frame_inds"]
        batch_size = len(self.inference_state["obj_idx_to_id"])
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        if frame in consolidated_frame_inds["cond_frame_outputs"]:
            storage_key = "cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            pred_masks = current_out["pred_masks"]
            if clear_non_cond_mem:
                # clear non-conditioning memory of the surrounding frames
                self._clear_non_cond_mem_around_input(frame)
        elif frame in consolidated_frame_inds["non_cond_frame_outputs"]:
            storage_key = "non_cond_frame_outputs"
            current_out = output_dict[storage_key][frame]
            pred_masks = current_out["pred_masks"]
        else:
            storage_key = "non_cond_frame_outputs"
            current_out, pred_masks = self._run_single_frame_inference(
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
        self.inference_state["frames_already_tracked"][frame] = {"reverse": False}

        return pred_masks.flatten(0, 1), torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device)

    def postprocess(self, preds, img, orig_imgs):
        results = super().postprocess(preds, img, orig_imgs)
        if self.non_overlap_masks:
            for result in results:
                result.masks.data = self.model._apply_non_overlapping_constraints(result.masks.data.unsqueeze(0))[0]
        return results

    @smart_inference_mode()
    def add_new_points(
        self,
        obj_id,
        points,
        labels,
        frame_idx=0,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(obj_id)
        point_inputs_per_frame = self.inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = self.inference_state["mask_inputs_per_obj"][obj_idx]

        point_inputs = {"point_coords": points, "point_labels": labels}

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in self.inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        reverse = False if is_init_cond_frame else self.inference_state["frames_already_tracked"][frame_idx]["reverse"]
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
        consolidated_out = self._consolidate_temp_output_across_obj(
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            # consolidate_at_video_res=True,
            consolidate_at_video_res=False,
        )
        pred_masks = consolidated_out["pred_masks"].flatten(0, 1)
        return pred_masks.flatten(0, 1), torch.ones(1, dtype=pred_masks.dtype, device=pred_masks.device)

    @smart_inference_mode()
    def propagate_in_video_preflight(self):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
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
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outptus
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temprary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(frame_idx, consolidated_out, storage_key)
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
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
        """Initialize a inference state."""
        assert predictor.dataset is not None
        assert predictor.dataset.mode == "video"

        inference_state = {}
        inference_state["num_frames"] = predictor.dataset.frames
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
        predictor.inference_state = inference_state

    def get_im_features(self, im, batch=1):
        """Extracts and processes image features using SAM2's image encoder for subsequent segmentation tasks."""
        backbone_out = self.model.forward_image(im)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos
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
            maskmem_features = maskmem_features.to(dtype=torch.bfloat16, device=self.device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        # NOTE: Do not support the `fill_holes_in_mask_scores` function since it needs cuda extensions
        # potentially fill holes in the predicted masks
        # if self.fill_hole_area > 0:
        #     pred_masks_gpu = fill_holes_in_mask_scores(pred_masks_gpu, self.fill_hole_area)
        pred_masks = pred_masks_gpu.to(self.device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(current_out)
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

    def _get_maskmem_pos_enc(self, current_out):
        """`maskmem_pos_enc` is the same across frames and objects, so we cache it as a constant in the inference
        session to reduce session storage size.
        """
        model_constants = self.inference_state["constants"]
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

    def _consolidate_temp_output_across_obj(
        self,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
    ):
        """Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on a frame into a single output
        for all objects, including 1) fill any missing objects either from `output_dict_per_obj` (if they exist in
        `output_dict_per_obj` for this frame) or leave them as placeholder values (if they don't exist in
        `output_dict_per_obj` for this frame); 2) if specified, rerun memory encoder after apply non-overlapping
        constraints on the object scores.
        """
        batch_size = len(self.inference_state["obj_idx_to_id"])
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = self.inference_state["video_height"]
            consolidated_W = self.inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = self.imgsz[0] // 4
            consolidated_W = self.imgsz[1] // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=-1024.0,
                dtype=torch.float32,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(batch_size, self.model.hidden_dim),
                fill_value=-1024.0,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = self.inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.
            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(frame_idx)
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr
                continue
            # Add the temporary object output mask to consolidated output mask
            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]
            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = torch.nn.functional.interpolate(
                    obj_mask,
                    size=consolidated_pred_masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            high_res_masks = torch.nn.functional.interpolate(
                consolidated_out["pred_masks"].to(self.device, non_blocking=True),
                size=self.imgsz,
                mode="bilinear",
                align_corners=False,
            )
            if self.model.non_overlap_masks_for_mem_enc:
                high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                batch_size=batch_size,
                high_res_masks=high_res_masks,
                is_mask_from_pts=True,  # these frames are what the user interacted with
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def _get_empty_mask_ptr(self, frame_idx):
        """Get a dummy object pointer based on an empty mask on the current frame."""
        # A dummy (empty) mask with a single object
        batch_size = 1
        mask_inputs = torch.zeros(
            (batch_size, 1, *self.imgsz),
            dtype=torch.float32,
            device=self.device,
        )

        # Retrieve correct image features
        current_vision_feats, current_vision_pos_embeds, feat_sizes = self.get_im_features(frame_idx, batch_size)

        # Feed the empty mask and image feature above to get a dummy object pointer
        current_out = self.model.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=True,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={},
            num_frames=self.inference_state["num_frames"],
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        return current_out["obj_ptr"]

    def _run_memory_encoder(self, batch_size, high_res_masks, is_mask_from_pts):
        """
        Run the memory encoder on `high_res_masks`.

        This is usually after applying non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        current_vision_feats, _, feat_sizes = self.get_im_features(self.inference_state["im"], batch_size)
        maskmem_features, maskmem_pos_enc = self.model._encode_new_memory(
            current_vision_feats=current_vision_feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            is_mask_from_pts=is_mask_from_pts,
        )

        # optionally offload the output to CPU memory to save GPU space
        maskmem_features = maskmem_features.to(dtype=torch.bfloat16, device=self.device, non_blocking=True)
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc({"maskmem_pos_enc": maskmem_pos_enc})
        return maskmem_features, maskmem_pos_enc

    def _add_output_per_object(self, frame_idx, current_out, storage_key):
        """
        Split a multi-object output into per-object output slices and add them into `output_dict_per_obj`.

        The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        assert maskmem_features is None or isinstance(maskmem_features, torch.Tensor)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = self.inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
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
        """
        Remove the non-conditioning memory around the input frame. When users provide correction clicks, the surrounding
        frames' non-conditioning memories can still contain outdated object appearance information and could confuse the
        model.

        This method clears those non-conditioning memories surrounding the interacted frame to avoid giving the model
        both old and new information about the object.
        """
        r = self.model.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.model.num_maskmem
        frame_idx_end = frame_idx + r * self.model.num_maskmem
        output_dict = self.inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in self.inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)
