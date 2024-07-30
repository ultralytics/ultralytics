# Ultralytics YOLO 🚀, AGPL-3.0 license

import torch

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
