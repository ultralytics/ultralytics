# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license


from typing import Dict, List, Tuple, Union, Optional, Any
import cv2 
from datetime import datetime
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ultralytics.data.augment import LetterBox
from ultralytics.engine.predictor import BasePredictor
from ultralytics.engine.results import Results
from ultralytics.utils import DEFAULT_CFG, ops
from ultralytics.utils.torch_utils import select_device, smart_inference_mode

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


from .predict import SAM2Predictor


class ImageState(object):

    """
    ImageState is a class that encapsulates the state of an image during the inference process, which includes the image data, its dimensions, and various features extracted from it.
    It is designed to handle the image preprocessing, feature extraction, and storage of various outputs related to the image, such as backbone features, positional encodings, and mask features.

    It also manages the prompts (points, boxes, and masks) that are used for segmentation tasks, allowing for the initialization and updating of these prompts as needed.
    Attributes:
        image_data (torch.Tensor): The preprocessed image tensor ready for model input.
        img_w (int): The width of the original image.
        img_h (int): The height of the original image.
        img_name (str): The name of the image, generated based on the current date and time if not provided.
        device (torch.device): The device on which the image data is stored (e.g., CPU or GPU).
        backbone_fpn (dict): A dictionary containing backbone features.
        vision_pos_enc (list): A list of positional encodings for the visual features.
        expanded_image (torch.Tensor): The expanded image tensor for processing multiple objects.
        expanded_backbone_out (torch.Tensor): The expanded backbone output tensor for processing multiple objects.
        high_res_features (torch.Tensor): High-resolution features extracted from the image.
        vision_feats (torch.Tensor): The visual features extracted from the image.
        vision_pos_embeds (torch.Tensor): The positional embeddings for the visual features.
        feat_sizes (list): A list containing the sizes of the extracted features.
        pix_feat_with_mem (torch.Tensor): Pixel features with memory for segmentation tasks.
        maskmem_features (torch.Tensor): Features related to mask memory for segmentation tasks.
        point_inputs (dict): A dictionary mapping object indices to their point inputs.
        mask_inputs (dict): A dictionary mapping object indices to their mask inputs.
        current_out (dict): A dictionary mapping object indices to their current outputs.
        prev_out (dict): A dictionary mapping object indices to their previous outputs.
    Methods:
        init_consolidated_out: Initializes the consolidated output for the image state, preparing it for segmentation tasks.
        perpare_data: Prepares the image data by resizing and normalizing it, returning the processed tensor and its dimensions.
        set_prompt: Sets the prompt data (points, box, mask) for the image state.
        _prepare_backbone_features: Prepares and flattens visual features for processing by the model.
        get_im_features: Extracts and processes image features using the model's image encoder for 

    Example:
        imgState = ImageState(image=img, img_name=img_name,
                              image_size=self.image_size,
                              device=self.device,
                              max_obj_num=self._max_obj_num
                              )

    """

    def __init__(
        self, 
        image: Union[torch.Tensor, np.ndarray, Image.Image], 
        image_size: int, 
        device: torch.device,
        max_obj_num: int, 
        img_name: Optional[str] = None
    ) -> None:
        """
        Initializes the ImageState with the provided image, frame index, image size, device, and maximum number of objects.
        Args:
            image (torch.Tensor | np.ndarray | PIL.Image): The input image to be processed.
            image_size (int): The target size to which the image will be resized.
            device (torch.device): The device on which the image data will be stored (e.g., CPU or GPU).
            max_obj_num (int): The maximum number of objects that can be tracked in the image.
            img_name (str, optional): The name of the image. If not provided, a unique name will be generated based on the current date and time.

        """
        self.device = device
        self.image_data, self.img_w, self.img_h = self.perpare_data(image, image_size=image_size)
        if img_name is not None:
            self.img_name = img_name
        else:
            # use date and timestamp (ms) to generate a unique image name
            date_time= datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            self.img_name=f"image_{date_time}"



        # self.cached_features=None
        self.backbone_fpn = None  # 
        self.vision_pos_enc = None

        self.expanded_image = None  # tensor shape {num_object,c,h,w}  # lyus todel

        self.expanded_backbone_out = None  # the input of _prepare_backbone_features   # lyus todel

        self.high_res_features = None
        self.vision_feats = None
        self.vision_pos_embeds = None
        self.feat_sizes = None

        self.pix_feat_with_mem = None
        self.maskmem_features = None
        self._max_obj_num = max_obj_num

        self.point_inputs = {i: None for i in range(self._max_obj_num)}
        self.mask_inputs = {i: None for i in range(self._max_obj_num)}
        self.current_out = {i: None for i in range(self._max_obj_num)}
        self.prev_out = {i: None for i in range(self._max_obj_num)}
        self.no_obj_score=-1024.0  #


    def init_consolidated_out(self, hidden_dim: int) -> Dict[str, torch.Tensor]:
        """
        Initialize the consolidated output for the image state.
        This method prepares a dictionary that will hold the consolidated outputs for the image state, including mask features, positional encodings, predicted masks, object pointers, and object score logits.
        
        Args:
            hidden_dim (int): The hidden dimension size for object pointers.        
        Returns:
            Dict[str, torch.Tensor]: A dictionary containing initialized consolidated outputs with keys:
                - maskmem_features: None initially
                - maskmem_pos_enc: None initially  
                - pred_masks: Tensor of shape (max_obj_num, 1, img_h//4, img_w//4)
                - obj_ptr: Tensor of shape (max_obj_num, hidden_dim)
                - object_score_logits: Tensor of shape (max_obj_num, 1)
        """
        consolidated_mask_key = "pred_masks"
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: torch.full(
                size=(self._max_obj_num, 1, self.img_h // 4, self.img_w // 4),  # self.img_w,self.img_h
                fill_value=self.no_obj_score,
                dtype=torch.float32,
                device=self.device,
            ),
            "obj_ptr": torch.full(
                size=(self._max_obj_num, hidden_dim),
                fill_value=self.no_obj_score,
                dtype=torch.float32,
                device=self.device,
            ),
            "object_score_logits": torch.full(
                size=(self._max_obj_num, 1),
                # default to 10.0 for object_score_logits, i.e. assuming the object is
                # present as sigmoid(10)=1, same as in `predict_masks` of `MaskDecoder`
                fill_value= -32 ,#10.0,
                dtype=torch.float32,
                device=self.device,
            ),
        }
        return consolidated_out

    def perpare_data(
        self,
        img: Union[torch.Tensor, np.ndarray, Image.Image],
        image_size: int = 1024,
        img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> Tuple[torch.Tensor, int, int]:
        """
        Prepare the image data for processing by resizing and normalizing it. if the input is a tensor, it will be returned directly with its width and height. If the input is a numpy array or PIL image, it will be resized to the specified image size, normalized using the provided mean and standard deviation, and converted to a torch tensor.
        Args:
            img (torch.Tensor | np.ndarray | PIL.Image): The input image to be processed.
            image_size (int): The target size to which the image will be resized.
            img_mean (tuple): The mean values for normalization.
            img_std (tuple): The standard deviation values for normalization.       
        Returns:
            Tuple[torch.Tensor, int, int]: A tuple containing:
                - img (torch.Tensor): The processed image tensor ready for model input.
                - width (int): The width of the original image.
                - height (int): The height of the original image.
        """

        if isinstance(img, torch.Tensor):
            height, width = img.shape[-2:]
            img=img.view(-1, height,width)
            return img, width, height
        elif isinstance(img, np.ndarray):
            img_np = img
            img_np = cv2.resize(img_np, (image_size, image_size)) / 255.0
            height, width = img.shape[:2]
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()
            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
            img -= img_mean.to(self.device)
            img /= img_std.to(self.device)
            return img, width, height
        elif isinstance(img, Image.Image):
            img_np = (
                    np.array(img.convert("RGB").resize((image_size, image_size))) / 255.0
            )
            width, height = img.size
            img = torch.from_numpy(img_np).permute(2, 0, 1).float()

            img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
            img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
            img -= img_mean.to(self.device)
            img /= img_std.to(self.device)
            return img, width, height

    def set_prompt(
        self, 
        points: Optional[torch.Tensor], 
        box: Optional[torch.Tensor], 
        mask: Optional[torch.Tensor]
    ) -> None:
        """
        Set the prompt data for the image state.
        
        Args:
            points (torch.Tensor | None): Point coordinates for prompting.
            box (torch.Tensor | None): Bounding box coordinates for prompting.
            mask (torch.Tensor | None): Mask data for prompting.
        """
        self.points = points
        self.box = box
        self.mask = mask


    def _prepare_backbone_features(
        self, backbone_out: Dict[str, List[torch.Tensor]],
        num_feature_levels: int):
        """Prepare and flatten visual features. 
        Resulting features will be stored in the image state, including:
            - backbone_fpn: list of backbone features from the image encoder.
            - vision_pos_enc: list of positional encodings from the image encoder.
            - high_res_features: list of high-resolution features for skip connections in the mask decoder.
            - vision_feats: list of flattened visual features for transformer input.
            - vision_pos_embeds: list of flattened positional embeddings for transformer input.
            - feat_sizes: list of tuples representing the sizes of the features.
        
        Args: 
            backbone_out (dict): A dictionary containing backbone features with keys "backbone_fpn" and "vision_pos_enc" that output from the image encoder.
            num_feature_levels (int): Number of feature levels to be used from the backbone.
            

        """
        self.backbone_fpn = backbone_out["backbone_fpn"]
        self.vision_pos_enc = backbone_out["vision_pos_enc"]


        expanded_backbone_out = {
            "backbone_fpn": self.backbone_fpn.copy(),
            "vision_pos_enc": self.vision_pos_enc.copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(
                self._max_obj_num, -1, -1, -1
            )
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(self._max_obj_num, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        assert len(expanded_backbone_out["backbone_fpn"]) == len(expanded_backbone_out["vision_pos_enc"])
        assert len(expanded_backbone_out["backbone_fpn"]) >= num_feature_levels

        feature_maps = expanded_backbone_out["backbone_fpn"][-num_feature_levels:]
        vision_pos_embeds = expanded_backbone_out["vision_pos_enc"][-num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        def get_high_res_features(current_vision_feats, current_feat_sizes):
            if len(current_vision_feats) > 1:
                high_res_features = [
                    x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                    for x, s in zip(current_vision_feats[:-1], current_feat_sizes[:-1])
                ]
            else:
                high_res_features = None
            return high_res_features

        self.high_res_features = get_high_res_features(vision_feats, feat_sizes)
        self.vision_feats = vision_feats
        self.vision_pos_embeds = vision_pos_embeds
        self.feat_sizes = feat_sizes




class SAM2DynamicInteractivePredictor(SAM2Predictor):

    """
    SAM2DynamicInteractivePredictor extends SAM2Predictor to support dynamic interactions with video frames or a sequence of images. 
    Attributes:
        memory_bank: OrderedDict: Stores the states of each image with prompts.
        obj_idx_set (set): A set to keep track of the object indices that have been added.
        obj_id_to_idx (OrderedDict): Maps object IDs to their corresponding indices.
        obj_idx_to_id (OrderedDict): Maps object indices to their corresponding IDs.
    Methods:  
        __init__(cfg, overrides=None, max_obj_num=9, _callbacks=None): Initializes the predictor with the given configuration and overrides.
        get_model(): Retrieves and configures the model with binarization enabled.
        inference(img, image_name=None, bboxes=None, obj_ids=None, update_memory=False, *args, **kwargs): Performs inference on a single image with optional bounding boxes and object IDs.
        postprocess(preds, img, orig_imgs): Post-processes the predictions to apply non-overlapping constraints if required.  d

    Examples:
            >>> predictor = SAM2DynamicInteractivePredictor(cfg=DEFAULT_CFG)
            >>> predictor(source=support_img1, bboxes=bboxes1,obj_ids=labels1,update_memory=True)
            >>> results1= predictor(source=query_img1)
            >>> predictor(source=support_img2, bboxes=bboxes2,obj_ids=labels2,update_memory=True)
            >>> results2= predictor(source=query_img2)

    """



    
    def __init__(
        self, 
        cfg: Dict[str, Any] = DEFAULT_CFG, 
        overrides: Optional[Dict[str, Any]] = None,
        max_obj_num: int = 3, 
        _callbacks: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Initialize the predictor with configuration and optional overrides.

        This constructor initializes the SAM2DynamicInteractivePredictor with a given configuration, applies any
        specified overrides

        Args:
            cfg (Dict[str, Any]): Configuration dictionary containing default settings.
            overrides (Dict[str, Any] | None): Dictionary of values to override default configuration.
            _callbacks (Dict[str, Any] | None): Dictionary of callback functions to customize behavior.
            max_obj_num (int): Maximum number of objects to track. Default is 3. this is set to keep fix feature size for the model.

        Examples:
            >>> predictor = SAM2DynamicInteractivePredictor(cfg=DEFAULT_CFG)
            >>> predictor_example_with_imgsz = SAM2DynamicInteractivePredictor(overrides={"imgsz": 640})
            >>> predictor_example_with_callback = SAM2DynamicInteractivePredictor(_callbacks={"on_predict_start": custom_callback})
        """
        super().__init__(cfg, overrides, _callbacks)



        self.done_warmup = True
        self.device= torch.device("cuda" if torch.cuda.is_available() else "cpu")




        self.use_high_res_features_in_sam = True
        self.num_feature_levels = 3

        self.use_mask_input_as_output_without_sam = False
        self.no_obj_score = -1024.0

        self.non_overlap_masks = True
        self.memory_bank = OrderedDict()
        self.obj_idx_set=set()
        if not hasattr(self, "obj_id_to_idx"):
            self.obj_id_to_idx = OrderedDict()
        if not hasattr(self, "obj_idx_to_id"):
            self.obj_idx_to_id = OrderedDict()
        self._max_obj_num = max_obj_num  # Maximum number of objects to track, can be adjusted as needed
        for i in range(self._max_obj_num):
            self.obj_id_to_idx[i + 1] = i
            self.obj_idx_to_id[i] = i + 1


    def get_model(self) -> torch.nn.Module:
        """
        Retrieve and configure the model with binarization enabled.

        Returns:
            torch.nn.Module: The configured SAM2 model with binarization enabled.
            
        Note:
            This method overrides the base class implementation to set the binarize flag to True.
        """
        model = super().get_model()
        model.set_binarize(True)

        self.no_obj_embed_spatial = None
        #if no_obj_embed_spatial:
        # self.mem_dim = self.hidden_dim
        self.no_obj_embed_spatial = torch.nn.Parameter(torch.zeros(1, 64))
        from torch.nn.init import trunc_normal_
        trunc_normal_(self.no_obj_embed_spatial, std=0.02)
        self.no_obj_embed_spatial=self.no_obj_embed_spatial.to(self.device, non_blocking=True)

        return model
        
    @property
    def image_size(self) -> int:
        """
        Returns the image size of the model.
        
        Returns:
            int: The image size of the model input size
        """
        return self.model.image_size

    @smart_inference_mode()
    def inference(
        self,
        img: Union[torch.Tensor, np.ndarray], 
        image_name: Optional[str] = None, 
        bboxes: Optional[List[List[float]]] = None,
        obj_ids: Optional[List[int]] = None,
        update_memory: bool = False, 
        *args: Any, 
        **kwargs: Any
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform inference on a single image with optional bounding boxes and object IDs.
        
        It has two modes: one is to run inference on a single image without updating the memory, 
        and the other is to update the memory with the provided bounding boxes and object IDs.
        
        When update_memory is True, it will update the memory with the provided bboxes and obj_ids.
        When update_memory is False, it will only run inference on the provided image without updating the memory.

        Args:
            img (torch.Tensor | np.ndarray): The input image tensor or numpy array.
            image_name (str | None): Optional name for the image, used for identification.
            bboxes (List[List[float]] | None): Optional list of bounding boxes to update the memory.
            obj_ids (List[int] | None): Optional list of object IDs corresponding to the bounding boxes.
            update_memory (bool): Flag to indicate whether to update the memory with new objects.
            *args (Any): Additional arguments for the inference process.
            **kwargs (Any): Additional keyword arguments for the inference process.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - video_res_masks (torch.Tensor): The output masks in shape (C, H, W)
                - object_score_logits (torch.Tensor): Quality scores for each mask
        """
        if self.model is None:
            self.setup_model(model=None)


        imgState = self.createState(img, image_name)


        src_shape = self.batch[1][0].shape[:2]
        dst_shape = img.shape[2:]
        # apply letterbox resize to the bboxes
        if bboxes is not None:
            bboxes = torch.tensor(bboxes, dtype=torch.float32)  # Ensure float32 dtype
            r= min(dst_shape[0] / src_shape[0], dst_shape[1] / src_shape[1])
            bboxes *=r


        if update_memory:
            assert bboxes is not None, "bboxes must be provided when update_memory is True"
            assert obj_ids is not None, "obj_ids must be provided when update_memory is True"
            assert len(bboxes) == len(obj_ids), "bboxes and obj_ids must have the same length" 
            for box ,obj_id in zip(bboxes,obj_ids):
                self.add_new_prompt(imgState,bbox= box, obj_id=int(obj_id))
            self.update_memory(imgState)

 
        current_out = self.track_step(
            imgState=imgState,
            obj_idx=None)
        pred_masks_gpu = current_out["pred_masks"]

        _, video_res_masks = self._get_orig_video_res_output(pred_masks_gpu)

        # filter the masks and logits based on the object indices
        obj_idx_set= self.obj_idx_set
        if len(obj_idx_set) == 0:
            raise RuntimeError("No objects have been added to the state. Please add objects before inference.")
        
        video_res_masks = video_res_masks.to(self.device, non_blocking=True)
        video_res_masks = video_res_masks[torch.tensor(list(obj_idx_set), device=self.device)]
        object_score_logits= current_out["object_score_logits"].to(self.device, non_blocking=True)

        print(object_score_logits)

        object_score_logits = object_score_logits[torch.tensor(list(obj_idx_set), device=self.device)]
         # the orginal score are in [-32,32], and a object score larger than 0 means the object is present, we map it to [-1,1] range
        object_score_logits=object_score_logits/32


        #  we use a activate function to make sure the object score logits are non-negative, so that we can use it as a mask
        object_score_logits =torch.relu(object_score_logits)
        video_res_masks= video_res_masks.squeeze()
        if len(video_res_masks.shape)==2: 
            video_res_masks=video_res_masks.unsqueeze(0)
        if len(object_score_logits.shape)>1: 
            object_score_logits=object_score_logits.squeeze(1)
        return  video_res_masks, object_score_logits
        
    def postprocess(
        self, 
        preds: Tuple[torch.Tensor, ...], 
        img: torch.Tensor, 
        orig_imgs: List[np.ndarray]
    ) -> List[Results]:
        """
        Post-process the predictions to apply non-overlapping constraints if required.

        This method extends the post-processing functionality by applying non-overlapping constraints
        to the predicted masks if the `non_overlap_masks` flag is set to True. This ensures that
        the masks do not overlap, which can be useful for certain applications.

        Args:
            preds (tuple): The predictions from the model.
            img (torch.Tensor): The processed image tensor.
            orig_imgs (List[np.ndarray]): The original images before processing.

        Returns:
            (list): The post-processed predictions.

        Note:
            If `non_overlap_masks` is True, the method applies constraints to ensure non-overlapping masks.
        """
        # (N, 1, H, W), (N, 1)
        pred_masks, pred_scores = preds[:2]
        pred_bboxes = preds[2] if self.segment_all else None
        names = dict(enumerate(str(i) for i in range(len(pred_masks))))

        if not isinstance(orig_imgs, list):  # input images are a torch.Tensor, not a list
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)

        results = []
        for masks, orig_img, img_path in zip([pred_masks], orig_imgs, self.batch[0]):
            if len(masks) == 0:
                masks, pred_bboxes = None, torch.zeros((0, 6), device=pred_masks.device)
            else:
                masks = ops.scale_masks(masks[None].float(), orig_img.shape[:2], padding=False)[0]
                masks = masks > self.model.mask_threshold  # to bool
       
                masks[pred_scores <=self.args.conf]=False

                if pred_bboxes is not None:
                    pred_bboxes = ops.scale_boxes(img.shape[2:], pred_bboxes.float(), orig_img.shape, padding=False)
                else:
                    pred_bboxes = batched_mask_to_box(masks)
                # NOTE: SAM models do not return cls info. This `cls` here is just a placeholder for consistency.
                cls = torch.arange(len(pred_masks), dtype=torch.int32, device=pred_masks.device)
                pred_bboxes = torch.cat([pred_bboxes, pred_scores[:, None], cls[:, None]], dim=-1)
                filtered_index= (pred_scores>self.args.conf).cpu().numpy().tolist()
                result=Results(orig_img, path=img_path, 
                               names=names,
                                 masks=masks[filtered_index],
                                   boxes=pred_bboxes[filtered_index])

            results.append(result)
        # Reset segment-all mode.
        self.segment_all = False

        if self.non_overlap_masks:
            for result in results:
                if result.masks is None or len(result.masks) == 0:
                    continue
                result.masks.data = self.model._apply_non_overlapping_constraints(result.masks.data.unsqueeze(0))[0]
        return results
    





    def forward_image(self, imgState: ImageState):
        """
        Forward the image through the model to extract features and cache them in the ImageState.
        
        Args:
        imgState (ImageState): The ImageState object containing the image data and other attributes.
            
        
        """
        img_batch = imgState.image_data.cuda().float().unsqueeze(0)


        backbone_out = self.model.image_encoder(img_batch)

        if self.use_high_res_features_in_sam:  # ?
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.model.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.model.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        imgState._prepare_backbone_features(backbone_out,self.num_feature_levels)




    @smart_inference_mode()
    def createState(self, img, img_name=None):
        """
        Create a new ImageState object for the given image.

        Args:
            img (torch.Tensor | np.ndarray): The input image tensor or numpy array.
            img_name (str | None): Optional name for the image, used for identification.
        """
        imgState = ImageState(image=img, img_name=img_name,
                              image_size=self.image_size,
                              device=self.device,
                              max_obj_num=self._max_obj_num
                              )
        self.forward_image(imgState)

        return imgState

    @smart_inference_mode()
    def add_new_prompt(
            self,
            imgState,
            obj_id,
            points=None,
            labels=None,
            bbox=None,
            normalize_coords=True,

    ):
        """
        Add new bboxes to a specific frame for a given object ID.

        This method updates the imgState with new prompts (points or masks) for a specified
        object and updates the internal state accordingly. /
        Args:
            imgState (ImageState): the imgState to be updated. 
            obj_id (int): The ID of the object to which the prompts are associated.
            points (torch.Tensor | None): The coordinates of the points of interest.
            labels (torch.Tensor | None): The labels corresponding to the points.
            bbox (torch.Tensor | list | None): The bounding box coordinates for the object.    
            normalize_coords (bool): Whether to normalize the coordinates of the points based on the image size.
        
        Returns:
            pred_masks (torch.Tensor): The flattened predicted masks.
            pred_scores (torch.Tensor): A tensor of ones indicating the number of objects.

        Raises:
            AssertionError: If bbox is not provided. 

        """
        obj_idx = self._obj_id_to_idx(obj_id)
        self.obj_idx_set.add(obj_idx)

        assert (
                bbox is not None 
        ), "only bbox prompt is supported for now, please provide bbox"



        # assert (
        #         bbox is not None or points is not None
        # ), "Either bbox or points is required"

        if points is None:
            points = torch.zeros(0, 2, dtype=torch.float32)
        elif not isinstance(points, torch.Tensor):
            points = torch.tensor(points, dtype=torch.float32)
        if labels is None:
            labels = torch.zeros(0, dtype=torch.int32)
        elif not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels, dtype=torch.int32)
        if points.dim() == 2:
            points = points.unsqueeze(0)  # add batch dimension
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)  # add batch dimension
        if bbox is not None:
            bbox = torch.tensor(bbox, dtype=torch.float32, device=points.device)
            box_coords = bbox.reshape(1, 2, 2)
            box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
            box_labels = box_labels.reshape(1, 2)
            points = torch.cat([box_coords, points], dim=1)
            labels = torch.cat([box_labels, labels], dim=1)
        if normalize_coords:
            video_H = self.image_size
            video_W = self.image_size
            points = points / torch.tensor([video_W, video_H]).to(points.device)
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size
        points = points.to(self.device)
        labels = labels.to(self.device)
        clear_old_points=False
        if clear_old_points and obj_idx in imgState.point_inputs.keys():
            imgState.point_inputs[obj_idx] = None
        from sam2.utils.misc import concat_points

        imgState.point_inputs[obj_idx] = concat_points(imgState.point_inputs[obj_idx], points, labels)

    @smart_inference_mode()
    def update_memory(self, imgState: ImageState):
        """
        append the imgState to the memory_bank and update the memory for the model.

        Args:
            imgState (ImageState): The ImageState object containing the image data and prompts.
        """
        consolidated_out = imgState.init_consolidated_out(self.model.hidden_dim)
        for obj_idx in range(imgState._max_obj_num):

            if obj_idx not in  self.obj_idx_set:
                continue
            prev_out = imgState.prev_out.get(obj_idx)
            prev_sam_mask_logits = None
            if prev_out is not None and prev_out["pred_masks"] is not None:
                prev_sam_mask_logits = prev_out["pred_masks"].cuda(non_blocking=True)
                # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
                prev_sam_mask_logits = torch.clamp(prev_sam_mask_logits, -32.0, 32.0)

            current_out = self.track_step(
                imgState=imgState,
                obj_idx=obj_idx,
  
            )

            imgState.current_out[obj_idx] = current_out


            out = imgState.current_out[obj_idx]
            if out is None:
                pass
            else:
                obj_mask = out["pred_masks"]
                assert obj_mask.shape[-2:] == consolidated_out["pred_masks"].shape[-2:], f"Expected mask shape {consolidated_out['pred_masks'].shape[-2:]} but got {obj_mask.shape[-2:]} for object {obj_idx}."
                consolidated_out["pred_masks"][obj_idx: obj_idx + 1] = obj_mask
                consolidated_out["obj_ptr"][obj_idx: obj_idx + 1] = out["obj_ptr"]

                if "object_score_logits" in out.keys():
                    consolidated_out["object_score_logits"][obj_idx: obj_idx + 1] = out[
                        "object_score_logits"
                    ]
                else:
                    print("warining, KeyError: 'object_score_logits'  ")


      


        device = self.device
        high_res_masks = torch.nn.functional.interpolate(
            consolidated_out["pred_masks"].to(device, non_blocking=True),
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        if self.model.non_overlap_masks_for_mem_enc:
            high_res_masks = self.model._apply_non_overlapping_constraints(high_res_masks)
        maskmem_features, maskmem_pos_enc = self._encode_new_memory(
            current_vision_feats=imgState.vision_feats,
            feat_sizes=imgState.feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=consolidated_out["object_score_logits"],
            is_mask_from_pts=True,
        )
        consolidated_out["maskmem_features"] = maskmem_features
        consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc
        imgState.consolidated_out = consolidated_out
        self.memory_bank[imgState.img_name] = imgState


    def _get_maskmem_pos_enc(self, current_out):
        """
        Get the mask memory positional encoding from the current output.
        Args:
            current_out (dict): The current output dictionary containing the mask memory positional encoding.
        Returns:
            expanded_maskmem_pos_enc (list | None): The expanded mask memory positional encoding, or None if not present.

        """
        model_constants = self.model.condition_state["constants"]
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
            expanded_maskmem_pos_enc = [
                x.expand(batch_size, -1, -1, -1) for x in maskmem_pos_enc
            ]
        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc


    def _prepare_memory_conditioned_features(self, imgState, obj_idx):
        """
        Prepare the memory-conditioned features for the current image state.

        Args:
            imgState (ImageState): The current image state containing vision features and positional encodings.
            obj_idx (int | None): The index of the object for which to prepare the features.
        """
        current_vision_feats = imgState.vision_feats
        feat_sizes = imgState.feat_sizes
        if len(self.memory_bank) == 0 or isinstance(obj_idx, int):
            # # for initial conditioning frames with, encode them without using any previous memory
            directly_add_no_mem_embed=True
            if directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.model.no_mem_embed
                C =self.model.memory_attention.d_model
                B = imgState._max_obj_num
                # B=1
                feat_sizes = imgState.feat_sizes
                H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size

                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem
        else:
            # for inference frames, use the memory features from previous frames
            valued_memory_bank = self.selected_memory_bank()
            memory, memory_pos_embed = self.get_maskmem_enc(valued_memory_bank)

            pix_feat_with_mem = self.model.memory_attention(
                curr=imgState.vision_feats[-1:],
                curr_pos=imgState.vision_pos_embeds[-1:],
                memory=memory,
                memory_pos=memory_pos_embed,
                num_obj_ptr_tokens=0,  # num_obj_ptr_tokens
            )
            # reshape the output (HW)BC => BCHW
            C =self.model.memory_attention.d_model
            B = imgState._max_obj_num
            feat_sizes = imgState.feat_sizes
            H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
            pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
            return pix_feat_with_mem
    def selected_memory_bank(self):
        """
        Get the list of states that have valid memory features, based on the assumption that all states in `self.memory_bank` are valid.
        """
        return self.memory_bank  # keep all states 

    def get_maskmem_enc(self, valued_memory_bank):
        """

        Get the memory and positional encoding from the memory, which is used to condition the current image features.
        Args:
            valued_memory_bank (OrderedDict): A dictionary containing the states of each image with valid memory features.
        """
        if len(valued_memory_bank) == 0:
            return None, None
        to_cat_memory, to_cat_memory_pos_embed = [], []
        t_pos = 0
        for img_name, state in valued_memory_bank.items():
            feats = state.consolidated_out["maskmem_features"].cuda(non_blocking=True)
            to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
            maskmem_enc = state.consolidated_out["maskmem_pos_enc"][-1].cuda()
            maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
            # Temporal positional encoding
            maskmem_enc = (
                    maskmem_enc + self.model.maskmem_tpos_enc[self.model.num_maskmem - t_pos - 1]
            )
            to_cat_memory_pos_embed.append(maskmem_enc)

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)
        return memory, memory_pos_embed

    def _obj_id_to_idx(self, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = self.obj_id_to_idx.get(obj_id, None)
        return obj_idx

    def track_step(
            self,
            imgState,
            obj_idx,
    ):
        """
        Trakking step for the current image state, which involves processing the image features and running the SAM heads to predict masks.

        Args:
            imgState (ImageState): The current image state containing the image data and prompts.
            obj_idx (int | None): The index of the object for which to predict masks. If None, it processes all objects.
            run_mem_encoder (bool): Flag to indicate whether to run the memory encoder on the predicted masks.

        """
        if obj_idx is not None:
            point_inputs = imgState.point_inputs[obj_idx]
            mask_inputs = imgState.mask_inputs[obj_idx]
        else:
            point_inputs = None
            mask_inputs = None

        current_vision_feats = imgState.vision_feats
        high_res_features = imgState.high_res_features
        feat_sizes = imgState.feat_sizes


        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}

        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:

            # When use_mask_input_as_output_without_sam=True, we directly output the mask input
            # (see it as a GT mask) without using a SAM prompt encoder + mask decoder.
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1,self.model.memory_attention.d_model, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
                )
        else:
            # fused the visual feature with previous memory features in the memory bank
            imgState.pix_feat_with_mem = self._prepare_memory_conditioned_features(imgState, obj_idx)
            sam_outputs = self._forward_sam_heads(
                backbone_features=imgState.pix_feat_with_mem,
                obj_idx=obj_idx,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=False,
                keep_sparse_dense_embeddings=False,
            )

        (
            _,
            _,
            _,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["object_score_logits"] = object_score_logits


        return current_out
    


    def _forward_sam_heads(
            self,
            backbone_features,
            obj_idx,
            point_inputs=None,
            mask_inputs=None,
            high_res_features=None,
            multimask_output=False,
            **kwargs
    ):
        """
        Forward SAM prompt encoders and mask heads.

        Inputs:
        - backbone_features: image features of [B, C, H, W] shape
        - point_inputs: a dictionary with "point_coords" and "point_labels", where
          1) "point_coords" has [B, P, 2] shape and float32 dtype and contains the
             absolute pixel-unit coordinate in (x, y) format of the P input points
          2) "point_labels" has shape [B, P] and int32 dtype, where 1 means
             positive clicks, 0 means negative clicks, and -1 means padding
        - mask_inputs: a mask of [B, 1, H*16, W*16] shape, float or bool, with the
          same spatial size as the image.
        - high_res_features: either 1) None or 2) or a list of length 2 containing
          two feature maps of [B, C, 4*H, 4*W] and [B, C, 2*H, 2*W] shapes respectively,
          which will be used as high-resolution feature maps for SAM decoder.
        - multimask_output: if it's True, we output 3 candidate masks and their 3
          corresponding IoU estimates, and if it's False, we output only 1 mask and
          its corresponding IoU estimate.

        Outputs:
        - low_res_multimasks: [B, M, H*4, W*4] shape (where M = 3 if
          `multimask_output=True` and M = 1 if `multimask_output=False`), the SAM
          output mask logits (before sigmoid) for the low-resolution masks, with 4x
          the resolution (1/4 stride) of the input backbone_features.
        - high_res_multimasks: [B, M, H*16, W*16] shape (where M = 3
          if `multimask_output=True` and M = 1 if `multimask_output=False`),
          upsampled from the low-resolution masks, with shape size as the image
          (stride is 1 pixel).
        - ious, [B, M] shape, where (where M = 3 if `multimask_output=True` and M = 1
          if `multimask_output=False`), the estimated IoU of each output mask.
        - low_res_masks: [B, 1, H*4, W*4] shape, the best mask in `low_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `low_res_multimasks`.
        - high_res_masks: [B, 1, H*16, W*16] shape, the best mask in `high_res_multimasks`.
          If `multimask_output=True`, it's the mask with the highest IoU estimate.
          If `multimask_output=False`, it's the same as `high_res_multimasks`.
        - obj_ptr: [B, C] shape, the object pointer vector for the output mask, extracted
          based on the output token from the SAM mask decoder.
        """
        B = backbone_features.size(0)
        if isinstance(obj_idx, int):
            B = 1
        device = backbone_features.device
        assert backbone_features.size(1) == self.model.sam_prompt_embed_dim
        assert backbone_features.size(2) == self.model.sam_image_embedding_size
        assert backbone_features.size(3) == self.model.sam_image_embedding_size

        if not kwargs.get("keep_sparse_dense_embeddings", False) or not hasattr(self, "_sparse_dense_embeddings"):
            # a) Handle point prompts
            if point_inputs is not None:
                sam_point_coords = point_inputs["point_coords"]
                sam_point_labels = point_inputs["point_labels"]
                assert sam_point_coords.size(0) == B and sam_point_labels.size(0) == B
            else:
                # If no points are provide, pad with an empty point (with label -1)
                sam_point_coords = torch.zeros(B, 1, 2, device=device)
                sam_point_labels = -torch.ones(B, 1, dtype=torch.int32, device=device)

            # b) Handle mask prompts
            if mask_inputs is not None:
                # If mask_inputs is provided, downsize it into low-res mask input if needed
                # and feed it as a dense mask prompt into the SAM mask encoder
                assert len(mask_inputs.shape) == 4 and mask_inputs.shape[:2] == (B, 1)
                if mask_inputs.shape[-2:] != self.model.sam_prompt_encoder.mask_input_size:
                    sam_mask_prompt = F.interpolate(
                        mask_inputs.float(),
                        size=self.model.sam_prompt_encoder.mask_input_size,
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                else:
                    sam_mask_prompt = mask_inputs
            else:
                # Otherwise, simply feed None (and SAM's prompt encoder will add
                # a learned `no_mask_embed` to indicate no mask input in this case).
                sam_mask_prompt = None

            sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                points=(sam_point_coords, sam_point_labels),
                boxes=None,
                masks=sam_mask_prompt,
            )
            # print(torch.unique(sparse_embeddings))
            # print(torch.unique(dense_embeddings))
            # if kwargs.get("keep_sparse_dense_embeddings"):
            #     self._sparse_dense_embeddings = (sparse_embeddings, dense_embeddings)
        else:
                pass
                # sparse_embeddings = torch.stack([self._sparse_dense_embeddings[0][0]] * B)
                # dense_embeddings = torch.stack([self._sparse_dense_embeddings[1][0]] * B)

        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.model.sam_mask_decoder(
            image_embeddings=backbone_features[:B],
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=False,  # the image is already batched
            high_res_features=[feat[:B] for feat in high_res_features],
        )


        if self.model.pred_obj_scores:
            is_obj_appearing = object_score_logits > 0

            # Mask used for spatial memories is always a *hard* choice between obj and no obj,
            # consistent with the actual mask prediction
            low_res_multimasks = torch.where(
                is_obj_appearing[:, None, None],
                low_res_multimasks,
                self.no_obj_score,
            )

        # convert masks from possibly bfloat16 (or float16) to float32
        # (older PyTorch versions before 2.1 don't support `interpolate` on bf16)
        low_res_multimasks = low_res_multimasks.float()
        high_res_multimasks = F.interpolate(
            low_res_multimasks,
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )

        sam_output_token = sam_output_tokens[:, 0]
        if multimask_output:
            # take the best mask prediction (with the highest IoU estimation)
            best_iou_inds = torch.argmax(ious, dim=-1)
            batch_inds = torch.arange(B, device=device)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds].unsqueeze(1)
            if sam_output_tokens.size(1) > 1:
                sam_output_token = sam_output_tokens[batch_inds, best_iou_inds]
        else:
            low_res_masks, high_res_masks = low_res_multimasks, high_res_multimasks

        # Extract object pointer from the SAM output token (with occlusion handling)
        obj_ptr = self.model.obj_ptr_proj(sam_output_token)
        if self.model.pred_obj_scores:
            # Allow *soft* no obj ptr, unlike for masks
            if self.model.soft_no_obj_ptr:
                # Only hard possible with gt
                assert not self.teacher_force_obj_scores_for_mem
                lambda_is_obj_appearing = object_score_logits.sigmoid()
            else:
                lambda_is_obj_appearing = is_obj_appearing.float()

            if self.model.fixed_no_obj_ptr:
                obj_ptr = lambda_is_obj_appearing * obj_ptr
            obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.model.no_obj_ptr

        return (
            low_res_multimasks,
            high_res_multimasks,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        )


    def _get_orig_video_res_output(self, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        Args:
            any_res_masks (torch.Tensor): The masks to be resized, shape [B, C, H, W], where C is the number of masks,
                                          H and W are the height and width of the masks.
        Returns:
            any_res_masks (torch.Tensor): The original resolution masks, shape [B, C, H, W].
            video_res_masks (torch.Tensor): The resized masks to the video resolution, shape [B, C, video_H, video_W].
        """
        device = self.device
        video_H = self.image_size
        video_W = self.image_size
        any_res_masks = any_res_masks.to(device, non_blocking=True)
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = torch.nn.functional.interpolate(
                any_res_masks,
                size=(video_H, video_W),
                mode="bilinear",
                align_corners=False,
            )
        if self.non_overlap_masks:
            video_res_masks = self.model._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

   

    def _encode_new_memory(
            self,
            current_vision_feats,
            feat_sizes,
            pred_masks_high_res,
            object_score_logits,
            is_mask_from_pts,
    ):
        """Encode the current image and its prediction into a memory feature.
        Args:
            current_vision_feats (List[torch.Tensor]): List of vision features from the image encoder.
            feat_sizes (List[Tuple[int, int]]): List of feature sizes corresponding to the vision features.
            pred_masks_high_res (torch.Tensor): The predicted masks at high resolution, shape [B, 1, H, W].
            object_score_logits (torch.Tensor): The object score logits for the current frame.
            is_mask_from_pts (bool): Whether the mask is derived from points.
        Returns:
            maskmem_features (torch.Tensor): The encoded memory features, shape [B, C, H, W].
            maskmem_pos_enc (List[torch.Tensor]): The positional encodings for the memory features.
        """
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C =self.model.memory_attention.d_model
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
        if self.model.non_overlap_masks_for_mem_enc and not self.model.training:
            # optionally, apply non-overlapping constraints to the masks (it's applied
            # in the batch dimension and should only be used during eval, where all
            # the objects come from the same video under batch size 1).
            pred_masks_high_res = self._apply_non_overlapping_constraints(
                pred_masks_high_res
            )
        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.model.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts
        if binarize and not self.model.training:
            mask_for_mem = (pred_masks_high_res > 0).float()
        else:
            # apply sigmoid on the raw mask logits to turn them into range (0, 1)
            mask_for_mem = torch.sigmoid(pred_masks_high_res)
        # apply scale and bias terms to the sigmoid probabilities
        if self.model.sigmoid_scale_for_mem_enc != 1.0:
            mask_for_mem = mask_for_mem * self.model.sigmoid_scale_for_mem_enc
        if self.model.sigmoid_bias_for_mem_enc != 0.0:
            mask_for_mem = mask_for_mem + self.model.sigmoid_bias_for_mem_enc
        maskmem_out = self.model.memory_encoder(
            pix_feat, mask_for_mem, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                                        1 - is_obj_appearing[..., None, None]
                                ) * self.no_obj_embed_spatial[..., None, None].expand(
                *maskmem_features.shape
            )

        return maskmem_features, maskmem_pos_enc

