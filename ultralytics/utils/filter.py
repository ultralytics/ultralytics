import torch

from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors


class FilterFuncBase:

    def __init__(self):
        pass

    def __call__(self, dbox, embed, anchor, stride):

        raise NotImplementedError("FilterFuncBase is an abstract class and cannot be called directly.")


class IOUFilter(FilterFuncBase):

    def __init__(self,bbox_prompt):
        super().__init__()
        self.bbox_prompt=bbox_prompt
        self.top_k = 1



    def __call__(self, dbox_xywh, embed, anchors, strides):
        # dbox_xywh shape: [B, 4, N] -> transpose to [B, N, 4] for coordinate conversion
        dbox_xywh = dbox_xywh.transpose(1, 2)  # [B, N, 4]

        # xywh -> xyxy
        dbox = torch.zeros_like(dbox_xywh)
        dbox[..., 0:2] = dbox_xywh[..., 0:2] - dbox_xywh[..., 2:4] / 2  # x1, y1 = cx - w/2, cy - h/2
        dbox[..., 2:4] = dbox_xywh[..., 0:2] + dbox_xywh[..., 2:4] / 2  # x2, y2 = cx + w/2, cy + h/2



        # Now dbox is [B, N, 4], can be used directly for IoU calculation
        iou_score = self.iou(dbox, self.bbox_prompt)  # (N, M) where N=num_anchors, M=num_prompts
        iou_score0=iou_score[:,0]
        print(f"iou_score0 shape: {iou_score0.shape}, max: {iou_score0.max():.6f}, mean: {iou_score0.mean():.6f}")

        # Use top-k selection for each prompt
        k = min(self.top_k, iou_score.shape[0])  # Number of top matches to select per prompt, limited by available anchors
        topk_values, topk_indices = torch.topk(iou_score, k=k, dim=0)  # (k, M)


        # Gather embeddings for top-k anchors for each prompt
        B, embed_dim, N = embed.shape
        M = self.bbox_prompt.shape[1]  # Number of prompts
        assert B==1

        

        mask = torch.zeros(B, M, N, dtype=torch.bool, device=embed.device)
        for m in range(M):
            mask[:, m, topk_indices[:, m]] = True
        # # Optional: output average score of top-k

        # Create prompt_embed tensor to store results
        prompt_embed = torch.zeros(B, embed_dim, M, device=embed.device)
        # prompt_score = topk_values.mean(dim=0)
        # print(f"prompt_score: {prompt_score}")
        return mask  # (N, M) bool
    

    def iou(self,box, bbox_prompt):
        """
        Compute the IoU (Intersection over Union) between two sets of bounding boxes (xyxy format).
        Args:
            box: A tensor of shape (1, N, 4) representing the first set of bounding boxes.
            bbox_prompt: A tensor of shape (1, M, 4) representing the second set of bounding boxes.

        Returns:
            A tensor of shape (N, M) representing the IoU between each pair of boxes and bbox_prompts.
        """
        box = box.squeeze(0)  # (N, 4) - xyxy format
        bbox_prompt = bbox_prompt.squeeze(0)  # (M, 4) - xyxy format
        
        # Extract coordinates
        # box: (N, 4) -> x1, y1, x2, y2
        # bbox_prompt: (M, 4) -> x1, y1, x2, y2
        
        # Expand dimensions for broadcasting
        # box: (N, 1, 4), bbox_prompt: (1, M, 4)
        box_expanded = box.unsqueeze(1)  # (N, 1, 4)
        prompt_expanded = bbox_prompt.unsqueeze(0)  # (1, M, 4)
        
        # Calculate intersection coordinates
        # Get the coordinates of intersection rectangle
        x1_inter = torch.max(box_expanded[:, :, 0], prompt_expanded[:, :, 0])  # (N, M)
        y1_inter = torch.max(box_expanded[:, :, 1], prompt_expanded[:, :, 1])  # (N, M)
        x2_inter = torch.min(box_expanded[:, :, 2], prompt_expanded[:, :, 2])  # (N, M)
        y2_inter = torch.min(box_expanded[:, :, 3], prompt_expanded[:, :, 3])  # (N, M)
        
        # Calculate intersection area
        # If there's no intersection, width or height will be <= 0
        inter_width = torch.clamp(x2_inter - x1_inter, min=0)  # (N, M)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)  # (N, M)
        inter_area = inter_width * inter_height  # (N, M)
        
        # Calculate areas of both sets of boxes
        box_area = (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1])  # (N,)
        prompt_area = (bbox_prompt[:, 2] - bbox_prompt[:, 0]) * (bbox_prompt[:, 3] - bbox_prompt[:, 1])  # (M,)
        
        # Expand areas for broadcasting
        box_area_expanded = box_area.unsqueeze(1)  # (N, 1)
        prompt_area_expanded = prompt_area.unsqueeze(0)  # (1, M)
        
        # Calculate union area
        union_area = box_area_expanded + prompt_area_expanded - inter_area  # (N, M)
        
        # Calculate IoU
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        iou_scores = inter_area / (union_area + epsilon)  # (N, M)
        
        return iou_scores


from ultralytics.utils.tal import TORCH_1_10, dist2bbox, dist2rbox, make_anchors
from ultralytics.nn.modules import DFL
import torch.nn as nn

class FilterActuator:
    """
    Visual Prompt Embedding BBox Matcher.
    Used to filter anchors based on filter function and aggregate features as prompt embedding.
    """

    def __init__(self, filter_fn:FilterFuncBase, dfl=None ):
        """
        Args:
            filter_fn (callable): Filter function that takes (dbox, bbox_prompt) and returns mask (N, M).
        """
        self.filter_fn = filter_fn
        self.stride=[8,16,32]
    
        self.dist2bbox=dist2bbox
        self.anchors=None
        self.strides=None
        self.reg_max=16
        self.dfl = dfl if dfl is not None else DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

    def pre_process(self, x ):
        """
        Args:
            x (List[Tensor]): Multi-level features.
        Returns:
            prompt_embed (Tensor): (B, M, embed_dim)
            mask (Tensor): (N, M) bool
        """
        shape = x[0].shape  # BCHW
        reg_max=self.reg_max
        dfl=self.dfl
        x_cat = torch.cat([xi.view(shape[0], shape[1], -1) for xi in x], 2)
        box = x_cat[:, :reg_max * 4, :]
        embed = x_cat[:, reg_max * 4:, :]

        # 2. Decode anchor boxes
        if self.anchors is None or self.strides is None:
            self.anchors, self.strides = (x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5))

        dfl_box= dfl(box)
        # Convert feature map coordinates to original image coordinates
        dbox = dist2bbox(dfl_box, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
        
        # Debug: check coordinate conversion
        print(f"pre_process - anchors range: {self.anchors.min():.3f} ~ {self.anchors.max():.3f}")
        print(f"pre_process - strides: {self.strides.unique()}")
        print(f"pre_process - dbox before strides: {dist2bbox(dfl_box, self.anchors.unsqueeze(0), xywh=True, dim=1).min():.3f} ~ {dist2bbox(dfl_box, self.anchors.unsqueeze(0), xywh=True, dim=1).max():.3f}")
        print(f"pre_process - dbox after strides: {dbox.min():.3f} ~ {dbox.max():.3f}")
        
        return dbox, embed

    def __call__(self, x):
        """
        
        Args:
            x(List[Tensor])
        """
        assert len(x) == len(self.stride), "Input feature layers do not match initialization"

        dbox, embed = self.pre_process(x)

        filter_mask= self.filter_fn(dbox, embed, self.anchors, self.strides)
        return dbox, embed, filter_mask 

