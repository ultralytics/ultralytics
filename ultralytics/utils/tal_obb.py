import torch

from .metrics import probiou
from .ops import xywhr2xyxyxyxy
from .tal import TaskAlignedAssigner


def select_candidates_in_rotated_gts(xy_centers, gt_bboxes, eps=1e-9):
    """
    Select the positive anchor center in gt.

    Args:
        xy_centers (Tensor): shape(h*w, 2)
        gt_bboxes (Tensor): shape(b, n_boxes, 5)

    Returns:
        (Tensor): shape(b, n_boxes, h*w)
    """
    bs, n_boxes, _ = gt_bboxes.shape
    # (b, n_boxes, 5) --> (b, n_boxes, 4, 2)
    corners = xywhr2xyxyxyxy(gt_bboxes).view(bs, n_boxes, 4, 2)
    # (b, n_boxes, 1, 2)
    a, b, _, d = corners.split(1, dim=-2)
    ab = b - a
    ad = d - a

    # (b, n_boxes, h*w, 2)
    ap = xy_centers - a
    norm_ab = (ab * ab).sum(dim=-1)
    norm_ad = (ad * ad).sum(dim=-1)
    ap_dot_ab = (ap * ab).sum(dim=-1)
    ap_dot_ad = (ap * ad).sum(dim=-1)
    is_in_box = (ap_dot_ab >= 0) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= 0) & (ap_dot_ad <= norm_ad)
    return is_in_box


class RotatedTaskAlignedAssigner(TaskAlignedAssigner):

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        """Compute alignment metric given predicted and ground truth bounding boxes."""
        na = pd_bboxes.shape[-2]
        mask_gt = mask_gt.bool()  # b, max_num_obj, h*w
        overlaps = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros([self.bs, self.n_max_boxes, na], dtype=pd_scores.dtype, device=pd_scores.device)

        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)  # 2, b, max_num_obj
        ind[0] = torch.arange(end=self.bs).view(-1, 1).expand(-1, self.n_max_boxes)  # b, max_num_obj
        ind[1] = gt_labels.squeeze(-1)  # b, max_num_obj
        # Get the scores of each grid for each gt cls
        bbox_scores[mask_gt] = pd_scores[ind[0], :, ind[1]][mask_gt]  # b, max_num_obj, h*w

        # (b, max_num_obj, 1, 5), (b, 1, h*w, 5)
        pd_boxes = pd_bboxes.unsqueeze(1).expand(-1, self.n_max_boxes, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes.unsqueeze(2).expand(-1, -1, na, -1)[mask_gt]
        overlaps[mask_gt] = probiou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        """Get in_gts mask, (b, max_num_obj, h*w)."""
        mask_in_gts = select_candidates_in_rotated_gts(anc_points, gt_bboxes)
        # Get anchor_align metric, (b, max_num_obj, h*w)
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt)
        # Get topk_metric mask, (b, max_num_obj, h*w)
        mask_topk = self.select_topk_candidates(align_metric, topk_mask=mask_gt.expand(-1, -1, self.topk).bool())
        # Merge all mask to a final mask, (b, max_num_obj, h*w)
        mask_pos = mask_topk * mask_in_gts * mask_gt

        return mask_pos, align_metric, overlaps


def dist2rbox(pred_distance, anchor_points):
    """
    Decode predicted object bounding box coordinates from anchor points and distribution.

    Args:
        anchor_points (torch.Tensor): Anchor points, (h*w, 2).
        distance (torch.Tensor): Predicted rotated distance, (bs, h*w, 5).
    """
    pred_distance, angle = pred_distance.split([4, 1], dim=-1)
    cos, sin = torch.cos(angle), torch.sin(angle)
    r_matrix = torch.cat([cos, -sin, sin, cos], dim=-1)
    # (bs, h*w, 2, 2)
    r_matrix = r_matrix.view(*angle.shape[:-1], 2, 2)
    wh = pred_distance[..., :2] + pred_distance[..., 2:]

    # (bs, h*w, 2)
    offset = (pred_distance[..., 2:] - pred_distance[..., :2]) / 2
    offset = torch.matmul(r_matrix, offset[..., None]).squeeze(-1)
    xy = anchor_points[..., :2] + offset
    return torch.cat([xy, wh, angle], dim=-1)
