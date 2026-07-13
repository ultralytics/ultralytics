import sys
import torch
# Add the project root to import local ultralytics package
sys.path.insert(0, "/Users/mac/yolo/ultralytics")

from ultralytics.utils.loss import BboxLoss

def test_bbox_loss_nwd():
    # 1. Initialize BboxLoss
    bbox_loss = BboxLoss(reg_max=16)
    
    # 2. Mock inputs matching YOLOv8 shapes:
    batch_size = 2
    num_anchors = 100
    reg_max = 16
    
    pred_dist = torch.randn(batch_size, num_anchors, 4 * reg_max)
    
    # pred_bboxes & target_bboxes: [B, num_anchors, 4] in xyxy format
    pred_bboxes = torch.rand(batch_size, num_anchors, 4) * 100
    pred_bboxes[..., 2:] += pred_bboxes[..., :2]  # ensure x2 > x1, y2 > y1
    
    target_bboxes = torch.rand(batch_size, num_anchors, 4) * 100
    target_bboxes[..., 2:] += target_bboxes[..., :2]
    
    anchor_points = torch.rand(num_anchors, 2) * 100
    
    # fg_mask: [B, num_anchors] boolean mask specifying foreground anchors
    fg_mask = torch.zeros(batch_size, num_anchors, dtype=torch.bool)
    fg_mask[0, :3] = True
    fg_mask[1, :2] = True
    
    # target_scores: [B, num_anchors, num_classes] to compute weight
    target_scores = torch.zeros(batch_size, num_anchors, 1)
    target_scores[fg_mask] = 1.0
    
    target_scores_sum = torch.tensor(fg_mask.sum().item())
    imgsz = torch.tensor([640, 640])
    stride = torch.tensor([8.0])
    
    print("Running forward pass of BboxLoss...")
    try:
        loss_iou, loss_dfl = bbox_loss(
            pred_dist=pred_dist,
            pred_bboxes=pred_bboxes,
            anchor_points=anchor_points,
            target_bboxes=target_bboxes,
            target_scores=target_scores,
            target_scores_sum=target_scores_sum,
            fg_mask=fg_mask,
            imgsz=imgsz,
            stride=stride
        )
        print("Success!")
        print(f"loss_iou: {loss_iou.item()} (shape: {loss_iou.shape})")
        print(f"loss_dfl: {loss_dfl.item()} (shape: {loss_dfl.shape})")
    except Exception as e:
        print("Failed with exception:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_bbox_loss_nwd()
