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


def test_bbox_loss_iou_calculation():
    """Test the IoU loss calculation specifically (lines 132-137 of loss.py)."""
    from ultralytics.utils.ops import xyxy2xywh
    from ultralytics.utils.inner_iou import inner_ciou

    # Initialize BboxLoss
    bbox_loss = BboxLoss(reg_max=16)

    # Inputs setup
    batch_size = 1
    num_anchors = 2
    
    # Coordinates in xyxy format
    # Anchor 0: perfect match, anchor 1: partial overlap
    pred_bboxes = torch.tensor([[[10.0, 10.0, 30.0, 30.0], [10.0, 10.0, 30.0, 30.0]]], requires_grad=True)
    target_bboxes = torch.tensor([[[10.0, 10.0, 30.0, 30.0], [20.0, 20.0, 40.0, 40.0]]])
    
    # Both are foreground anchors
    fg_mask = torch.tensor([[True, True]], dtype=torch.bool)
    
    # Target scores (used for weight)
    target_scores = torch.tensor([[[0.8], [0.5]]])
    target_scores_sum = torch.tensor(1.3)
    
    # DFL variables (dummy inputs for forward pass)
    pred_dist = torch.randn(batch_size, num_anchors, 4 * 16, requires_grad=True)
    anchor_points = torch.tensor([[20.0, 20.0], [30.0, 30.0]])
    imgsz = torch.tensor([640, 640])
    stride = torch.tensor([8.0])

    # Run forward pass of BboxLoss
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

    # 1. Manual check of IoU Loss
    weight_expected = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
    pred_expected = xyxy2xywh(pred_bboxes[fg_mask])
    target_expected = xyxy2xywh(target_bboxes[fg_mask])
    iou_expected = inner_ciou(pred_expected, target_expected, xywh=True)
    loss_iou_expected = ((1.0 - iou_expected) * weight_expected).sum() / target_scores_sum

    # Assert IoU Loss matches expected calculation exactly
    assert torch.allclose(loss_iou, loss_iou_expected)

    # 2. Assert specific values (e.g. first anchor has loss near 0.0)
    # The first anchor has identical pred and target, so iou should be 1.0
    iou_first_anchor = iou_expected[0].item()
    assert abs(iou_first_anchor - 1.0) < 1e-5
    
    # The second anchor has a partial overlap, so iou should be less than 1.0
    iou_second_anchor = iou_expected[1].item()
    assert iou_second_anchor < 1.0

    # 3. Test backward pass / gradient flow
    loss_iou.backward()
    assert pred_bboxes.grad is not None
    assert torch.any(pred_bboxes.grad != 0)


if __name__ == "__main__":
    test_bbox_loss_nwd()
    test_bbox_loss_iou_calculation()

