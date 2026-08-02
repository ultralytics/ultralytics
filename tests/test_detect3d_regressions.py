from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.augment import (
    Format,
    LetterBox,
    RandomFlip,
)
from ultralytics.data.dataset import YOLODataset, parse_calib_p2
from ultralytics.data.utils import verify_image_label
from ultralytics.engine.results import D3Params, Results
from ultralytics.models.yolo.detect3d.predict import Detection3DPredictor
from ultralytics.models.yolo.detect3d.train import Detection3DTrainer
from ultralytics.models.yolo.detect3d.utils import (
    find_detect3d_head,
    set_detect3d_quality_power,
)
from ultralytics.models.yolo.detect3d.val import (
    Detection3DMetrics,
    Detection3DValidator,
)
from ultralytics.nn.modules.head import Detect3D
from ultralytics.nn.tasks import Detection3DModel
from ultralytics.cfg import DEFAULT_CFG
from ultralytics.utils.geometry3d import (
    backproject_points,
    backproject_points_torch,
    boxes3d_corners_torch,
    decode_alpha_multibin,
    encode_alpha_multibin,
    project_points,
    project_points_torch,
    wrap_angle_torch,
)
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import (
    E2EDetect3DLoss,
    E2ELoss,
    v8Detection3DLoss,
    quality_focal_loss_with_logits,
    weighted_smooth_l1_loss_fp32,
)
from ultralytics.utils import nms


P2_WITH_TRANSLATION = np.array([[700.0, 3.0, 620.0, -350.0], [2.0, 710.0, 180.0, 70.0], [0.001, 0.002, 1.0, 0.2]])


def _synthetic_detect3d_batch(image: torch.Tensor) -> dict:
    """Build one geometrically valid target for Detect3D loss smoke tests."""
    p2 = np.array([[50.0, 0.0, 32.0, 0.0], [0.0, 50.0, 32.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    return {
        "img": image,
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25, 10.0, 0.0, 1.5, 1.8, 1.5, 4.0, 0.0]]),
        "d3_valid": torch.tensor([[True]]),
        "p2s_aug": [p2],
    }


class _SingleTargetAssigner:
    """Assign the only anchor to the only GT so 3D loss geometry can be tested deterministically."""

    def __call__(self, pd_scores, pd_bboxes, anchor_points, gt_labels, gt_bboxes, mask_gt):
        del pd_bboxes, anchor_points, mask_gt
        batch_size, num_anchors, num_classes = pd_scores.shape
        target_labels = gt_labels[:, :1].expand(batch_size, num_anchors, 1)
        target_bboxes = gt_bboxes[:, :1].expand(batch_size, num_anchors, 4)
        target_scores = pd_scores.new_zeros((batch_size, num_anchors, num_classes))
        target_scores[..., 0] = 1.0
        fg_mask = torch.ones((batch_size, num_anchors), dtype=torch.bool, device=pd_scores.device)
        target_gt_idx = torch.zeros((batch_size, num_anchors), dtype=torch.long, device=pd_scores.device)
        return target_labels, target_bboxes, target_scores, fg_mask, target_gt_idx


def _controlled_single_anchor_3d_loss(
    predicted_box_pixels: torch.Tensor, raw_d3: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return deterministic loss items for one 8x8 image and one geometrically exact 3D target."""
    model = Detection3DModel("yolo11n-3d.yaml", ch=3, nc=1, verbose=False)
    model.args = DEFAULT_CFG
    criterion = v8Detection3DLoss(model)
    criterion.assigner = _SingleTargetAssigner()
    criterion.bbox_loss = lambda pred_distri, *args: (
        pred_distri.sum() * 0.0,
        pred_distri.sum() * 0.0,
    )

    stride = float(criterion.stride[0])
    predicted_box = (predicted_box_pixels.float() / stride).reshape(1, 1, 4).requires_grad_()
    criterion.bbox_decode = lambda anchor_points, pred_dist: predicted_box

    head = model.model[-1]
    if raw_d3 is None:
        raw_d3 = torch.zeros(head.nr)
        raw_d3[2] = math.log(10.0)
        raw_d3[6] = -3.0
    raw_d3 = raw_d3.reshape(1, head.nr, 1).clone().requires_grad_()
    predictions = {
        "boxes": torch.zeros((1, 64, 1), requires_grad=True),
        "scores": torch.zeros((1, 1, 1), requires_grad=True),
        "feats": [torch.zeros((1, 1, 1, 1))],
        "d3_params": raw_d3,
    }
    p2 = np.array([[4.0, 0.0, 4.0, 0.0], [0.0, 4.0, 4.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    batch = {
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        # cx,cy,w,h,z,x,y_bottom,w3d,h3d,l3d,ry; the projected 3D center is exactly pixel (4,4).
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25, 10.0, 0.0, 0.75, 1.6, 1.5, 3.9, 0.0]]),
        "d3_valid": torch.tensor([[True]]),
        "p2s_aug": [p2],
    }
    loss_items = criterion.get_assigned_targets_and_loss(predictions, batch)[1]
    return loss_items, raw_d3, predicted_box


def test_full_p2_projection_round_trip_numpy_and_torch():
    xyz = np.array([[2.0, 1.5, 20.0], [-3.0, 2.0, 35.0]])
    uv = project_points(xyz, P2_WITH_TRANSLATION)
    np.testing.assert_allclose(backproject_points(uv, xyz[:, 2], P2_WITH_TRANSLATION), xyz, atol=1e-10)

    xyz_t = torch.tensor(xyz)
    p2_t = torch.tensor(P2_WITH_TRANSLATION)
    uv_t = project_points_torch(xyz_t, p2_t)
    torch.testing.assert_close(backproject_points_torch(uv_t, xyz_t[:, 2], p2_t), xyz_t, atol=1e-5, rtol=1e-5)


def test_project_points_torch_disables_outer_autocast():
    points = torch.tensor([[40.0, 5.0, 103.6]], dtype=torch.float32)
    p2 = torch.tensor(
        [
            [721.5377, 0.0, 609.5593, 45.0],
            [0.0, 721.5377, 172.854, 0.2],
            [0.0, 0.0, 1.0, 0.0027],
        ],
        dtype=torch.float32,
    )

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        projected = project_points_torch(points, p2)

    expected = project_points(points.numpy(), p2.numpy())
    assert projected.dtype == torch.float32
    assert torch.isfinite(projected).all()
    np.testing.assert_allclose(projected.numpy(), expected, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_project_points_torch_avoids_fp16_projection_overflow():
    points = torch.tensor([[40.0, 5.0, 103.6]], device="cuda")
    p2 = torch.tensor(
        [
            [721.5377, 0.0, 609.5593, 45.0],
            [0.0, 721.5377, 172.854, 0.2],
            [0.0, 0.0, 1.0, 0.0027],
        ],
        device="cuda",
    )

    with torch.autocast(device_type="cuda", dtype=torch.float16):
        projected = project_points_torch(points, p2)

    assert projected.dtype == torch.float32
    assert torch.isfinite(projected).all()


def test_letterbox_and_flip_compose_augmented_projection():
    p2 = np.array([[100.0, 0.0, 100.0, 5.0], [0.0, 100.0, 50.0, -2.0], [0.0, 0.0, 1.0, 0.1]])
    xyz = np.array([[1.0, 0.5, 10.0]])
    uv_original = project_points(xyz, p2)[0]
    labels = {
        "img": np.zeros((100, 200, 3), dtype=np.uint8),
        "instances": Instances(
            np.array([[0.5, 0.5, 0.2, 0.2]], dtype=np.float32),
            segments=np.zeros((0, 1000, 2), dtype=np.float32),
            bbox_format="xywh",
            normalized=True,
        ),
        "ratio_pad": (1.0, 1.0),
        "p2": p2.copy(),
        "p2_aug": p2.copy(),
        "d3_params": np.array([[10.0, 1.0, 1.0, 2.0, 1.5, 4.0, 0.2]], dtype=np.float32),
        "d3_valid": np.array([[True]]),
    }
    labels = LetterBox(new_shape=(320, 320), scaleup=True)(labels)
    p2_letterboxed = labels["p2_aug"].copy()
    labels = RandomFlip(p=1.0, direction="horizontal", flip_idx=[])(labels)

    xyz_mirrored = xyz.copy()
    xyz_mirrored[:, 0] *= -1
    uv_aug = project_points(xyz_mirrored, labels["p2_aug"])[0]
    np.testing.assert_allclose(uv_aug, [319.0 - uv_original[0] * 1.6, uv_original[1] * 1.6 + 80.0])
    h_flip = np.array([[-1.0, 0.0, 319.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    camera_mirror = np.diag([-1.0, 1.0, 1.0, 1.0])
    np.testing.assert_allclose(labels["p2_aug"], h_flip @ p2_letterboxed @ camera_mirror)
    expected_ry = np.pi - 0.2
    np.testing.assert_allclose(labels["d3_params"], [[10.0, -1.0, 1.0, 2.0, 1.5, 4.0, expected_ry]])
    alpha = 0.2 - np.arctan2(1.0, 10.0)
    mirrored_alpha = labels["d3_params"][0, 6] - np.arctan2(labels["d3_params"][0, 1], 10.0)

    def wrap(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    assert wrap(mirrored_alpha) == pytest.approx(wrap(np.pi - alpha))
    np.testing.assert_array_equal(labels["d3_valid"], [[True]])
    np.testing.assert_array_equal(labels["p2"], p2)


def test_detect3d_valid_mask_uses_native_box_height_and_configured_geometry(tmp_path):
    calib = tmp_path / "calib.txt"
    calib.write_text(
        "P2: " + " ".join(map(str, P2_WITH_TRANSLATION.reshape(-1))) + "\n",
        encoding="utf-8",
    )
    dataset = object.__new__(YOLODataset)
    dataset.use_3d = True
    dataset.use_obb = False
    dataset.min_3d_depth = 2.0
    dataset.max_3d_depth = 65.0
    dataset.min_3d_box_height = 25.0
    dataset.max_3d_center_offset = float("inf")  # isolate the depth/height boundary checks in this test
    rows = np.array(
        [
            [0.5, 0.5, 0.2, 0.25, 2.0, 0.0, 1.5, 1.8, 1.5, 4.0, 0.0],
            [0.5, 0.5, 0.2, 0.25, 1.99, 0.0, 1.5, 1.8, 1.5, 4.0, 0.0],
            [0.5, 0.5, 0.2, 0.25, 65.0, 0.0, 1.5, 1.8, 1.5, 4.0, 0.0],
            [0.5, 0.5, 0.2, 0.25, 65.01, 0.0, 1.5, 1.8, 1.5, 4.0, 0.0],
            [0.5, 0.5, 0.2, 0.25, 20.0, 0.0, 1.5, 0.0, 1.5, 4.0, 0.0],
            [0.5, 0.5, 0.2, 0.249, 20.0, 0.0, 1.5, 1.8, 1.5, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = dataset.update_labels_info(
        {
            "img": np.zeros((100, 200, 3), dtype=np.uint8),
            "cls": np.zeros((len(rows), 1), dtype=np.float32),
            "bboxes": rows,
            "segments": [],
            "keypoints": None,
            "bbox_format": "xywh",
            "normalized": True,
            "calib_path": str(calib),
            "ori_shape": (100, 200),
            "resized_shape": (100, 200),
            "ratio_pad": (1.0, 1.0),
        }
    )
    np.testing.assert_array_equal(labels["d3_valid"].squeeze(1), [True, False, True, False, False, False])

    formatted = Format(bbox_format="xywh", normalize=True, return_3d=True, batch_idx=True)(labels)
    assert formatted["d3_valid"].dtype == torch.bool
    assert formatted["d3_valid"].shape == (len(rows), 1)
    batch = YOLODataset.collate_fn([formatted])
    torch.testing.assert_close(batch["d3_valid"], formatted["d3_valid"])
    assert batch["bboxes"].shape == (len(rows), 11)


def test_detect3d_valid_mask_rejects_extreme_truncated_center_target(tmp_path):
    calib = tmp_path / "calib.txt"
    calib.write_text("P2: 100 0 100 0 0 100 50 0 0 0 1 0\n", encoding="utf-8")
    dataset = object.__new__(YOLODataset)
    dataset.use_3d = True
    dataset.use_obb = False
    dataset.min_3d_depth = 2.0
    dataset.max_3d_depth = 65.0
    dataset.min_3d_box_height = 25.0
    dataset.max_3d_center_offset = 0.5
    # First target projects to the 2D box center. The second mimics a narrow boundary sliver whose full 3D center lies
    # many box widths away; it must retain 2D supervision but be excluded from all 3D losses.
    rows = np.array(
        [
            [0.5, 0.5, 0.2, 0.3, 10.0, 0.0, 1.5, 1.8, 1.5, 4.0, 0.0],
            [0.975, 0.5, 0.05, 0.3, 10.0, 20.0, 1.5, 1.8, 1.5, 4.0, 0.0],
        ],
        dtype=np.float32,
    )
    labels = dataset.update_labels_info(
        {
            "img": np.zeros((100, 200, 3), dtype=np.uint8),
            "cls": np.zeros((len(rows), 1), dtype=np.float32),
            "bboxes": rows,
            "segments": [],
            "keypoints": None,
            "bbox_format": "xywh",
            "normalized": True,
            "calib_path": str(calib),
            "ori_shape": (100, 200),
            "resized_shape": (100, 200),
            "ratio_pad": (1.0, 1.0),
        }
    )
    np.testing.assert_array_equal(labels["d3_valid"].squeeze(1), [True, False])
    assert len(labels["instances"]) == 2


def test_detect3d_rejects_spatial_albumentations_that_do_not_update_projection():
    albumentations = pytest.importorskip("albumentations")
    dataset = object.__new__(YOLODataset)
    dataset.augment = True
    dataset.rect = True
    dataset.use_3d = True
    dataset.imgsz = 1280
    dataset.use_segments = dataset.use_keypoints = dataset.use_obb = False
    hyp = SimpleNamespace(
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        augmentations=[albumentations.Crop(x_min=0, y_min=0, x_max=50, y_max=100, p=1.0)],
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        mask_ratio=4,
        overlap_mask=True,
        bgr=0.0,
    )

    with pytest.raises(ValueError, match="only supports photometric Albumentations"):
        dataset.build_transforms(hyp)


def test_detect3d_rejects_nested_spatial_albumentations():
    albumentations = pytest.importorskip("albumentations")
    dataset = object.__new__(YOLODataset)
    dataset.augment = True
    dataset.rect = True
    dataset.use_3d = True
    dataset.imgsz = 1280
    dataset.use_segments = dataset.use_keypoints = dataset.use_obb = False
    hyp = SimpleNamespace(
        mosaic=0.0,
        mixup=0.0,
        cutmix=0.0,
        augmentations=[
            albumentations.OneOf(
                [albumentations.Crop(x_min=0, y_min=0, x_max=50, y_max=100, p=1.0)],
                p=1.0,
            )
        ],
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        fliplr=0.0,
        mask_ratio=4,
        overlap_mask=True,
        bgr=0.0,
    )

    with pytest.raises(ValueError, match="only supports photometric Albumentations"):
        dataset.build_transforms(hyp)


def test_dataset_initial_resize_is_in_augmented_projection(tmp_path):
    calib = tmp_path / "calib.txt"
    calib.write_text(
        "P2: " + " ".join(map(str, P2_WITH_TRANSLATION.reshape(-1))) + "\n",
        encoding="utf-8",
    )
    dataset = object.__new__(YOLODataset)
    dataset.use_3d = True
    dataset.use_obb = False
    labels = dataset.update_labels_info(
        {
            "bboxes": np.array(
                [[0.5, 0.5, 0.2, 0.2, 20.0, 1.0, 1.5, 1.8, 1.5, 4.0, 0.0]],
                dtype=np.float32,
            ),
            "segments": [],
            "keypoints": None,
            "bbox_format": "xywh",
            "normalized": True,
            "calib_path": str(calib),
            "ori_shape": (375, 1242),
            "resized_shape": (193, 640),
        }
    )
    expected = np.diag([640 / 1242, 193 / 375, 1.0]) @ P2_WITH_TRANSLATION
    np.testing.assert_allclose(labels["p2_aug"], expected)


def test_multiscale_preprocess_updates_projection_and_ratio_pad():
    trainer = object.__new__(Detection3DTrainer)
    trainer.device = torch.device("cpu")
    trainer.args = SimpleNamespace(multi_scale=0.5, imgsz=320)
    trainer.stride = 32
    p2 = np.arange(12, dtype=np.float64).reshape(3, 4)
    batch = {
        "img": torch.zeros((1, 3, 320, 320), dtype=torch.uint8),
        "p2s_aug": [p2.copy()],
        "ratio_pad": (((0.25, 0.25), (0.0, 100.0)),),
    }
    with patch("ultralytics.models.yolo.detect.train.random.randrange", return_value=384):
        output = trainer.preprocess_batch(batch)
    assert output["img"].shape[-2:] == (384, 384)
    expected = np.diag([1.2, 1.2, 1.0]) @ p2
    np.testing.assert_allclose(output["p2s_aug"][0], expected)
    assert output["ratio_pad"] == (((0.3, 0.3), (0.0, 120.0)),)


def test_detect3d_training_label_plot_uses_only_2d_box_columns(tmp_path):
    trainer = object.__new__(Detection3DTrainer)
    boxes = np.arange(11, dtype=np.float32).reshape(1, 11)
    trainer.train_loader = SimpleNamespace(
        dataset=SimpleNamespace(labels=[{"bboxes": boxes, "cls": np.array([[0.0]], dtype=np.float32)}])
    )
    trainer.data = {"names": {0: "Car"}}
    trainer.save_dir = tmp_path
    trainer.on_plot = None

    with patch("ultralytics.models.yolo.detect3d.train.plot_labels") as plot_labels:
        trainer.plot_training_labels()

    plotted_boxes = plot_labels.call_args.args[0]
    assert plotted_boxes.shape == (1, 4)
    np.testing.assert_array_equal(plotted_boxes, boxes[:, :4])


def test_results_preserve_3d_params_and_plot_with_explicit_calibration():
    image = np.zeros((375, 1242, 3), dtype=np.uint8)
    p2 = np.array([[721.5, 0.0, 609.5, 0.0], [0.0, 721.5, 172.8, 0.0], [0.0, 0.0, 1.0, 0.0]])
    xyz_center = np.array([[0.0, 1.0, 20.0]])
    center = project_points(xyz_center, p2)[0]
    d3 = torch.tensor([[center[0], center[1], 20.0, 0.0, 1.0, 1.5, 1.8, 4.0]])
    result = Results(
        image,
        path="frame.png",
        names={0: "car"},
        boxes=torch.tensor([[500.0, 140.0, 740.0, 300.0, 0.9, 0.0]]),
        d3_params=d3,
    )

    assert isinstance(result.d3_params, D3Params)
    torch.testing.assert_close(result.cpu().d3_params.data, d3)
    np.testing.assert_allclose(result.numpy().d3_params.data, d3.numpy())
    np.testing.assert_allclose(result[:1].d3_params.data.numpy(), d3.numpy())
    plotted = result.plot(p2=p2)
    assert plotted.shape == image.shape and np.count_nonzero(plotted) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_results_cpu_numpy_converts_all_detection_data():
    image = np.zeros((32, 64, 3), dtype=np.uint8)
    result = Results(
        image,
        path="frame.png",
        names={0: "car"},
        boxes=torch.tensor([[1.0, 2.0, 10.0, 20.0, 0.9, 0.0]], device="cuda"),
        d3_params=torch.zeros((1, 8), device="cuda"),
    )
    converted = result.cpu().numpy()
    assert isinstance(converted.boxes.data, np.ndarray)
    assert isinstance(converted.d3_params.data, np.ndarray)


def test_predictor_maps_3d_center_to_native_image():
    predictor = object.__new__(Detection3DPredictor)
    predictor.model = SimpleNamespace(names={0: "car"})
    pred = torch.tensor(
        [
            [
                100.0,
                200.0,
                300.0,
                400.0,
                0.9,
                0.0,
                320.0,
                100.0,
                20.0,
                0.0,
                1.0,
                1.5,
                1.8,
                4.0,
            ]
        ]
    )
    result = predictor.construct_result(pred, torch.zeros(1, 3, 640, 640), np.zeros((375, 1242, 3)), "x.png")
    expected_center = torch.tensor([621.0, -238.6969])
    torch.testing.assert_close(result.d3_params.center[0], expected_center)


def test_predictor_attaches_single_calibration_and_plots_3d(tmp_path):
    calib = tmp_path / "camera.txt"
    calib.write_text("P2: 100 0 320 0 0 100 200 0 0 0 1 0\n", encoding="utf-8")
    predictor = object.__new__(Detection3DPredictor)
    predictor.model = SimpleNamespace(names={0: "car"})
    predictor.args = SimpleNamespace(calib=str(calib))
    pred = torch.tensor([[200.0, 140.0, 440.0, 300.0, 0.9, 0.0, 320.0, 200.0, 20.0, 0.0, 1.0, 1.5, 1.8, 4.0]])

    result = predictor.construct_result(pred, torch.zeros(1, 3, 640, 640), np.zeros((640, 640, 3)), "frame.png")

    np.testing.assert_allclose(result.p2, parse_calib_p2(str(calib)))
    plotted = result.plot()
    assert np.any(np.all(plotted == (0, 255, 0), axis=2))


def test_predictor_matches_calibration_directory_by_image_stem(tmp_path):
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    calib = calib_dir / "000001.txt"
    calib.write_text("P2: 100 0 320 0 0 100 200 0 0 0 1 0\n", encoding="utf-8")
    predictor = object.__new__(Detection3DPredictor)
    predictor.model = SimpleNamespace(names={0: "car"})
    predictor.args = SimpleNamespace(calib=str(calib_dir))
    pred = torch.empty((0, 14))

    result = predictor.construct_result(
        pred, torch.zeros(1, 3, 640, 640), np.zeros((640, 640, 3)), "/images/000001.png"
    )

    np.testing.assert_allclose(result.p2, parse_calib_p2(str(calib)))


def test_predictor_rejects_missing_stem_matched_calibration(tmp_path):
    calib_dir = tmp_path / "calib"
    calib_dir.mkdir()
    predictor = object.__new__(Detection3DPredictor)
    predictor.model = SimpleNamespace(names={0: "car"})
    predictor.args = SimpleNamespace(calib=str(calib_dir))

    with pytest.raises(FileNotFoundError, match=r"000001\.txt"):
        predictor.construct_result(
            torch.empty((0, 14)),
            torch.zeros(1, 3, 640, 640),
            np.zeros((640, 640, 3)),
            "/images/000001.png",
        )


def test_detect3d_validator_keeps_only_the_highest_class_per_anchor():
    validator = object.__new__(Detection3DValidator)
    validator.args = SimpleNamespace(conf=0.1, iou=0.7, single_cls=False, agnostic_nms=False, max_det=300)
    validator.nc = 2
    validator.end2end = False
    prediction = torch.zeros((1, 4 + validator.nc + 8, 1))
    prediction[0, :4, 0] = torch.tensor([50.0, 50.0, 20.0, 20.0])
    prediction[0, 4:6, 0] = torch.tensor([0.9, 0.8])
    prediction[0, 6:, 0] = torch.arange(8)

    output = validator.postprocess(prediction)[0]

    assert output["cls"].tolist() == [0.0]
    torch.testing.assert_close(output["extra"], torch.arange(8, dtype=torch.float32).view(1, 8))


def test_detect3d_validator_aligns_raw_q3d_through_nms_without_changing_eight_value_output():
    validator = object.__new__(Detection3DValidator)
    validator.args = SimpleNamespace(
        conf=0.1,
        iou=0.7,
        single_cls=False,
        agnostic_nms=False,
        max_det=300,
        quality3d_power=0.25,
    )
    validator.nc = 1
    validator.end2end = False
    validator.training = False
    prediction = torch.zeros((1, 4 + validator.nc + 8, 3))
    prediction[0, :4] = torch.tensor([[10.0, 30.0, 50.0], [10.0, 30.0, 50.0], [5.0, 5.0, 5.0], [5.0, 5.0, 5.0]])
    prediction[0, 4] = torch.tensor([0.2, 0.9, 0.05])  # last anchor is removed before NMS
    raw_d3 = torch.zeros((1, 7, 3))
    raw_d3[0, 6] = torch.tensor([-2.0, 2.0, 0.5])
    raw = {"d3_params": raw_d3, "scores": torch.zeros((1, 1, 3))}

    output = validator.postprocess((prediction, raw))[0]

    assert output["extra"].shape == (2, 8)
    torch.testing.assert_close(output["q3d"], raw_d3[0, 6, [1, 0]].sigmoid())


def test_yolo26_validator_aligns_raw_q3d_through_end2end_topk():
    validator = object.__new__(Detection3DValidator)
    validator.args = SimpleNamespace(
        conf=0.0,
        iou=0.7,
        single_cls=False,
        agnostic_nms=False,
        max_det=3,
        quality3d_power=0.25,
    )
    validator.nc = 1
    validator.end2end = True
    validator.training = False
    raw_d3 = torch.zeros((1, 7, 4))
    raw_d3[0, 6] = torch.tensor([-3.0, 2.0, 0.0, -1.0])
    raw_scores = torch.tensor([[[3.0, 1.5, 2.0, -2.0]]])
    quality = raw_d3[:, 6].sigmoid()
    ranking = (raw_scores.sigmoid() * quality.unsqueeze(1).pow(validator.args.quality3d_power)).max(1).values
    topk_indices = ranking.topk(3, dim=1).indices
    # The native end-to-end head has already emitted these three rows in top-k order.
    prediction = torch.zeros((1, 3, 14))
    prediction[0, :, 4] = ranking.gather(1, topk_indices)
    raw = {"one2one": {"d3_params": raw_d3, "scores": raw_scores}}

    output = validator.postprocess((prediction, raw))[0]

    assert output["extra"].shape == (3, 8)
    torch.testing.assert_close(output["q3d"], quality.gather(1, topk_indices)[0])


def test_detect3d_inference_preserves_raw_outputs_for_validation_loss():
    head = Detect3D(nc=3, ch=(16, 32, 64))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.eval()
    features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    raw = head.forward_head(features, **head.one2many)
    raw_d3 = raw["d3_params"].clone()
    assert raw_d3.shape[1] == head.nr and "d3_aux" not in raw

    # Make the decode deterministic with class 1 dimensions and a large Truck-sized residual.
    raw["scores"].fill_(-10.0)
    raw["scores"][:, 1].fill_(10.0)
    raw["d3_params"].zero_()
    raw["d3_params"][:, 0].fill_(0.1)
    raw["d3_params"][:, 1].fill_(-0.2)
    raw["d3_params"][:, 2].fill_(math.log(20.0))
    target_dims = torch.tensor([4.0, 3.0, 16.0])
    raw["d3_params"][:, 3:6] = torch.log(target_dims / head.dim_priors[1])[:, None]
    raw_d3 = raw["d3_params"].clone()

    decoded = head._inference(raw)

    torch.testing.assert_close(raw["d3_params"], raw_d3)
    boxes = head._get_decode_boxes(raw)
    expected_center = boxes[:, :2] + raw_d3[:, :2] * boxes[:, 2:4]
    torch.testing.assert_close(decoded[:, 7:9], expected_center)
    torch.testing.assert_close(decoded[:, 9], torch.full_like(decoded[:, 9], 20.0))
    torch.testing.assert_close(decoded[:, 12:15], target_dims.view(1, 3, 1).expand_as(decoded[:, 12:15]))

    half_raw = {key: value for key, value in raw.items()}
    half_raw["d3_params"] = raw_d3.half()
    half_raw["d3_params"][:, 2] = -100.0
    half_decoded = head._inference(half_raw)
    assert torch.isfinite(half_decoded[:, 9]).all()


def test_detect3d_direct_layout_and_decode():
    head = Detect3D(nc=1, ch=(16, 32, 64))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.eval()
    features = [
        torch.randn(1, 16, 8, 8),
        torch.randn(1, 32, 4, 4),
        torch.randn(1, 64, 2, 2),
    ]
    raw = head.forward_head(features, **head.one2many)

    assert head.geo_channels == 7 and head.nr == 31
    assert raw["d3_params"].shape[1] == 31
    raw["d3_params"].zero_()
    raw["d3_params"][:, 2].fill_(math.log(20.0))
    decoded = head._inference(raw)
    depth = decoded[:, 4 + head.nc + 2]
    torch.testing.assert_close(depth, torch.full_like(depth, 20.0))


def test_detect3d_bias_init_uses_log_depth_and_neutral_multibin_priors():
    head = Detect3D(nc=3, ch=(16, 32, 64))
    head.stride = torch.tensor([8.0, 16.0, 32.0])
    head.bias_init()

    for branch in head.cv4:
        assert branch.primary.bias[2].item() == pytest.approx(math.log(18.0))
        torch.testing.assert_close(branch.primary.bias[3:6], torch.zeros(3))
        assert branch.primary.bias[6].item() == pytest.approx(-3.0)
        torch.testing.assert_close(branch.auxiliary.bias, torch.zeros(16))
    for branch in head.cv5:
        torch.testing.assert_close(branch.output.bias, torch.zeros(24))


def test_quality3d_power_reaches_native_models_through_runtime_wrappers():
    model = Detection3DModel("yolo26n-3d.yaml", ch=3, nc=1, verbose=False)
    wrapper = SimpleNamespace(model=SimpleNamespace(model=model))

    assert find_detect3d_head(wrapper) is model.model[-1]
    assert set_detect3d_quality_power(wrapper, 0.1)
    assert model.model[-1].quality3d_power == pytest.approx(0.1)
    with pytest.raises(ValueError, match="non-negative"):
        set_detect3d_quality_power(wrapper, -0.1)


def test_detect3d_loss_uses_detached_matched_prediction_as_box_reference():
    exact_loss, _, exact_box = _controlled_single_anchor_3d_loss(torch.tensor([3.0, 3.0, 5.0, 5.0]))
    shifted_loss, shifted_raw, shifted_box = _controlled_single_anchor_3d_loss(torch.tensor([4.0, 3.0, 6.0, 5.0]))

    assert exact_loss[3].item() == pytest.approx(0.0, abs=1e-7)
    assert shifted_loss[3].item() > 0.1
    shifted_loss[3].backward()
    assert shifted_raw.grad is not None and torch.count_nonzero(shifted_raw.grad[:, :2]) > 0
    assert shifted_box.grad is None  # 3D losses must not push the independently optimized 2D box head.
    assert exact_box.grad is None

    # Invalid and zero-area early predictions fall back to the finite GT reference instead of creating NaN/Inf targets.
    for malformed in (
        torch.tensor([float("nan"), 3.0, 5.0, 5.0]),
        torch.tensor([4.0, 3.0, 4.0, 5.0]),
    ):
        malformed_loss, _, _ = _controlled_single_anchor_3d_loss(malformed)
        assert torch.isfinite(malformed_loss).all()
        assert malformed_loss[3].item() == pytest.approx(0.0, abs=1e-7)


def test_disentangled_rotation_corner_uses_gt_ray_but_full_center_uses_predicted_box():
    loss, _, _ = _controlled_single_anchor_3d_loss(torch.tensor([4.0, 3.0, 6.0, 5.0]))

    # The shifted reference back-projects to x=2.5m. Only the center third of the disentangled corner objective should
    # be non-zero; a perfect alpha must not acquire a second rotation penalty from the erroneous predicted center ray.
    center_element = torch.nn.functional.smooth_l1_loss(
        torch.tensor([2.5, 0.0, 0.0]), torch.zeros(3), beta=1.0, reduction="mean"
    )
    expected = center_element / 3.0 * DEFAULT_CFG.d3_geometry_gain
    assert loss[7].item() == pytest.approx(expected.item(), rel=1e-5)


def test_quality3d_treats_yaw_plus_pi_as_the_same_cuboid_geometry():
    head = Detect3D(nc=1, ch=(16, 32, 64))
    raw = torch.zeros(head.nr)
    raw[2] = math.log(10.0)
    raw[6] = 2.0
    raw[head.geo_channels : head.geo_channels + 12] = -20.0
    raw[head.geo_channels + 6] = 20.0  # bin 6 decodes alpha=pi while GT alpha=0
    loss, _, _ = _controlled_single_anchor_3d_loss(torch.tensor([3.0, 3.0, 5.0, 5.0]), raw)

    expected = quality_focal_loss_with_logits(torch.tensor([[2.0]]), torch.tensor([[1.0]]))
    assert loss[9].item() == pytest.approx(expected.item(), rel=1e-5)


def test_detect3d_end2end_postprocess_keeps_only_best_class_per_anchor():
    head = Detect3D(nc=3, end2end=True, ch=(16, 32, 64))
    head.max_det = 10
    predictions = torch.zeros((1, 2, 4 + head.nc + head.nd))
    predictions[0, 0, :4] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    predictions[0, 1, :4] = torch.tensor([5.0, 6.0, 7.0, 8.0])
    predictions[0, 0, 4:7] = torch.tensor([0.90, 0.89, 0.88])
    predictions[0, 1, 4:7] = torch.tensor([0.80, 0.10, 0.20])
    predictions[0, 0, 7:] = 11.0
    predictions[0, 1, 7:] = 22.0

    result = head.postprocess(predictions)

    assert result.shape == (1, 2, 6 + head.nd)
    torch.testing.assert_close(result[0, :, 4], torch.tensor([0.90, 0.80]))
    torch.testing.assert_close(result[0, :, 5], torch.tensor([0.0, 0.0]))
    torch.testing.assert_close(result[0, 0, 6:], torch.full((head.nd,), 11.0))
    torch.testing.assert_close(result[0, 1, 6:], torch.full((head.nd,), 22.0))


def test_multibin_round_trip_and_camera_corner_geometry():
    alpha = torch.tensor([-torch.pi + 1e-4, -0.4, 0.0, 0.7, torch.pi - 1e-4])
    bin_index, residual = encode_alpha_multibin(alpha, 12)
    logits = torch.full((len(alpha), 12), -20.0)
    logits.scatter_(1, bin_index[:, None], 20.0)
    residual_logits = torch.zeros_like(logits)
    half_bin = torch.pi / 12
    residual_logits.scatter_(
        1,
        bin_index[:, None],
        torch.atanh((residual / half_bin).clamp(-0.999, 0.999))[:, None],
    )
    decoded = decode_alpha_multibin(logits, residual_logits)
    torch.testing.assert_close(wrap_angle_torch(decoded - alpha), torch.zeros_like(alpha), atol=1e-4, rtol=0)

    center = torch.tensor([[0.0, 1.0, 10.0]])
    dims = torch.tensor([[2.0, 2.0, 4.0]])
    corners = boxes3d_corners_torch(center, dims, torch.tensor([0.0]))
    assert corners.shape == (1, 8, 3)
    torch.testing.assert_close(corners[..., 0].amin(1), torch.tensor([-2.0]))
    torch.testing.assert_close(corners[..., 0].amax(1), torch.tensor([2.0]))
    torch.testing.assert_close(corners[..., 1].amin(1), torch.tensor([0.0]))
    torch.testing.assert_close(corners[..., 1].amax(1), torch.tensor([2.0]))


def test_corner_rotation_uses_target_bin_but_quality_uses_inference_bin():
    head = Detect3D(nc=1, ch=(16, 32, 64))
    raw = torch.zeros(head.nr)
    raw[2] = math.log(10.0)
    raw[6] = 2.0
    raw[head.geo_channels : head.geo_channels + head.num_alpha_bins] = -20.0
    raw[head.geo_channels + 3] = 20.0  # inference decodes alpha=pi/2 while GT alpha=0
    captured = {}
    original_quality_loss = quality_focal_loss_with_logits

    def capture_quality(logits, target, beta=2.0):
        captured["target"] = target.detach().clone()
        return original_quality_loss(logits, target, beta)

    with patch(
        "ultralytics.utils.loss.quality_focal_loss_with_logits",
        side_effect=capture_quality,
    ):
        loss, _, _ = _controlled_single_anchor_3d_loss(torch.tensor([3.0, 3.0, 5.0, 5.0]), raw)

    # Teacher-forcing only the residual used by the differentiable corner loss keeps that objective stable, while
    # q3d still sees the wrong argmax-decoded orientation that will actually be produced at inference.
    assert loss[7].item() == pytest.approx(0.0, abs=1e-6)
    assert loss[5].item() > 1.0
    assert captured["target"].item() < 0.8


def test_center3d_smooth_l1_uses_fp32_accumulation():
    count = 70_000
    pred = torch.zeros((count, 2), dtype=torch.float16)
    target = torch.full((count, 2), 1_000.0, dtype=torch.float16)
    weight = torch.ones((count, 1), dtype=torch.float16)

    loss = weighted_smooth_l1_loss_fp32(pred, target, weight, float(count))

    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(999.5)


def test_yolo26_detect3d_end2end_loss_and_inference():
    model = Detection3DModel("yolo26n-3d.yaml", ch=3, nc=1, verbose=False)
    model.args = DEFAULT_CFG
    image = torch.randn(1, 3, 64, 64)
    batch = _synthetic_detect3d_batch(image)

    model.train()
    predictions = model.predict(image)
    assert predictions.keys() == {"one2many", "one2one"}
    assert "d3_params" in predictions["one2many"] and "d3_params" in predictions["one2one"]

    loss, loss_items = model.loss(batch, predictions)
    assert isinstance(model.criterion, E2ELoss)
    assert type(model.criterion) is E2EDetect3DLoss
    assert set(loss_items) == {
        "box_loss",
        "cls_loss",
        "l1_loss",
        "center3d_loss",
        "depth_loss",
        "alpha_loss",
        "dim_loss",
        "corner3d_loss",
        "keypoint3d_loss",
        "quality3d_loss",
    }
    assert torch.isfinite(loss).all() and all(torch.isfinite(value) for value in loss_items.values())
    loss.sum().backward()

    head = model.model[-1]
    assert model.yaml["scale"] == "n"
    # The two end-to-end branches use different 2D boxes and TAL assignments.  Their box-relative
    # 3D center targets are therefore different, so geometry towers must be independent.
    assert head.one2many["d3_geo_head"] is not head.one2one["d3_geo_head"]
    assert head.one2many["d3_dir_head"] is not head.one2one["d3_dir_head"]
    assert head.cv4[0].primary.weight.grad is not None
    assert head.cv4[0].primary.weight.grad.shape[0] == head.geo_channels
    assert head.one2one_cv4[0].primary.weight.grad is not None
    assert head.one2one_cv4[0].primary.weight.grad.shape[0] == head.geo_channels
    assert head.geo_channels == 7
    assert torch.count_nonzero(head.cv4[0].primary.weight.grad[2]) > 0
    assert torch.count_nonzero(head.cv4[0].primary.weight.grad[6]) > 0
    assert torch.count_nonzero(head.cv4[0].auxiliary.weight.grad) > 0
    assert torch.count_nonzero(head.cv5[0].output.weight.grad) > 0
    assert torch.count_nonzero(head.one2one_cv4[0].auxiliary.weight.grad) > 0
    assert torch.count_nonzero(head.one2one_cv5[0].output.weight.grad) > 0

    model.eval()
    decoded, raw = model.predict(image)
    assert raw.keys() == {"one2many", "one2one"}
    assert decoded.ndim == 3 and decoded.shape[2] == 14  # xyxy, confidence, class, 8 Detect3D values
    outputs = nms.non_max_suppression(decoded, conf_thres=0.0, nc=1, end2end=True)
    assert len(outputs) == 1 and outputs[0].shape[1] == 14
    assert torch.isfinite(outputs[0]).all()


@pytest.mark.parametrize("config", ("yolo11n-3d.yaml", "yolo26n-3d.yaml"))
def test_detect3d_fuse_preserves_eval_output_and_end2end_deploy_towers(config):
    torch.manual_seed(0)
    model = Detection3DModel(config, ch=3, nc=3, verbose=False).eval()
    image = torch.randn(1, 3, 64, 96)

    with torch.inference_mode():
        before = model.predict(image)[0]
        model.fuse(verbose=False)
        after, raw = model.predict(image)

    torch.testing.assert_close(after, before)
    head = model.model[-1]
    if head.end2end:
        assert head.cv2 is None and head.cv3 is None and head._fused3d
        assert raw["one2many"] == {}
        assert raw["one2one"]["d3_params"].shape[1] == head.nr
        assert head.cv4 is None and head.cv5 is None
        assert head.one2one["d3_geo_head"] is head.one2one_cv4
        assert head.one2one["d3_dir_head"] is head.one2one_cv5
    else:
        assert head.cv4 is not None and head.cv5 is not None
        assert raw["d3_params"].shape[1] == head.nr


def test_yolo11_detect3d_uses_monocon_lite_smooth_l1_with_eight_decoded_outputs():
    model = Detection3DModel("yolo11n-3d.yaml", ch=3, nc=1, verbose=False)
    model.args = DEFAULT_CFG
    image = torch.randn(1, 3, 64, 64)

    model.train()
    predictions = model.predict(image)
    assert predictions.keys() >= {"boxes", "scores", "feats", "d3_params"}
    loss, loss_items = model.loss(_synthetic_detect3d_batch(image), predictions)
    assert isinstance(model.criterion, v8Detection3DLoss)
    assert "dfl_loss" in loss_items and "depth_loss" in loss_items
    assert torch.isfinite(loss).all()
    loss.sum().backward()

    head = model.model[-1]
    assert predictions["d3_params"].shape[1] == head.nr
    assert predictions["d3_aux"].shape[1] == 16
    assert head.cv4[0].primary.weight.shape[0] == head.geo_channels
    assert head.geo_channels == 7
    assert torch.count_nonzero(head.cv4[0].primary.weight.grad[2]) > 0
    assert torch.count_nonzero(head.cv4[0].primary.weight.grad[6]) > 0
    assert torch.count_nonzero(head.cv4[0].auxiliary.weight.grad) > 0


def test_results_save_txt_keeps_detect3d_parameters(tmp_path):
    result = Results(
        np.zeros((32, 64, 3), dtype=np.uint8),
        path="frame.png",
        names={0: "Car"},
        boxes=torch.tensor([[1.0, 2.0, 10.0, 20.0, 0.9, 0.0]]),
        d3_params=torch.arange(8, dtype=torch.float32).view(1, 8),
    )
    output = tmp_path / "prediction.txt"

    result.save_txt(output, save_conf=True)

    values = [float(value) for value in output.read_text(encoding="utf-8").split()]
    assert len(values) == 14  # class + normalized xywh + 8 Detect3D values + confidence
    assert values[5:13] == list(range(8))
    assert values[-1] == pytest.approx(0.9)


def test_results_summary_keeps_detect3d_parameters():
    result = Results(
        np.zeros((100, 200, 3), dtype=np.uint8),
        path="frame.png",
        names={0: "Car"},
        boxes=torch.tensor([[10.0, 20.0, 30.0, 40.0, 0.9, 0.0]]),
        d3_params=torch.tensor([[50.0, 25.0, 20.0, 0.0, 1.0, 1.5, 1.6, 4.0]]),
    )

    summary = result.summary(normalize=True)[0]

    assert summary["box3d"] == {
        "center_x": 0.25,
        "center_y": 0.25,
        "depth": 20.0,
        "sin_alpha": 0.0,
        "cos_alpha": 1.0,
        "height": 1.5,
        "width": 1.6,
        "length": 4.0,
    }
    assert '"box3d"' in result.to_json(normalize=True)


def test_detect3d_fitness_uses_generic_3d_map50_only():
    metrics = Detection3DMetrics(names={0: "Car"})
    metrics.box.p = np.array([0.6])
    metrics.box.r = np.array([0.6])
    metrics.box.all_ap = np.full((1, 10), 0.4)
    metrics.box.ap_class_index = np.array([0])
    metrics.d3.p = np.array([0.7])
    metrics.d3.r = np.array([0.6])
    metrics.d3.all_ap = np.array([[0.35, *([0.15] * 9)]])
    metrics.d3.ap_class_index = np.array([0])
    metrics.d3_results = {"3d/dist_MAE": 5.0, "3d/xc_MAE": 1.0, "3d/ry_deg_MAE": 30.0}
    baseline = metrics.fitness

    metrics.d3_results = {"3d/dist_MAE": 2.5, "3d/xc_MAE": 0.5, "3d/ry_deg_MAE": 15.0}

    assert metrics.fitness == baseline == pytest.approx(0.35)

    metrics.d3_results["kitti/AP3D_R40_moderate"] = 42.5
    assert metrics.fitness == pytest.approx(0.35)


def test_detect3d_generic_process_reports_all_three_classes_and_3d_keys():
    metrics = Detection3DMetrics(names={0: "Car", 1: "Van", 2: "Truck"})
    for image_index, class_id in enumerate((0, 1, 2)):
        stat = {
            "tp": np.ones((1, 10), dtype=bool),
            "conf": np.array([0.9 - image_index * 0.1]),
            "pred_cls": np.array([class_id], dtype=np.float32),
            "target_cls": np.array([class_id], dtype=np.float32),
            "target_img": np.array([class_id], dtype=np.float32),
            "im_name": f"{image_index}.png",
            "tp_d3": np.ones((1, 10), dtype=bool),
            "target_cls_d3": np.array([class_id], dtype=np.float32),
            "target_img_d3": np.array([class_id], dtype=np.float32),
        }
        metrics.update_stats(stat)

    metrics.process()
    results = metrics.results_dict

    assert results["metrics/precision(3D)"] == pytest.approx(1.0)
    assert results["metrics/recall(3D)"] == pytest.approx(1.0)
    # Ultralytics' 101-point interpolation integrates perfect finite predictions to 0.995.
    assert results["metrics/mAP50(3D)"] == pytest.approx(0.995)
    assert results["metrics/mAP50-95(3D)"] == pytest.approx(0.995)
    assert metrics.fitness == pytest.approx(results["metrics/mAP50(3D)"])
    assert len(metrics.class_result(0)) == 8
    summary = metrics.summary()
    assert len(summary) == 3
    assert summary[0]["3D-Instances"] == 1
    assert summary[0]["3D-P"] == pytest.approx(1.0)
    assert summary[0]["3D-R"] == pytest.approx(1.0)
    assert summary[0]["3D-mAP50"] == pytest.approx(0.995)
    assert summary[0]["3D-mAP50-95"] == pytest.approx(0.995)


def test_detect3d_generic_process_keeps_invalid_prediction_as_false_positive():
    metrics = Detection3DMetrics(names={0: "Car"})
    metrics.update_stats(
        {
            "tp": np.zeros((1, 10), dtype=bool),
            "conf": np.array([0.9]),
            "pred_cls": np.array([0.0]),
            "target_cls": np.array([0.0]),
            "target_img": np.array([0.0]),
            "im_name": "invalid.png",
            "tp_d3": np.zeros((1, 10), dtype=bool),
            "target_cls_d3": np.array([0.0]),
            "target_img_d3": np.array([0.0]),
        }
    )
    metrics.process()

    assert metrics.d3.p[0] == pytest.approx(0.0)
    assert metrics.d3.r[0] == pytest.approx(0.0)
    assert metrics.d3.map50 == pytest.approx(0.0)


def test_detect3d_generic_process_handles_no_valid_3d_targets():
    metrics = Detection3DMetrics(names={0: "Car"})
    metrics.update_stats(
        {
            "tp": np.zeros((0, 10), dtype=bool),
            "conf": np.zeros(0),
            "pred_cls": np.zeros(0),
            "target_cls": np.array([0.0]),
            "target_img": np.array([0.0]),
            "im_name": "empty3d.png",
            "tp_d3": np.zeros((0, 10), dtype=bool),
            "target_cls_d3": np.zeros(0),
            "target_img_d3": np.zeros(0),
        }
    )

    metrics.process()

    assert metrics.d3.mean_results() == [0.0, 0.0, 0.0, 0.0]
    assert metrics.fitness == 0.0
    assert metrics.class_result(0)[4:] == [0.0, 0.0, 0.0, 0.0]


def test_detect3d_class_result_maps_sparse_3d_classes_by_class_id():
    metrics = Detection3DMetrics(names={0: "Car", 1: "Van", 2: "Truck"})
    metrics.box.p = metrics.box.r = np.ones(3)
    metrics.box.all_ap = np.ones((3, 10))
    metrics.box.ap_class_index = np.array([0, 1, 2])
    metrics.d3.p = np.array([0.2, 0.8])
    metrics.d3.r = np.array([0.3, 0.9])
    metrics.d3.all_ap = np.vstack((np.full(10, 0.4), np.full(10, 0.7)))
    metrics.d3.ap_class_index = np.array([0, 2])

    assert metrics.class_result(1)[4:] == [0.0, 0.0, 0.0, 0.0]
    assert metrics.class_result(2)[4:] == pytest.approx([0.8, 0.9, 0.7, 0.7])


def test_3d_metric_matching_is_one_to_one():
    validator = object.__new__(Detection3DValidator)
    validator.d3_err = []
    validator._decode_d3 = lambda extra, image_info: torch.zeros((len(extra), 7))
    pred = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "extra": torch.zeros((1, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.5, 0.5, 10.5, 10.5]]),
        "cls": torch.tensor([0.0, 0.0]),
        "d3": torch.ones((2, 7)),
    }
    validator._update_d3_stats(pred, batch)
    assert len(validator.d3_err) == 1
    assert validator.d3_err[0].shape[0] == 1


def test_generic_3d_matching_uses_exact_iou_without_a_2d_gate():
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    gt_box = torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]])
    validator._decode_d3 = lambda extra, image_info: gt_box.expand(len(extra), -1)
    preds = {
        "bboxes": torch.tensor([[100.0, 100.0, 110.0, 110.0]]),  # no 2D overlap with the target
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
        "extra": torch.zeros((1, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "d3": gt_box,
        "d3_valid": torch.tensor([False]),  # training-policy masks must not affect generic evaluation
    }

    stats = validator._process_batch(preds, batch)

    assert stats["target_cls_d3"].tolist() == [0.0]
    assert stats["tp_d3"].shape == (1, 10)
    assert stats["tp_d3"].all()


def test_generic_3d_matching_keeps_invalid_prediction_as_false_positive():
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    validator._decode_d3 = lambda extra, image_info: torch.tensor(
        [[float("nan"), 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]]
    ).expand(len(extra), -1)
    preds = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
        "extra": torch.zeros((1, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "d3": torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]]),
    }

    stats = validator._process_batch(preds, batch)

    assert stats["target_cls_d3"].tolist() == [0.0]
    assert not stats["tp_d3"].any()


@pytest.mark.parametrize(
    "invalid_box",
    (
        [float("nan"), 0.0, 1.5, 1.6, 1.5, 4.0, 0.0],
        [0.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0],
        [-1.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0],
        [20.0, 0.0, 1.5, 0.0, 1.5, 4.0, 0.0],
    ),
)
def test_generic_3d_matching_zeros_every_invalid_prediction_geometry(invalid_box):
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    validator._decode_d3 = lambda extra, image_info: torch.tensor(invalid_box).view(1, 7)
    preds = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
        "extra": torch.zeros((1, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "d3": torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]]),
    }

    stats = validator._process_batch(preds, batch)

    assert stats["tp_d3"].shape == (1, 10)
    assert not stats["tp_d3"].any()


def test_generic_3d_matching_requires_the_correct_class():
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    box = torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]])
    validator._decode_d3 = lambda extra, image_info: box.expand(len(extra), -1)
    preds = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([1.0]),
        "extra": torch.zeros((1, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "d3": box,
    }

    stats = validator._process_batch(preds, batch)

    assert not stats["tp_d3"].any()


def test_generic_3d_matching_counts_duplicate_predictions_once():
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    box = torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]])
    validator._decode_d3 = lambda extra, image_info: box.expand(len(extra), -1)
    preds = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]),
        "conf": torch.tensor([0.9, 0.8]),
        "cls": torch.tensor([0.0, 0.0]),
        "extra": torch.zeros((2, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "d3": box,
    }

    stats = validator._process_batch(preds, batch)

    assert stats["tp_d3"].shape == (2, 10)
    assert stats["tp_d3"].sum(0).tolist() == [1] * 10


def test_generic_3d_matching_handles_no_ground_truth():
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    validator._decode_d3 = lambda extra, image_info: torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]])
    preds = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
        "extra": torch.zeros((1, 8)),
    }
    batch = {
        "bboxes": torch.empty((0, 4)),
        "cls": torch.empty(0),
        "d3": torch.empty((0, 7)),
    }

    stats = validator._process_batch(preds, batch)

    assert stats["target_cls_d3"].size == 0
    assert stats["tp_d3"].shape == (1, 10)
    assert not stats["tp_d3"].any()


def test_generic_3d_matching_handles_no_predictions():
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    preds = {
        "bboxes": torch.empty((0, 4)),
        "conf": torch.empty(0),
        "cls": torch.empty(0),
        "extra": torch.empty((0, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "d3": torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]]),
    }

    stats = validator._process_batch(preds, batch)

    assert stats["target_cls_d3"].tolist() == [0.0]
    assert stats["tp_d3"].shape == (0, 10)


@pytest.mark.parametrize("extra", (None, torch.empty((1, 7)), torch.empty((2, 8))))
def test_generic_3d_matching_rejects_malformed_prediction_geometry(extra):
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    preds = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
    }
    if extra is not None:
        preds["extra"] = extra
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([0.0]),
        "d3": torch.tensor([[20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]]),
    }

    with pytest.raises(ValueError, match="at least 8 geometry values"):
        validator._process_batch(preds, batch)


def test_generic_3d_matching_excludes_only_geometrically_invalid_ground_truth():
    validator = object.__new__(Detection3DValidator)
    validator.iouv = torch.linspace(0.5, 0.95, 10)
    validator.niou = 10
    valid_box = torch.tensor([20.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0])
    validator._decode_d3 = lambda extra, image_info: valid_box.view(1, 7).expand(len(extra), -1)
    preds = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([2.0]),
        "extra": torch.zeros((1, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[20.0, 20.0, 30.0, 30.0], [0.0, 0.0, 10.0, 10.0]]),
        "cls": torch.tensor([1.0, 2.0]),
        "d3": torch.stack((torch.tensor([float("nan"), 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]), valid_box)),
        "d3_valid": torch.tensor([True, False]),
    }

    stats = validator._process_batch(preds, batch)

    assert stats["target_cls_d3"].tolist() == [2.0]
    assert stats["tp_d3"].all()


def test_3d_diagnostic_metrics_exclude_targets_masked_out_of_3d_training():
    validator = object.__new__(Detection3DValidator)
    validator.d3_err = []
    validator._decode_d3 = lambda extra, image_info: torch.tensor([[10.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]] * len(extra))
    pred = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "cls": torch.tensor([0.0, 0.0]),
        "extra": torch.zeros((2, 8)),
    }
    batch = {
        "bboxes": pred["bboxes"].clone(),
        "cls": torch.tensor([0.0, 0.0]),
        "d3": torch.tensor(
            [
                [10.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0],
                [10.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0],
            ]
        ),
        "d3_valid": torch.tensor([False, True]),
    }

    validator._update_d3_stats(pred, batch)

    assert len(validator.d3_err) == 1
    assert validator.d3_err[0].shape == (1, 7)
    assert torch.isfinite(validator.d3_err[0]).all()


def test_extended_3d_diagnostic_recall_denominator_keeps_gt_without_predictions():
    validator = object.__new__(Detection3DValidator)
    validator.d3_err = []
    validator.d3_diagnostics = []
    validator.d3_gt_depths = []
    validator.extended_d3_diagnostics = True
    pred = {
        "bboxes": torch.empty((0, 4)),
        "cls": torch.empty(0),
        "extra": torch.empty((0, 8)),
    }
    batch = {
        "bboxes": torch.tensor([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]),
        "cls": torch.tensor([0.0, 0.0]),
        "d3": torch.tensor([[10.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0], [50.0, 0.0, 1.5, 1.6, 1.5, 4.0, 0.0]]),
    }

    validator._update_d3_stats(pred, batch)

    assert validator.d3_err == [] and validator.d3_diagnostics == []
    torch.testing.assert_close(validator.d3_gt_depths[0], torch.tensor([10.0, 50.0]))


def test_calibration_and_3d_label_validation_are_strict(tmp_path):
    bad_calib = tmp_path / "bad.txt"
    bad_calib.write_text("P2: 1 2 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="12 values"):
        parse_calib_p2(str(bad_calib))

    image_path = tmp_path / "image.jpg"
    label_path = tmp_path / "label.txt"
    cv2.imwrite(str(image_path), np.zeros((32, 64, 3), dtype=np.uint8))
    # cls cx cy w h depth x y w3d h3d l3d ry -- invalid because depth is negative.
    label_path.write_text("0 0.5 0.5 0.2 0.2 -1 0 1 2 1.5 4 0\n", encoding="utf-8")
    result = verify_image_label((str(image_path), str(label_path), "", False, 1, 0, 0, False, True))
    assert result[0] is None and result[8] == 1
    assert "depth must be positive" in result[9]


def _make_detect3d_cache_hash_dataset(tmp_path):
    image = tmp_path / "images" / "000000.png"
    label = tmp_path / "labels" / "000000.txt"
    calib = tmp_path / "calib" / "000000.txt"
    for path in (image, label, calib):
        path.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"small-image-placeholder")
    label.write_text("label-content-a\n", encoding="utf-8")
    calib.write_text("calib-content-a\n", encoding="utf-8")

    dataset = object.__new__(YOLODataset)
    dataset.use_3d = True
    dataset.im_files = [str(image)]
    dataset.label_files = [str(label)]
    dataset.data = {"calib_dir": str(calib.parent), "names": {0: "Car"}}
    dataset.allow_default_calib = False
    dataset.min_3d_depth = 2.0
    dataset.max_3d_depth = 65.0
    dataset.min_3d_box_height = 25.0
    dataset.max_3d_center_offset = 0.5
    dataset.single_cls = False
    return dataset, label, calib


def test_detect3d_cache_hash_tracks_same_size_label_and_calibration_edits(tmp_path):
    dataset, label, calib = _make_detect3d_cache_hash_dataset(tmp_path)
    original = dataset.get_cache_hash()

    label.write_text("label-content-b\n", encoding="utf-8")  # same byte count
    label_changed = dataset.get_cache_hash()
    calib.write_text("calib-content-b\n", encoding="utf-8")  # same byte count
    calib_changed = dataset.get_cache_hash()

    assert label_changed != original
    assert calib_changed != label_changed


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("allow_default_calib", True),
        ("min_3d_depth", 3.0),
        ("max_3d_depth", 70.0),
        ("min_3d_box_height", 30.0),
        ("max_3d_center_offset", 0.75),
    ],
)
def test_detect3d_cache_hash_tracks_3d_validity_config(tmp_path, attribute, value):
    dataset, _, _ = _make_detect3d_cache_hash_dataset(tmp_path)
    original = dataset.get_cache_hash()

    setattr(dataset, attribute, value)

    assert dataset.get_cache_hash() != original
