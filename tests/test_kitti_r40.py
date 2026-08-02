# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ultralytics.models.yolo.detect3d.val import (
    Detection3DMetrics,
    Detection3DValidator,
    _summarize_d3_diagnostics,
)
from ultralytics.utils.kitti_eval import (
    _clean_data,
    _compute_statistics,
    bev_box_overlap,
    build_kitti_predictions,
    d3_box_overlap,
    evaluate_kitti_metric,
    evaluate_kitti_r40,
    format_kitti_r40,
    image_box_overlap,
    paired_d3_box_overlap,
    parse_kitti_label,
    plot_kitti_r40,
    write_kitti_predictions,
)


def make_annotation(
    names=("Car",),
    bboxes=((0.0, 0.0, 100.0, 100.0),),
    dimensions=((1.5, 1.6, 4.0),),
    locations=((0.0, 1.5, 20.0),),
    rotations=(0.0,),
    alphas=(0.0,),
    scores=(1.0,),
):
    """Build a compact KITTI annotation for deterministic metric tests."""
    annotation = build_kitti_predictions(
        list(names),
        np.asarray(bboxes),
        np.asarray(scores),
        np.asarray(alphas),
        np.asarray(dimensions),
        np.asarray(locations),
        np.asarray(rotations),
    )
    annotation.truncated[:] = 0.0
    annotation.occluded[:] = 0
    return annotation


def test_kitti_prediction_file_round_trip(tmp_path):
    prediction = make_annotation(scores=(0.875,))
    path = tmp_path / "000001.txt"
    write_kitti_predictions(prediction, path)
    loaded = parse_kitti_label(path)

    assert loaded.name.tolist() == ["Car"]
    np.testing.assert_allclose(loaded.bbox, prediction.bbox)
    np.testing.assert_allclose(loaded.dimensions, prediction.dimensions)
    np.testing.assert_allclose(loaded.location, prediction.location)
    np.testing.assert_allclose(loaded.score, prediction.score)


def test_validator_collects_native_kitti_prediction_columns(tmp_path):
    image_dir = tmp_path / "val" / "images"
    label_dir = tmp_path / "val" / "label_2"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    write_kitti_predictions(make_annotation(scores=(0.0,)), label_dir / "000001.txt")

    validator = object.__new__(Detection3DValidator)
    validator.kitti_label_dir = "label_2"
    validator.kitti_records = []
    validator.names = {0: "Car"}
    validator._decode_d3 = lambda extra, image_info: torch.tensor(
        [[20.0, 2.0, 1.5, 1.6, 1.5, 4.0, 0.2]], dtype=extra.dtype
    )
    alpha = 0.1
    prediction = {
        "bboxes": torch.tensor([[10.0, 20.0, 50.0, 80.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
        "extra": torch.tensor([[30.0, 40.0, 20.0, math.sin(alpha), math.cos(alpha), 1.5, 1.6, 4.0]]),
    }
    batch = {
        "im_file": str(image_dir / "000001.png"),
        "imgsz": (100, 100),
        "ori_shape": (100, 100),
        "ratio_pad": ((1.0, 1.0), (0.0, 0.0)),
    }
    validator._collect_kitti_record(prediction, batch)
    _, _, collected = validator.kitti_records[0]

    np.testing.assert_allclose(collected.bbox, [[10.0, 20.0, 50.0, 80.0]])
    np.testing.assert_allclose(collected.dimensions, [[1.5, 1.6, 4.0]])
    np.testing.assert_allclose(collected.location, [[2.0, 1.5, 20.0]])
    np.testing.assert_allclose(collected.alpha, [alpha], atol=1e-6)


def test_validator_scales_native_kitti_boxes_with_independent_xy_gains(tmp_path):
    image_dir = tmp_path / "val" / "images"
    label_dir = tmp_path / "val" / "label_2"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    write_kitti_predictions(make_annotation(scores=(0.0,)), label_dir / "000001.txt")

    validator = object.__new__(Detection3DValidator)
    validator.kitti_label_dir = "label_2"
    validator.kitti_records = []
    validator.names = {0: "Car"}
    validator._decode_d3 = lambda extra, image_info: torch.tensor(
        [[20.0, 2.0, 1.5, 1.6, 1.5, 4.0, 0.2]], dtype=extra.dtype
    )
    native_box = torch.tensor([[100.0, 50.0, 1000.0, 300.0]])
    gain_y, gain_x = 387.0 / 375.0, 1280.0 / 1242.0
    pad_x, pad_y = 0.0, 6.0
    model_box = native_box.clone()
    model_box[:, [0, 2]] = model_box[:, [0, 2]] * gain_x + pad_x
    model_box[:, [1, 3]] = model_box[:, [1, 3]] * gain_y + pad_y
    alpha = 0.1
    prediction = {
        "bboxes": model_box,
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
        "extra": torch.tensor([[300.0, 200.0, 20.0, math.sin(alpha), math.cos(alpha), 1.5, 1.6, 4.0]]),
    }
    batch = {
        "im_file": str(image_dir / "000001.png"),
        "imgsz": (400, 1280),
        "ori_shape": (375, 1242),
        "ratio_pad": ((gain_y, gain_x), (pad_x, pad_y)),
    }

    validator._collect_kitti_record(prediction, batch)
    np.testing.assert_allclose(validator.kitti_records[0][2].bbox, native_box.numpy(), atol=1e-4)


def test_validator_caches_kitti_ground_truth_across_training_epochs(tmp_path):
    image_dir = tmp_path / "val" / "images"
    label_dir = tmp_path / "val" / "label_2"
    image_dir.mkdir(parents=True)
    label_dir.mkdir()
    (label_dir / "000001.txt").write_text("", encoding="utf-8")

    validator = object.__new__(Detection3DValidator)
    validator.kitti_label_dir = "label_2"
    validator.kitti_records = []
    validator.kitti_gt_cache = {}
    empty_prediction = {
        "bboxes": torch.empty((0, 4)),
        "conf": torch.empty(0),
        "cls": torch.empty(0),
        "extra": torch.empty((0, 8)),
    }
    batch = {"im_file": str(image_dir / "000001.png")}

    with patch(
        "ultralytics.models.yolo.detect3d.val.parse_kitti_label",
        return_value=make_annotation(scores=(0.0,)),
    ) as parser:
        validator._collect_kitti_record(empty_prediction, batch)
        validator.kitti_records.clear()  # init_metrics clears records between epochs but intentionally keeps the cache.
        validator._collect_kitti_record(empty_prediction, batch)

    parser.assert_called_once()


@pytest.mark.parametrize(("requested", "expected_dir"), [("off", None), ("fast", "label_2"), ("full", "label_2")])
def test_kitti_eval_modes_are_explicit(requested, expected_dir):
    validator = object.__new__(Detection3DValidator)
    validator.training = False
    validator.data = {"kitti_label_dir": "label_2"}
    validator.names = {0: "Car"}
    validator.metrics = SimpleNamespace(d3_results={})
    validator.args = SimpleNamespace(kitti_eval=requested)

    with patch("ultralytics.models.yolo.detect.val.DetectionValidator.init_metrics"):
        validator.init_metrics(SimpleNamespace())

    assert validator.kitti_eval_mode == requested
    assert validator.kitti_label_dir == expected_dir


def test_kitti_eval_missing_argument_defaults_to_off():
    """An incomplete custom args namespace must not implicitly read label_2."""
    validator = object.__new__(Detection3DValidator)
    validator.training = False
    validator.data = {"kitti_label_dir": "label_2"}
    validator.names = {0: "Car"}
    validator.metrics = SimpleNamespace(d3_results={})
    validator.args = SimpleNamespace()

    with patch("ultralytics.models.yolo.detect.val.DetectionValidator.init_metrics"):
        validator.init_metrics(SimpleNamespace())

    assert validator.kitti_eval_mode == "off"
    assert validator.kitti_label_dir is None


def test_kitti_eval_rejects_implicit_or_unknown_modes():
    validator = object.__new__(Detection3DValidator)
    validator.training = False
    validator.data = {"kitti_label_dir": "label_2"}
    validator.names = {0: "Car"}
    validator.metrics = SimpleNamespace(d3_results={})
    validator.args = SimpleNamespace(kitti_eval="auto")

    with patch("ultralytics.models.yolo.detect.val.DetectionValidator.init_metrics"):
        with pytest.raises(ValueError, match="kitti_eval"):
            validator.init_metrics(SimpleNamespace())


def test_fast_kitti_metric_is_auxiliary_and_does_not_drive_generic_fitness(tmp_path):
    validator = object.__new__(Detection3DValidator)
    validator.d3_err = []
    validator.d3_diagnostics = []
    validator.d3_gt_depths = []
    validator.kitti_records = [("000001", make_annotation(scores=(0.0,)), make_annotation(scores=(0.9,)))]
    validator.kitti_eval_mode = "fast"
    validator.kitti_classes = ("Car",)
    validator.kitti_plot_paths = []
    validator.metrics = Detection3DMetrics(names={0: "Car"})
    validator.metrics.d3.p = np.array([0.4])
    validator.metrics.d3.r = np.array([0.5])
    validator.metrics.d3.all_ap = np.full((1, 10), 0.125)
    validator.metrics.d3.ap_class_index = np.array([0])
    validator.args = SimpleNamespace(plots=False)
    validator.save_dir = tmp_path

    with (
        patch("ultralytics.models.yolo.detect.val.DetectionValidator.get_stats"),
        patch(
            "ultralytics.models.yolo.detect3d.val.evaluate_kitti_metric",
            return_value=12.5,
        ) as fast_eval,
        patch("ultralytics.models.yolo.detect3d.val.evaluate_kitti_r40") as full_eval,
    ):
        results = validator.get_stats()

    fast_eval.assert_called_once()
    full_eval.assert_not_called()
    assert results["kitti/Car_AP3D_R40_moderate"] == 12.5
    assert results["kitti/AP3D_R40_moderate"] == 12.5
    assert results["fitness"] == pytest.approx(0.125)


def test_rotated_bev_and_3d_iou_are_not_axis_aligned_approximations():
    reference = make_annotation()
    same = make_annotation()
    perpendicular = make_annotation(rotations=(math.pi / 2,))
    distant = make_annotation(locations=((20.0, 1.5, 20.0),))

    assert bev_box_overlap(reference, same)[0, 0] == pytest.approx(1.0, abs=3e-6)
    assert d3_box_overlap(reference, same)[0, 0] == pytest.approx(1.0, abs=3e-6)
    assert bev_box_overlap(reference, perpendicular)[0, 0] == pytest.approx(0.25, abs=1e-6)
    assert d3_box_overlap(reference, perpendicular)[0, 0] == pytest.approx(0.25, abs=1e-6)
    assert bev_box_overlap(reference, distant)[0, 0] == 0.0
    assert d3_box_overlap(reference, distant)[0, 0] == 0.0


def test_aligned_3d_iou_is_exact_and_rejects_mismatched_lengths():
    boxes = make_annotation(
        names=("Car", "Car"),
        bboxes=((0.0, 0.0, 100.0, 100.0),) * 2,
        dimensions=((1.5, 1.6, 4.0),) * 2,
        locations=((0.0, 1.5, 20.0),) * 2,
        rotations=(0.0, 0.0),
        alphas=(0.0, 0.0),
        scores=(1.0, 1.0),
    )
    queries = make_annotation(
        names=("Car", "Car"),
        bboxes=((0.0, 0.0, 100.0, 100.0),) * 2,
        dimensions=((1.5, 1.6, 4.0),) * 2,
        locations=((0.0, 1.5, 20.0),) * 2,
        rotations=(0.0, math.pi / 2),
        alphas=(0.0, 0.0),
        scores=(1.0, 1.0),
    )

    aligned = paired_d3_box_overlap(boxes, queries)
    np.testing.assert_allclose(aligned, [1.0, 0.25], atol=3e-6)
    np.testing.assert_allclose(aligned, np.diag(d3_box_overlap(boxes, queries)), atol=3e-6)
    queries.rotation_y[1] = np.nan
    assert paired_d3_box_overlap(boxes, queries)[1] == 0.0
    with pytest.raises(ValueError, match="equal lengths"):
        paired_d3_box_overlap(boxes, make_annotation())


def test_3d_diagnostic_summary_uses_all_valid_gt_as_recall_denominator():
    # columns: GT depth, seven absolute errors, exact IoU3D, raw sigmoid(q3d)
    matched = torch.tensor(
        [
            [10.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 5.0, 0.8, 0.9],
            [30.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.4, 10.0, 0.6, 0.5],
            [50.0, 9.0, 0.3, 0.3, 0.3, 0.3, 0.6, 20.0, 0.4, 0.1],
        ]
    )
    results = _summarize_d3_diagnostics(matched, torch.tensor([10.0, 30.0, 50.0, 70.0]))

    assert results["3d/diagnostic/match_recall"] == pytest.approx(0.75)
    assert results["3d/diagnostic/iou3d_recall_0.5"] == pytest.approx(0.5)
    assert results["3d/diagnostic/iou3d_recall_0.7"] == pytest.approx(0.25)
    assert results["3d/diagnostic/dist_P50"] == pytest.approx(2.0)
    assert results["3d/diagnostic/dist_P90"] == pytest.approx(7.6)
    assert results["3d/diagnostic/q3d_iou3d_pearson"] == pytest.approx(1.0)
    assert results["3d/diagnostic/q3d_iou3d_spearman"] == pytest.approx(1.0)
    assert results["3d/range_40m_plus/gt_count"] == 2
    assert results["3d/range_40m_plus/matched_count"] == 1
    assert results["3d/range_40m_plus/iou3d_recall_0.5"] == 0.0


def test_3d_diagnostic_summary_handles_empty_matches_and_constant_quality():
    empty = _summarize_d3_diagnostics(torch.empty((0, 10)), torch.tensor([10.0, 50.0]))
    assert empty["3d/diagnostic/match_recall"] == 0.0
    assert empty["3d/diagnostic/iou3d_recall_0.7"] == 0.0

    matched = torch.tensor(
        [
            [10.0, 1.0, 0.1, 0.1, 0.1, 0.1, 0.2, 5.0, 0.8, 0.5],
            [30.0, 2.0, 0.2, 0.2, 0.2, 0.2, 0.4, 10.0, 0.2, 0.5],
        ]
    )
    constant = _summarize_d3_diagnostics(matched, torch.tensor([10.0, 30.0]))
    assert constant["3d/diagnostic/q3d_iou3d_pearson"] == 0.0
    assert constant["3d/diagnostic/q3d_iou3d_spearman"] == 0.0


def test_kitti_difficulty_and_neighbor_class_rules():
    short_car = make_annotation(bboxes=((0.0, 0.0, 100.0, 30.0),))
    detection = make_annotation(bboxes=((0.0, 0.0, 100.0, 30.0),))
    valid_easy, ignored_easy, ignored_dt_easy, _ = _clean_data(short_car, detection, "Car", 0)
    valid_moderate, ignored_moderate, _, _ = _clean_data(short_car, detection, "Car", 1)

    assert valid_easy == 0
    assert ignored_easy.tolist() == [1]
    assert ignored_dt_easy.tolist() == [1]
    assert valid_moderate == 1
    assert ignored_moderate.tolist() == [0]

    van = make_annotation(names=("Van",))
    valid_car, ignored_van, _, _ = _clean_data(van, make_annotation(), "Car", 0)
    assert valid_car == 0
    assert ignored_van.tolist() == [1]


def test_dontcare_suppresses_unmatched_2d_detection():
    ground_truth = make_annotation()
    detections = make_annotation(
        names=("Car", "Car"),
        bboxes=((0.0, 0.0, 100.0, 100.0), (200.0, 0.0, 300.0, 100.0)),
        dimensions=((1.5, 1.6, 4.0), (1.5, 1.6, 4.0)),
        locations=((0.0, 1.5, 20.0), (20.0, 1.5, 20.0)),
        rotations=(0.0, 0.0),
        alphas=(0.0, 0.0),
        scores=(0.9, 0.8),
    )
    overlap = image_box_overlap(detections.bbox, ground_truth.bbox)
    tp, fp, fn, _, _ = _compute_statistics(
        overlap,
        ground_truth,
        detections,
        np.asarray([0]),
        np.asarray([0, 0]),
        np.asarray([[200.0, 0.0, 300.0, 100.0]]),
        "bbox",
        0.7,
        compute_fp=True,
    )
    assert (tp, fp, fn) == (1, 0, 0)


def test_perfect_predictions_reach_100_r40_and_aos_uses_alpha(tmp_path):
    ground_truth, perfect, reversed_orientation = [], [], []
    for index in range(41):
        score = 1.0 - index * 0.001
        ground_truth.append(make_annotation(scores=(0.0,)))
        perfect.append(make_annotation(scores=(score,)))
        reversed_orientation.append(make_annotation(alphas=(math.pi,), scores=(score,)))

    perfect_results, perfect_curves = evaluate_kitti_r40(ground_truth, perfect, classes=["Car"], return_curves=True)
    reversed_results = evaluate_kitti_r40(ground_truth, reversed_orientation, classes=["Car"])

    assert perfect_results["kitti/Car_AP3D_R40_moderate"] == pytest.approx(100.0)
    assert perfect_results["kitti/Car_APBEV_R40_moderate"] == pytest.approx(100.0)
    assert perfect_results["kitti/Car_AOS_R40_moderate"] == pytest.approx(100.0)
    assert reversed_results["kitti/Car_AP3D_R40_moderate"] == pytest.approx(100.0)
    assert reversed_results["kitti/Car_AOS_R40_moderate"] == pytest.approx(0.0, abs=1e-10)
    curve = perfect_curves["kitti/Car_AP3D_R40_moderate"]
    assert curve.shape == (41,)
    assert curve[1:].mean() * 100.0 == pytest.approx(perfect_results["kitti/Car_AP3D_R40_moderate"])

    plot_paths = plot_kitti_r40(perfect_results, perfect_curves, ["Car"], tmp_path)
    assert {path.name for path in plot_paths} == {
        "KITTI_R40_summary.png",
        "KITTI_R40_Car_curves.png",
    }
    assert all(path.stat().st_size > 10_000 for path in plot_paths)


def test_single_kitti_metric_perfect_predictions_reach_100_r40():
    ground_truth, detections = [], []
    for index in range(41):
        ground_truth.append(make_annotation(scores=(0.0,)))
        detections.append(make_annotation(scores=(1.0 - index * 0.001,)))

    ap, curve = evaluate_kitti_metric(ground_truth, detections, return_curve=True)

    assert ap == 100.0
    assert curve.shape == (41,)
    np.testing.assert_array_equal(curve, np.ones(41))


def test_single_kitti_metric_counts_interleaved_duplicate_as_false_positive():
    ground_truth = [make_annotation(scores=(0.0,)) for _ in range(41)]
    detections = [
        make_annotation(
            names=("Car", "Car"),
            bboxes=((0.0, 0.0, 100.0, 100.0),) * 2,
            dimensions=((1.5, 1.6, 4.0),) * 2,
            locations=((0.0, 1.5, 20.0),) * 2,
            rotations=(0.0, 0.0),
            alphas=(0.0, 0.0),
            scores=(1.0, 0.995),
        )
    ]
    detections.extend(make_annotation(scores=(0.98 - index * 0.01,)) for index in range(40))

    fast_ap = evaluate_kitti_metric(ground_truth, detections)
    full_ap = evaluate_kitti_r40(ground_truth, detections, classes=["Car"])["kitti/Car_AP3D_R40_moderate"]

    assert 0.0 < fast_ap < 100.0
    assert fast_ap == full_ap


def test_single_kitti_metric_empty_predictions_return_zero():
    ground_truth = [make_annotation(scores=(0.0,)) for _ in range(41)]
    empty_prediction = make_annotation(
        names=(),
        bboxes=(),
        dimensions=(),
        locations=(),
        rotations=(),
        alphas=(),
        scores=(),
    )
    detections = [empty_prediction for _ in ground_truth]

    ap, curve = evaluate_kitti_metric(ground_truth, detections, return_curve=True)

    assert ap == 0.0
    np.testing.assert_array_equal(curve, np.zeros(41))


@pytest.mark.parametrize(
    ("metric", "result_name"),
    [("3d", "AP3D"), ("bev", "APBEV"), ("aos", "AOS")],
)
def test_single_kitti_metric_strictly_matches_full_r40(metric, result_name):
    ground_truth, detections = [], []
    for index in range(41):
        ground_truth.append(make_annotation(scores=(0.0,)))
        location = (0.0, 1.5, 20.0) if index % 5 else (0.8, 1.5, 20.0)
        alpha = 0.0 if index % 7 else math.pi
        detections.append(make_annotation(locations=(location,), alphas=(alpha,), scores=(1.0 - index * 0.001,)))

    fast_ap, fast_curve = evaluate_kitti_metric(
        ground_truth,
        detections,
        class_name="Car",
        difficulty="moderate",
        metric=metric,
        return_curve=True,
    )
    full_results, full_curves = evaluate_kitti_r40(ground_truth, detections, classes=["Car"], return_curves=True)
    key = f"kitti/Car_{result_name}_R40_moderate"

    assert fast_ap == full_results[key]
    np.testing.assert_array_equal(fast_curve, full_curves[key])


def test_kitti_alpha_sentinel_omits_aos(tmp_path):
    ground_truth, detections = [], []
    for index in range(41):
        ground_truth.append(make_annotation(scores=(0.0,)))
        detections.append(make_annotation(alphas=(-10.0,), scores=(1.0 - index * 0.001,)))

    results, curves = evaluate_kitti_r40(ground_truth, detections, classes=["Car"], return_curves=True)

    assert not any("AOS" in key for key in results)
    assert not any("AOS" in key for key in curves)
    assert "AOS" not in format_kitti_r40(results, ["Car"])
    plot_paths = plot_kitti_r40(results, curves, ["Car"], tmp_path)
    assert {path.name for path in plot_paths} == {
        "KITTI_R40_summary.png",
        "KITTI_R40_Car_curves.png",
    }
