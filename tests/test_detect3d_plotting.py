"""Tests for Detect3D-specific training result plots."""

from __future__ import annotations

import csv
from unittest.mock import patch

import polars as pl

from ultralytics.models.yolo.detect3d.train import Detection3DTrainer
from ultralytics.utils.plotting import _detect3d_result_columns, plot_detect3d_results


def _write_results(path) -> None:
    """Write a compact Detect3D results table."""
    fieldnames = [
        "epoch",
        "train/box_loss",
        "train/depth_loss",
        "val/depth_loss",
        "train/keypoint3d_loss",
        "val/keypoint3d_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
        "metrics/precision(3D)",
        "metrics/recall(3D)",
        "metrics/mAP50(3D)",
        "metrics/mAP50-95(3D)",
        "3d/dist_MAE",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in range(1, 5):
            row = {
                "epoch": epoch,
                "train/box_loss": 1.0 / epoch,
                "train/depth_loss": 2.0 / epoch,
                "val/depth_loss": 2.5 / epoch,
                "train/keypoint3d_loss": 0.5 / epoch,
                "val/keypoint3d_loss": 0.0,
                "metrics/precision(B)": 0.4 + epoch * 0.01,
                "metrics/recall(B)": 0.3 + epoch * 0.01,
                "metrics/mAP50(B)": 0.35 + epoch * 0.01,
                "metrics/mAP50-95(B)": 0.2 + epoch * 0.01,
                "metrics/precision(3D)": 0.2 + epoch * 0.01,
                "metrics/recall(3D)": 0.18 + epoch * 0.01,
                "metrics/mAP50(3D)": 0.15 + epoch * 0.01,
                "metrics/mAP50-95(3D)": 0.08 + epoch * 0.01,
                "3d/dist_MAE": 3.0 / epoch,
            }
            writer.writerow(row)


def test_detect3d_plot_columns_are_3d_focused_and_skip_inactive_loss(tmp_path):
    """The main figure should exclude 2D losses and misleading all-zero auxiliary losses."""
    results = tmp_path / "results.csv"
    _write_results(results)
    columns_3d, columns_2d = _detect3d_result_columns(pl.read_csv(results))

    assert columns_3d[:5] == [
        "metrics/mAP50(3D)",
        "metrics/mAP50-95(3D)",
        "metrics/precision(3D)",
        "metrics/recall(3D)",
        "3d/dist_MAE",
    ]
    assert "train/depth_loss" in columns_3d and "val/depth_loss" in columns_3d
    assert "train/keypoint3d_loss" in columns_3d
    assert "val/keypoint3d_loss" not in columns_3d
    assert "train/box_loss" not in columns_3d
    assert "3d/fitness" not in columns_3d
    assert columns_2d == [
        "train/box_loss",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]


def test_detect3d_plotting_renders_both_figures(tmp_path):
    """Detect3D plotting should render the primary 3D and secondary 2D figures."""
    results = tmp_path / "results.csv"
    _write_results(results)
    plotted = []

    plot_detect3d_results(results, on_plot=plotted.append)

    assert (tmp_path / "results.png").is_file()
    assert (tmp_path / "results_2d.png").is_file()
    assert plotted == [tmp_path / "results.png", tmp_path / "results_2d.png"]


def test_detect3d_trainer_uses_task_specific_result_plotter(tmp_path):
    """Detect3D should not change or call the generic result plotter used by other tasks."""
    trainer = object.__new__(Detection3DTrainer)
    trainer.csv = tmp_path / "results.csv"
    trainer.on_plot = lambda *_: None

    with patch("ultralytics.models.yolo.detect3d.train.plot_detect3d_results") as plotter:
        trainer.plot_metrics()

    plotter.assert_called_once_with(file=trainer.csv, on_plot=trainer.on_plot)
