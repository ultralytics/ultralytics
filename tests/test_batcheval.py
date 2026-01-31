"""Tests for ultralytics.analytics.batcheval module."""

from pathlib import Path

import pytest

from ultralytics.analytics.batcheval import (
    BatchEvalResult,
    ModelSpec,
    _collect_summary_rows,
    _run_conf_sweep,
    _write_csv,
    resolve_models,
)


def test_resolve_models_with_pt(tmp_path: Path) -> None:
    """Test that resolve_models correctly handles a direct .pt file path."""
    m = tmp_path / "m.pt"
    m.write_bytes(b"dummy")  # fake weights
    specs = resolve_models([str(m)])
    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, ModelSpec)
    assert spec.weights_path == m
    assert spec.name == "m"


def test_resolve_models_with_run_dir(tmp_path: Path) -> None:
    """Test that resolve_models correctly handles a run directory with weights/best.pt."""
    run = tmp_path / "train99"
    wdir = run / "weights"
    wdir.mkdir(parents=True)
    best = wdir / "best.pt"
    best.write_bytes(b"dummy")

    specs = resolve_models([str(run)])
    assert len(specs) == 1
    spec = specs[0]
    assert spec.weights_path == best
    assert spec.name == "train99"


def test_collect_summary_and_write_csv(tmp_path: Path) -> None:
    """Test that _collect_summary_rows and _write_csv produce valid CSV output."""
    results = [
        BatchEvalResult(model_name="a", metrics={"metrics/mAP50": 0.5, "metrics/mAP50-95": 0.4}),
        BatchEvalResult(model_name="b", metrics={"metrics/mAP50": 0.6}),
    ]
    rows = _collect_summary_rows(results)
    out_path = tmp_path / "summary.csv"
    _write_csv(out_path, rows)

    data = out_path.read_text(encoding="utf-8").strip().splitlines()
    # Header plus two rows.
    assert len(data) == 3
    assert "model_name" in data[0]
    assert "metrics/mAP50" in data[0]


def test_run_conf_sweep_strips_incoming_conf_and_forwards_other_kwargs(monkeypatch, tmp_path: Path) -> None:
    """Test that _run_conf_sweep strips user-provided conf and forwards other kwargs."""
    calls = []

    class DummyModel:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def val(self, **kwargs):
            calls.append(kwargs)
            return {"metrics/mAP50": 0.5}

    # Ensure batcheval uses our dummy model instead of creating real YOLO instances.
    monkeypatch.setitem(_run_conf_sweep.__globals__, "YOLO", DummyModel)

    weights = tmp_path / "m.pt"
    weights.write_bytes(b"dummy")
    spec = ModelSpec(name="m", weights_path=weights)

    _run_conf_sweep(
        specs=[spec],
        data="data.yaml",
        split="val",
        conf_min=0.1,
        conf_max=0.3,
        conf_step=0.1,
        save_root=tmp_path,
        conf=0.9,  # incoming user conf that must not clash with the sweep conf
        imgsz=640,
    )

    # Three sweep steps: 0.1, 0.2, 0.3
    assert len(calls) == 3
    confs = sorted(kwargs["conf"] for kwargs in calls)
    assert confs == pytest.approx([0.1, 0.2, 0.3], rel=1e-6)
    for kwargs in calls:
        assert kwargs["imgsz"] == 640


def test_handle_yolo_batcheval_forwards_extra_overrides(monkeypatch) -> None:
    """Test that handle_yolo_batcheval forwards extra kwargs to YOLO().val()."""
    import ultralytics.analytics as analytics_mod
    import ultralytics.cfg as cfg_mod

    captured: dict = {}

    def fake_run_cli(args):
        captured["args"] = args

    # Route the batcheval CLI call to our fake implementation.
    monkeypatch.setattr(analytics_mod, "run_cli", fake_run_cli)

    cfg_mod.handle_yolo_batcheval(
        [
            "models=model.pt",
            "data=coco8.yaml",
            "imgsz=640",
            "conf=0.25",
            "save_json=True",
        ]
    )

    forwarded = captured["args"]
    assert forwarded["models"] == "model.pt"
    assert forwarded["data"] == "coco8.yaml"
    # Extra overrides should be forwarded unchanged (after smart_value parsing).
    assert forwarded["imgsz"] == 640
    assert forwarded["conf"] == 0.25
    assert forwarded["save_json"] is True


def test_handle_yolo_batcheval_requires_models_and_data(monkeypatch) -> None:
    """Test that handle_yolo_batcheval raises SyntaxError when required args are missing."""
    import ultralytics.analytics as analytics_mod
    import ultralytics.cfg as cfg_mod

    # Prevent any real batcheval execution.
    monkeypatch.setattr(analytics_mod, "run_cli", lambda args: None)

    # Missing data argument should raise a SyntaxError.
    with pytest.raises(SyntaxError):
        cfg_mod.handle_yolo_batcheval(["models=model.pt"])
