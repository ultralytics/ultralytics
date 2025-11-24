from pathlib import Path

from ultralytics.analytics.batcheval import BatchEvalResult, ModelSpec, _collect_summary_rows, _write_csv, resolve_models


def test_resolve_models_with_pt(tmp_path: Path) -> None:
    m = tmp_path / "m.pt"
    m.write_bytes(b"dummy")  # fake weights
    specs = resolve_models([str(m)])
    assert len(specs) == 1
    spec = specs[0]
    assert isinstance(spec, ModelSpec)
    assert spec.weights_path == m
    assert spec.name == "m"


def test_resolve_models_with_run_dir(tmp_path: Path) -> None:
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


