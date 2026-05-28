"""End-to-end smoke test on Market-1501 — tiny config, single GPU.

Exercises the full train → val → predict → export pipeline through the modified code paths:
  * build_yolo_dataset reid branch
  * ReidTrainer.build_dataset delegation
  * gallery extraction via self.preprocess
  * Results.embeddings on predict()
  * ONNX export with output name 'embeddings'

Skips automatically if no GPU or no Market-1501.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("REID_E2E_SKIP") == "1", reason="REID_E2E_SKIP=1 — skipping e2e smoke"
)


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory, market1501_root, has_gpu):
    """Run one mini-epoch of ReID training. Returns the run directory."""
    if not has_gpu:
        pytest.skip("e2e smoke requires a GPU")
    from ultralytics import YOLO

    work = tmp_path_factory.mktemp("reid_e2e")
    # Symlink Market-1501 into the standard datasets layout
    datasets_dir = work / "datasets"
    datasets_dir.mkdir()
    (datasets_dir / "Market-1501-v15.09.15").symlink_to(market1501_root)

    os.environ.setdefault("YOLO_DATASETS_DIR", str(datasets_dir))
    # Resolve repo-local YAMLs explicitly to avoid pulling from network
    repo = Path(__file__).resolve().parents[2]
    data_yaml = repo / "ultralytics/cfg/datasets/Market-1501.yaml"
    model_yaml = repo / "ultralytics/cfg/models/26/yolo26-reid.yaml"

    model = YOLO(str(model_yaml))
    results = model.train(
        data=str(data_yaml),
        epochs=1,
        imgsz=64,
        batch=8,
        device=0,
        workers=2,
        project=str(work),
        name="run",
        exist_ok=True,
        verbose=False,
        cache=False,
        amp=False,
    )
    run_dir = Path(results.save_dir)
    return run_dir, model


def test_e2e_train_produces_weights(trained_run):
    run_dir, _ = trained_run
    weights = run_dir / "weights" / "last.pt"
    assert weights.exists(), f"trainer did not produce {weights}"


def test_e2e_predict_returns_embeddings(trained_run, market1501_root):
    """After training, predicting on a Market-1501 sample must return Results.embeddings (not .probs)."""
    run_dir, _ = trained_run
    from ultralytics import YOLO

    weights = run_dir / "weights" / "last.pt"
    m = YOLO(str(weights))
    sample = next((market1501_root / "query").glob("*.jpg"))
    results = m.predict(source=str(sample), imgsz=64, device=0, verbose=False)
    r = results[0]
    assert r.embeddings is not None, "ReID predict() must populate Results.embeddings"
    assert r.probs is None, "ReID predict() must NOT populate Results.probs"
    assert r.embeddings.dim > 0


def test_e2e_export_onnx_has_embeddings_output(trained_run, tmp_path_factory):
    """Export must produce a valid ONNX with single output named 'embeddings'."""
    pytest.importorskip("onnx")
    import onnx

    run_dir, _ = trained_run
    from ultralytics import YOLO

    weights = run_dir / "weights" / "last.pt"
    out_dir = tmp_path_factory.mktemp("onnx_out")
    # Copy the weight so export writes alongside it (default behaviour)
    target = out_dir / "last.pt"
    shutil.copy(weights, target)
    m = YOLO(str(target))
    onnx_path = m.export(format="onnx", imgsz=64, device=0, simplify=False)
    onnx_model = onnx.load(str(onnx_path))
    output_names = [o.name for o in onnx_model.graph.output]
    assert output_names == ["embeddings"], f"expected ['embeddings'], got {output_names}"


def test_e2e_export_rejects_nms_for_reid(trained_run):
    """Export with nms=True on a ReidModel must raise AssertionError (was a tracing crash before)."""
    run_dir, _ = trained_run
    from ultralytics import YOLO

    weights = run_dir / "weights" / "last.pt"
    m = YOLO(str(weights))
    with pytest.raises((AssertionError, ValueError), match=r"(?i)nms.*reid|reid.*nms|classification or reid"):
        m.export(format="onnx", imgsz=64, device=0, nms=True)


def test_e2e_export_rejects_imx_for_reid(trained_run):
    """Export with format=imx on ReidModel must raise — either via my new ReID format
    whitelist, or via IMX's pre-existing task whitelist (detect/pose/classify/segment).
    Either path is acceptable; just confirm IMX export does NOT succeed for ReID."""
    run_dir, _ = trained_run
    from ultralytics import YOLO

    weights = run_dir / "weights" / "last.pt"
    m = YOLO(str(weights))
    with pytest.raises((ValueError, AssertionError), match=r"(?i)reid|IMX export only"):
        m.export(format="imx", imgsz=64, device=0)
