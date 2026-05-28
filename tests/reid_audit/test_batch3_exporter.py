"""Batch 3: exporter integration for ReID.

What would have failed before the fix:
  * The ReID head's class-level `export=False` was never flipped → ONNX traced the eval-mode
    2-tuple `(emb, feat_bn)` against a single-name output_names=['output0'] list.
  * `yolo export task=reid nms=True` slipped past `ClassificationModel` guard and produced an
    obscure NMSModel tracing crash instead of a clean assert.
  * Unsupported formats (imx, edgetpu, axelera) were not blocked for ReID.
  * `TASK2CALIBRATIONDATA` had no 'reid' key, so anything iterating TASKS would KeyError.
  * `TASK_CUSTOM_KEYS` machinery aside, the streamlit T_ORD picker omitted '-reid' → ValueError.
"""
from __future__ import annotations

import inspect
import pytest


# ---------- Registry-level wiring (no model load needed) ------------------------------------


def test_task2calibrationdata_has_reid():
    from ultralytics.cfg import TASK2CALIBRATIONDATA, TASKS

    assert "reid" in TASKS
    assert "reid" in TASK2CALIBRATIONDATA, "TASK2CALIBRATIONDATA missing 'reid' — exporter int8 paths will KeyError"
    assert TASK2CALIBRATIONDATA["reid"]  # non-empty


def test_streamlit_t_ord_has_reid_suffix():
    """T_ORD must include '-reid' so the model picker doesn't ValueError on yolo26n-reid.pt."""
    from ultralytics.solutions.streamlit_inference import Inference

    src = inspect.getsource(Inference.configure)
    assert "'-reid'" in src or '"-reid"' in src, "T_ORD must include '-reid'"


# ---------- Exporter import wiring -----------------------------------------------------------


def test_exporter_imports_reid_classes():
    """Exporter must import ReID head class AND ReidModel — otherwise the m.export flip and
    nms-guard assert (added in Batch 3) reference undefined names."""
    import ultralytics.engine.exporter as exp_mod

    assert hasattr(exp_mod, "ReID"), "exporter.py must import ReID head class"
    assert hasattr(exp_mod, "ReidModel"), "exporter.py must import ReidModel"


def test_exporter_loop_flips_reid_head_export():
    """The export module-loop's isinstance check for `m.export = True` must include ReID
    alongside Classify — locks in the specific assignment, not just the presence of the name."""
    import ultralytics.engine.exporter as exp_mod

    src = inspect.getsource(exp_mod.Exporter.__call__)
    # Find the m.export = True line and confirm the immediately-preceding isinstance check covers ReID
    idx = src.find("m.export = True")
    assert idx != -1, "Exporter loop is missing `m.export = True`"
    preceding = src[max(0, idx - 200) : idx]
    assert "(Classify, ReID)" in preceding or "(ReID, Classify)" in preceding, (
        "isinstance check before `m.export = True` must include ReID (was Classify-only before fix)"
    )


def test_exporter_nms_guard_covers_reid_model():
    """The nms=True assertion must reject ReidModel — was ClassificationModel-only before fix."""
    import ultralytics.engine.exporter as exp_mod

    src = inspect.getsource(exp_mod.Exporter.__call__)
    # Direct check for the assertion message — survives line wrapping / arg reorder.
    assert "(ClassificationModel, ReidModel)" in src or "ReidModel, ClassificationModel" in src, (
        "nms=True assertion must reject (ClassificationModel, ReidModel) — was ClassificationModel-only before fix"
    )


def test_exporter_reid_format_whitelist_present():
    """Exporter must raise a clear error for ReID + unsupported format (axelera/edgetpu/imx)."""
    import ultralytics.engine.exporter as exp_mod

    src = inspect.getsource(exp_mod.Exporter.__call__)
    assert 'model.task == "reid"' in src
    # Sanity: the supported list must include the formats ReID can actually export to
    for fmt in ("torchscript", "onnx", "openvino", "engine"):
        assert fmt in src


# ---------- ONNX naming and dynamic axes ----------------------------------------------------


def test_export_onnx_uses_embeddings_output_name():
    """`output_names` for ReID export must be ['embeddings'], not ['output0'].

    `export_onnx` is wrapped by `@try_export`, which does NOT use functools.wraps, so
    inspect.getsource on the attribute returns the wrapper. Read the file directly.
    """
    from pathlib import Path
    import ultralytics.engine.exporter as exp_mod

    full = Path(exp_mod.__file__).read_text()
    # Slice the file from `def export_onnx(` to the next top-level `def ` (4-space indent)
    start = full.find("def export_onnx(")
    assert start != -1
    rest = full[start:]
    end = rest.find("\n    @try_export") + len(rest[:rest.find("\n    @try_export")])
    fn_src = rest[: end if end > 0 else min(len(rest), 4000)]
    assert "embeddings" in fn_src, "export_onnx must name the ReID output 'embeddings'"
    assert 'self.model.task == "reid"' in fn_src


# ---------- ReID head behaviour at export time ----------------------------------------------


def test_reid_head_export_flag_collapses_2tuple_to_single_output():
    """When ReID.export=True the head must return a single Tensor (not the (emb, feat_bn) tuple)."""
    import torch
    from ultralytics.nn.modules.head import ReID

    head = ReID(c1=64, c2=10, embed_dim=128)
    head.eval()
    x = torch.zeros(2, 64, 8, 8)
    # Default: export=False → 2-tuple in eval
    out_eval = head(x)
    assert isinstance(out_eval, tuple) and len(out_eval) == 2
    # After Exporter flips it: single tensor
    head.export = True
    out_export = head(x)
    assert isinstance(out_export, torch.Tensor)
    assert out_export.shape == (2, 128)


# ---------- benchmarks.py mirror -------------------------------------------------------------


def test_int8_calibration_uses_torch_transforms_attr():
    """Regression: ReidDataset has `.torch_transforms`, not `.transforms`. The exporter must
    gate the LetterBox-new_shape patch on attribute presence so int8 reid export doesn't
    crash with AttributeError on `dataset.transforms.transforms[0]`."""
    from pathlib import Path
    import ultralytics.engine.exporter as exp_mod

    full = Path(exp_mod.__file__).read_text()
    fn_src = full[full.find("def get_int8_calibration_dataloader(") : full.find("def export_torchscript(")]
    assert "getattr(dataset, \"transforms\"" in fn_src or "getattr(dataset, 'transforms'" in fn_src, (
        "exporter must use getattr(dataset, 'transforms', None) to avoid crashing on ReidDataset"
    )


def test_int8_calibration_routes_reid_via_build_yolo_dataset():
    """ReID INT8 calibration must dispatch via build_yolo_dataset (centralised routing),
    NOT instantiate ReidDataset directly."""
    from pathlib import Path
    import ultralytics.engine.exporter as exp_mod

    full = Path(exp_mod.__file__).read_text()
    fn_src = full[full.find("def get_int8_calibration_dataloader(") : full.find("def export_torchscript(")]
    assert "build_yolo_dataset" in fn_src, "INT8 reid calibration must call build_yolo_dataset"
    # And must NOT instantiate ReidDataset directly
    assert "ReidDataset(root=" not in fn_src, (
        "INT8 reid calibration must delegate to build_yolo_dataset, not instantiate ReidDataset directly"
    )


def test_int8_calibration_defaults_to_gallery_split_for_reid():
    """ReID INT8 calibration must default to the gallery split (not the small query split that
    args.split='val' would resolve to). DEFAULT_CFG sets args.split='val'; the exporter must
    treat 'val' as 'unset' for reid and prefer gallery for distributional coverage."""
    from pathlib import Path
    import ultralytics.engine.exporter as exp_mod

    full = Path(exp_mod.__file__).read_text()
    fn_src = full[full.find("def get_int8_calibration_dataloader(") : full.find("def export_torchscript(")]
    assert '"gallery"' in fn_src, "ReID INT8 calibration must default to 'gallery' split"
    # The `args.split or 'gallery'` form is wrong (args.split is always 'val' from DEFAULT_CFG)
    assert 'self.args.split or "gallery"' not in fn_src and "self.args.split or 'gallery'" not in fn_src, (
        "Naive `args.split or 'gallery'` fallback is dead code — args.split defaults to 'val'"
    )


def test_task2calibrationdata_reid_uses_mini():
    """TASK2CALIBRATIONDATA['reid'] must point at a calibration-sized mini dataset
    (reid8.yaml), not the full Market-1501 training set."""
    from ultralytics.cfg import TASK2CALIBRATIONDATA

    assert TASK2CALIBRATIONDATA["reid"] == "reid8.yaml", (
        "Expected reid8.yaml (calibration mini). Market-1501.yaml is the full train set and "
        "would make INT8 reid export ~100x slower than other tasks."
    )


def test_reid8_yaml_exists_and_well_formed():
    """reid8.yaml must exist, load cleanly, and declare gallery split for INT8 calibration."""
    from pathlib import Path
    import yaml

    repo = Path(__file__).resolve().parents[2]
    p = repo / "ultralytics/cfg/datasets/reid8.yaml"
    assert p.exists(), "reid8.yaml is the canonical reid calibration mini"
    d = yaml.safe_load(p.read_text())
    for key in ("path", "train", "val", "gallery", "nc", "filename_re"):
        assert key in d, f"reid8.yaml missing {key}"
    assert d["nc"] == 8


def test_benchmarks_has_reid_format_whitelist():
    """benchmarks.py must mirror the exporter's ReID format whitelist."""
    import ultralytics.utils.benchmarks as bm

    src = inspect.getsource(bm.benchmark)
    assert 'model.task == "reid"' in src, "benchmarks.benchmark() must gate reid by format"


# ---------- Live nms=True assertion ----------------------------------------------------------


@pytest.mark.skipif("not __import__('torch').cuda.is_available()", reason="needs torch")
def test_export_nms_true_rejected_for_reid_model():
    """A live ReidModel + nms=True must raise AssertionError with a ReID-specific message."""
    import torch
    from pathlib import Path
    from ultralytics.engine.exporter import Exporter
    from ultralytics.nn.tasks import ReidModel
    from ultralytics import settings  # noqa: F401  side-effect import

    # Find a model yaml
    yaml = Path(__file__).resolve().parents[2] / "ultralytics/cfg/models/26/yolo26-reid.yaml"
    m = ReidModel(str(yaml), nc=10, ch=3, verbose=False)
    m.task = "reid"
    m.eval()
    m.pt_path = "yolo26n-reid.pt"  # for filename derivation

    exporter = Exporter()
    exporter.args.nms = True
    exporter.args.format = "onnx"
    exporter.args.imgsz = 64
    with pytest.raises(AssertionError, match=r"(?i)nms=True.*reid"):
        exporter(model=m)
