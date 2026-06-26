"""Extract a clean YOLODETR student checkpoint from a distillation-trained .pt.

Distillation training on the ``detr_decoder`` branch wrapped the student model in a class living
under ``ultralytics.nn.distill_model`` — a module that no longer exists. Loading such a checkpoint
on the current branch fails with ``ModuleNotFoundError`` before any class lookup can run.

This script installs a ``sys.modules`` shim that synthesizes ``ultralytics.nn.distill_model`` and
redirects any class lookup inside it to ``YOLODETRDetectionModel``. The pickled object restores into
a YOLODETR-compatible container; the underlying student weights are written back into a regular
checkpoint dict that the standard ``yolodetr_convert.py`` can consume.

Examples:
    >>> python distill_extract.py --src ~/deim_dinov3_sp_student.pt
    >>> python yolodetr_convert.py --src ~/deim_dinov3_sp_student_extracted.pt --yaml yolo27x-detr.yaml \\
    ...     --out ~/yolo27x-detr.pt --verify
"""

from __future__ import annotations

import argparse
import sys
import types
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")


def _stub(*a, **kw):
    """Module-level placeholder for removed attributes referenced by the source pickle.

    Must live at module scope so torch.save can re-pickle objects whose state references it; a closure would fail with
    "Can't get local object" when serializing the extracted student.
    """
    raise NotImplementedError("unpickle-time stub")


def install_shims() -> None:
    """Register a synthetic ``ultralytics.nn.distill_model`` mapping any class to YOLODETRDetectionModel.

    Also reuses the attribute-level stubs from ``yolodetr_convert.install_pickle_shims`` so this script is
    self-contained and can be run before the convert script without depending on its side effects.
    """
    import ultralytics.nn.modules.utils as utm
    from ultralytics.nn.tasks import YOLODETRDetectionModel

    for name in ("deformable_attention_core_func_v2", "MSDeformAttnFunction", "dab_sine_embedding"):
        if not hasattr(utm, name):
            setattr(utm, name, _stub)

    if "ultralytics.nn.distill_model" not in sys.modules:
        stub_mod = types.ModuleType("ultralytics.nn.distill_model")

        def _resolve(name: str):
            # Skip dunder probes (inspect/importlib look up __file__, __path__, __spec__, etc.) so they
            # raise AttributeError instead of returning a class that breaks downstream introspection.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            return YOLODETRDetectionModel

        stub_mod.__getattr__ = _resolve
        sys.modules["ultralytics.nn.distill_model"] = stub_mod


def extract_student(src: Path, out: Path) -> None:
    """Load a distillation-wrapped checkpoint and rewrite it with the bare student model."""
    install_shims()
    ck = torch.load(src, map_location="cpu", weights_only=False)
    wrapper = ck["model"]

    student = getattr(wrapper, "student", None) or wrapper._modules.get("student") if hasattr(wrapper, "_modules") else None
    if student is None:
        student = wrapper
        print(f"  no '.student' submodule found; using top-level model object directly")
    else:
        print(f"  extracted '.student' submodule")

    print(f"  student class={type(student).__name__}, params={sum(p.numel() for p in student.parameters()):,}")

    new_ck = {**ck, "model": student}
    new_ck.pop("teacher", None)
    torch.save(new_ck, out)
    print(f"saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True, help="distillation-trained source .pt")
    p.add_argument("--out", type=Path, default=None, help="output path (default: <src_stem>_extracted.pt)")
    args = p.parse_args()
    out = args.out or args.src.with_name(f"{args.src.stem}_extracted.pt")
    print(f"loading: {args.src}")
    extract_student(args.src, out)


if __name__ == "__main__":
    main()
