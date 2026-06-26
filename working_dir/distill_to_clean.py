"""Unwrap a distillation-trained checkpoint into a clean RTDETR/YOLODETR-compatible .pt.

The ``detr_decoder`` branch trains with a ``DistillModel`` wrapper holding both teacher and student
submodules. The wrapper class lives in ``ultralytics.nn.distill_model`` — a module that was removed
from the ``detr_decoder_clean`` branch, so the source pickle can only be unpickled HERE (in the old
repo where the class still exists).

Run this in the OLD repo. The student submodule is already a regular ``RTDETRDetectionModel`` whose
``yaml_file`` attribute carries the matching architecture YAML path — no YAML arg needed, no fresh
model rebuild needed. We just unwrap, drop the teacher, and re-save. Because nothing is shimmed
here, the saved checkpoint has no synthetic function references in its state, so the new repo's
``yolodetr_convert.py`` can load it without any module-level workarounds.

Example:
    >>> /Users/esat/git/ultralytics/.venv/bin/python distill_to_clean.py \\
    ...     --src ~/deim_dinov3_sp_student.pt
    ...     # writes ~/deim_dinov3_sp_student_clean.pt next to the source

    Then in the new repo:
    >>> python yolodetr_convert.py --src ~/deim_dinov3_sp_student_clean.pt \\
    ...     --yaml yolo27x-detr.yaml --out ~/yolo27x-detr.pt --verify
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import torch

from ultralytics.nn.tasks import RTDETRDetectionModel

warnings.filterwarnings("ignore")


def unwrap_student(wrapper):
    """Return the student submodule from a DistillModel wrapper, or the wrapper itself if not wrapped."""
    student = getattr(wrapper, "student", None)
    if student is None and hasattr(wrapper, "_modules"):
        student = wrapper._modules.get("student")
    if student is None:
        print("  no '.student' submodule found; treating top-level model as the student")
        return wrapper
    print("  extracted '.student' submodule from distillation wrapper")
    return student


def resolve_yaml(student, override: Path | None) -> str:
    """Pick the architecture YAML from CLI override, ``model.yaml_file``, or ``model.yaml['yaml_file']``."""
    if override is not None:
        return str(override)
    yf = getattr(student, "yaml_file", None)
    if yf:
        return str(yf)
    y = getattr(student, "yaml", None)
    if isinstance(y, dict) and y.get("yaml_file"):
        return str(y["yaml_file"])
    raise ValueError("no YAML found in checkpoint and --yaml not provided")


def main() -> None:
    """Entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True, help="distillation-trained source .pt")
    p.add_argument("--out", type=Path, default=None, help="output path (default: <src_stem>_clean.pt)")
    p.add_argument("--yaml", type=Path, default=None, help="override the YAML path embedded in the checkpoint")
    args = p.parse_args()
    out = args.out or args.src.with_name(f"{args.src.stem}_clean.pt")

    print(f"loading distill source: {args.src}")
    ck = torch.load(args.src, map_location="cpu", weights_only=False)
    student = unwrap_student(ck["model"])
    src_dtype = next(student.parameters()).dtype
    nc = int(getattr(student, "nc", 0)) or None
    yaml_path = resolve_yaml(student, args.yaml)
    print(f"  student: class={type(student).__name__}, params={sum(p.numel() for p in student.parameters()):,}, dtype={src_dtype}, nc={nc}")
    print(f"  yaml:    {yaml_path}")

    # Rebuild from YAML in the current (old) repo so the saved object's class graph contains only
    # ultralytics.nn.tasks classes - no distill_model leakage through subclass MRO.
    fresh = RTDETRDetectionModel(yaml_path, ch=3, nc=nc, verbose=False)
    fresh.load_state_dict(student.state_dict(), strict=True)
    for attr in ("names", "nc", "stride", "args"):
        if hasattr(student, attr):
            setattr(fresh, attr, getattr(student, attr))
    fresh.yaml_file = Path(yaml_path).name
    fresh.to(src_dtype).eval()

    new_ck = {**ck, "model": fresh}
    new_ck.pop("teacher", None)
    torch.save(new_ck, out)
    print(f"saved: {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
