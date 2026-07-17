"""Scrub vestigial Python attributes from pickled Ultralytics checkpoints.

Some `nn.Module` subclasses in the DETR/DEIM family bound module-level utility functions to `self` (e.g.
``self.ms_deformable_attn_core = multi_scale_deformable_attn_pytorch``). Those bindings get serialized into
``__dict__`` and, once pickled, tie every checkpoint to that specific symbol path — renaming or moving the function
later breaks the checkpoint. This script strips such attributes from an already-saved checkpoint so re-saved weights
no longer carry them.

Backs up the original as ``<src>.bak`` (unless it already exists), rewrites the checkpoint in place, and reloads to
verify zero residual attributes. Also scrubs the EMA sub-model if present.

Examples:
    Default: scrub ``ms_deformable_attn_core`` from a single checkpoint (in place, with ``.bak`` backup)
    >>> python scrub_ckpt_attrs.py --src ~/yolo27x-detr.pt

    Dry-run: report offenders without touching the file
    >>> python scrub_ckpt_attrs.py --src ~/yolo27x-detr.pt --dry-run

    Scrub multiple attributes across every ``.pt`` in a directory
    >>> python scrub_ckpt_attrs.py --src ~/checkpoints --attrs ms_deformable_attn_core stale_flag

    Verify inference after scrubbing (loads scrubbed weights and runs a forward on ultralytics/assets/bus.jpg)
    >>> python scrub_ckpt_attrs.py --src ~/yolo27x-detr.pt --verify
"""

from __future__ import annotations

import argparse
import shutil
import warnings
from pathlib import Path

import torch

warnings.filterwarnings("ignore")

DEFAULT_ATTRS = ("ms_deformable_attn_core",)


def collect_offenders(model: torch.nn.Module, attrs: tuple[str, ...]) -> list[tuple[str, str, str]]:
    """Return ``(attr, module_name, class_name)`` for every submodule holding one of the target attributes.

    Args:
        model (torch.nn.Module): Model to scan.
        attrs (tuple[str, ...]): Attribute names to look for in each submodule's ``__dict__``.

    Returns:
        (list[tuple[str, str, str]]): Triples describing each hit; empty if none.
    """
    hits = []
    for name, mod in model.named_modules():
        for a in attrs:
            if a in mod.__dict__:
                hits.append((a, name, type(mod).__name__))
    return hits


def scrub(model: torch.nn.Module, attrs: tuple[str, ...]) -> int:
    """Delete the target attributes from every submodule that carries them, returning the count removed."""
    n = 0
    for _, mod in model.named_modules():
        for a in attrs:
            if a in mod.__dict__:
                del mod.__dict__[a]
                n += 1
    return n


def scrub_ckpt(src: Path, attrs: tuple[str, ...], dry_run: bool, no_backup: bool) -> tuple[int, int]:
    """Load one checkpoint, scrub, and re-save. Returns ``(scrub_model, scrub_ema)`` counts."""
    ckpt = torch.load(src, map_location="cpu", weights_only=False)

    model_hits = collect_offenders(ckpt["model"], attrs) if ckpt.get("model") is not None else []
    ema_hits = collect_offenders(ckpt["ema"], attrs) if ckpt.get("ema") is not None else []
    total_hits = len(model_hits) + len(ema_hits)

    print(f"  offenders: model={len(model_hits)}, ema={len(ema_hits)}, total={total_hits}")
    if model_hits and dry_run:
        for a, name, cls in model_hits[:5]:
            print(f"    [model] {name}  ({cls}.{a})")
        if len(model_hits) > 5:
            print(f"    ... {len(model_hits) - 5} more")

    if dry_run or total_hits == 0:
        return len(model_hits), len(ema_hits)

    bak = src.with_suffix(src.suffix + ".bak")
    if not no_backup:
        if bak.exists():
            print(f"  backup already exists (keeping older): {bak.name}")
        else:
            shutil.copy2(src, bak)
            print(f"  backup: {bak.name}")

    m = scrub(ckpt["model"], attrs) if ckpt.get("model") is not None else 0
    e = scrub(ckpt["ema"], attrs) if ckpt.get("ema") is not None else 0
    torch.save(ckpt, src)
    print(f"  stripped model={m}, ema={e}  |  saved: {src.name}")

    # Reload verify
    reloaded = torch.load(src, map_location="cpu", weights_only=False)
    residual = 0
    if reloaded.get("model") is not None:
        residual += len(collect_offenders(reloaded["model"], attrs))
    if reloaded.get("ema") is not None:
        residual += len(collect_offenders(reloaded["ema"], attrs))
    print(f"  verify residual: {residual}  {'OK' if residual == 0 else 'FAIL'}")
    return m, e


def verify_forward(src: Path) -> None:
    """Reload the scrubbed checkpoint, run a forward on bus.jpg, and print the top detections."""
    import numpy as np
    from PIL import Image

    repo_root = Path(__file__).resolve().parents[2] / "ultralytics2"
    bus = repo_root / "ultralytics/assets/bus.jpg"

    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    m = ckpt["model"].float().eval()
    img = Image.open(bus).convert("RGB").resize((640, 640))
    arr = np.asarray(img, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        out = m(x)
    y = out[0] if isinstance(out, tuple) else out
    top = y[0, y[0, :, 4].argsort(descending=True)][:6]
    names = getattr(m, "names", {}) or {}
    print(f"  verify forward on {bus.name}: output={tuple(y.shape)}")
    for d in top:
        cls = int(d[5])
        bb = [round(b, 2) for b in d[:4].tolist()]
        print(f"    cls={cls:3d} ({names.get(cls, '?')}) conf={d[4].item():.3f} bbox={bb}")


def iter_checkpoints(src: Path) -> list[Path]:
    """Expand a file or directory path into a list of ``.pt`` files (non-recursive)."""
    if src.is_file():
        return [src]
    if src.is_dir():
        return sorted(p for p in src.glob("*.pt") if p.is_file())
    raise FileNotFoundError(f"no such file or directory: {src}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--src", type=Path, required=True, help="checkpoint .pt path or directory of .pt files")
    p.add_argument(
        "--attrs",
        nargs="+",
        default=list(DEFAULT_ATTRS),
        help=f"attribute names to strip from submodule __dict__ (default: {' '.join(DEFAULT_ATTRS)})",
    )
    p.add_argument("--dry-run", action="store_true", help="report offenders without modifying files")
    p.add_argument("--no-backup", action="store_true", help="skip writing a <src>.bak copy (not recommended)")
    p.add_argument("--verify", action="store_true", help="after scrub, reload and run a bus.jpg forward pass")
    return p.parse_args()


def main() -> None:
    """Entry point."""
    args = parse_args()
    attrs = tuple(args.attrs)
    files = iter_checkpoints(args.src)
    print(f"scanning {len(files)} file(s) for attrs: {attrs}  (dry_run={args.dry_run})")

    total_m, total_e = 0, 0
    for f in files:
        print(f"\n[{f.name}]")
        m, e = scrub_ckpt(f, attrs, args.dry_run, args.no_backup)
        total_m += m
        total_e += e
        if args.verify and not args.dry_run and (m + e) > 0:
            verify_forward(f)

    print(f"\nsummary: model_attrs_removed={total_m}, ema_attrs_removed={total_e}, files={len(files)}")


if __name__ == "__main__":
    main()
