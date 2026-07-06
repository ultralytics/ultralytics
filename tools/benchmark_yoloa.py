#!/usr/bin/env python
"""YOLOA benchmark: params, FLOPs, buffer, ONNX size/speed, and mAP — no-memory vs with-memory.

One linear pass: validate BEFORE fit (bank empty → no-memory / pure detection), fit the memory
bank, then validate again (with-memory / heatmap prior). Finally export both graphs to ONNX for
deploy size + CPU speed.

mAP is read from ``metrics.box.all_ap`` on the anomaly validator's coarse IoU grid
(``linspace(0.10, 0.50, niou)``; see ``anomaly/val.py`` ``_ood_map_metrics``), so it stays
consistent with ``run_yoloa.py`` and the in-training OOD eval. The val pass uses the NMS head
(``--e2e`` opts into the end-to-end / NMS-free head); the ONNX export always uses the deploy
(end-to-end) graph regardless of ``--e2e``.

Guide (run from the repo root, ``PYTHONPATH=.`` so imports resolve to this repo, not the sibling):

    # Full benchmark of a trained checkpoint on the MVTec 'zipper' category (Apple-silicon MPS):
    PYTHONPATH=. python tools/benchmark_yoloa.py \\
        --ckpt ./yoloa-26l.pt --cat zipper \\
        --mvtec-root /Users/louis/workspace/ultra_louis_work/buffer/AnomalyData/MVTEC/MVTec-YOLO \\
        --device mps

    # Expected 'zipper' output (numbers vary a little run-to-run; bank calibration is unseeded):
    #                             params   FLOPs   buffer   mAP10   mAP25   mAP50  mAP10-50      CPU ONNX     size
    #   YOLOA (no memory)         26.18M       -        -  0.3813  0.1442  0.0121    0.1372  195.0±22.5ms  94.8MB
    #   YOLOA (with memory)       26.18M       -  19.53MB  0.6092  0.4273  0.1803    0.3756  231.4±34.8ms  114.3MB
    #   -> the memory bank lifts zipper mAP10-50 from 0.1372 to 0.3756 (+0.24).

    # Other useful invocations:
    PYTHONPATH=. python tools/benchmark_yoloa.py --ckpt ./yoloa-26l.pt --cat zipper \\
        --mvtec-root <root> --device cpu --val-batch 4   # tight-memory / CPU-only box
    PYTHONPATH=. python tools/benchmark_yoloa.py --ckpt yolo26l-anomaly.yaml --no-fit --device cpu
        # YAML (no trained weights): reports params/FLOPs/ONNX size+speed only, skips fit + mAP

Notes:
  - --cat is any MVTec category; it must have <cat>/<cat>_binary.yaml and <cat>/train/good under
    --mvtec-root. The memory bank is cached to runs/temp/bank_cache/<cat>.pt and reused on reruns
    (delete that file to force a rebuild).
  - If MPS OOMs during the bank score, lower --score-chunk (e.g. --score-chunk 4194304) and/or
    --val-batch; these change chunking only, not the results.
"""

import argparse
import gc
import statistics
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from ultralytics.yoloa import YOLOA

NAN = float("nan")


def free_cache(device: str) -> None:
    """Return freed memory to the device pool between heavy passes (avoids MPS OOM)."""
    gc.collect()
    dev = str(device).lower()
    if "mps" in dev and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif "cuda" in dev and torch.cuda.is_available():
        torch.cuda.empty_cache()


def buffer_size_mb(model) -> float:
    """Memory-bank buffer size in MB (0 if no bank)."""
    mb = getattr(model.model, "memory_bank", None)
    if mb is None or not hasattr(mb, "bank") or mb.bank.numel() == 0:
        return 0.0
    return mb.bank.numel() * mb.bank.element_size() / 1024 / 1024


def flops_from_yaml(yaml_path: str, imgsz: int) -> float:
    """GFLOPs from model.info() for a YAML-loaded model (clean graph, no prior fusion)."""
    import io
    from contextlib import redirect_stdout

    m = YOLOA(yaml_path, verbose=False)
    buf = io.StringIO()
    m.model.verbose = True
    with redirect_stdout(buf):
        m.model.info(imgsz=imgsz)
    for line in buf.getvalue().splitlines():
        if "GFLOPs" in line:
            return float(line.split()[-1])
    return NAN


def onnx_speed(onnx_path: str, n_warm: int = 5, n_run: int = 30) -> tuple[float, float]:
    """(mean_ms, std_ms) for ONNX CPU inference (drops the first n_warm timed runs)."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    _, c, h, w = sess.get_inputs()[0].shape
    dummy = np.random.randn(1, c, h, w).astype(np.float32)
    for _ in range(n_warm):
        sess.run(None, {name: dummy})
    times = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        sess.run(None, {name: dummy})
        times.append((time.perf_counter() - t0) * 1000)
    return statistics.mean(times[n_warm:]), statistics.stdev(times[n_warm:])


def extract_map(metrics) -> dict:
    """mAP10/25/50 + mAP10-50 from ``all_ap``, columns located on the validator's IoU grid.

    The anomaly validator evaluates on ``linspace(0.10, 0.50, niou)`` (val.py ``_ood_map_metrics``),
    so map each threshold to its nearest column instead of hardcoding indices — grid-change-proof.
    """
    ap = getattr(metrics.box, "all_ap", None)
    if ap is None or ap.ndim != 2 or ap.shape[1] < 2:
        return {k: NAN for k in ("mAP10", "mAP25", "mAP50", "mAP10-50")}
    iouv = np.linspace(0.10, 0.50, ap.shape[1])
    col = lambda thr: int(np.argmin(np.abs(iouv - thr)))  # nearest column to the IoU threshold
    return {
        "mAP10": float(ap[:, col(0.10)].mean()),
        "mAP25": float(ap[:, col(0.25)].mean()),
        "mAP50": float(ap[:, col(0.50)].mean()),
        "mAP10-50": float(ap.mean()),  # mean over the full .10:.50 grid
    }


def run_val(model, data_yaml: str, device: str, batch: int, use_prior: bool, e2e: bool = False) -> dict:
    """One val pass. ``use_prior=False`` → no-memory (gate off); ``True`` → with-memory (gate on).

    ``e2e=False`` forces the NMS head (matches ``run_yoloa.py``); the model's baked-in default is
    the end-to-end (NMS-free) head, which scores differently, so pass it explicitly for consistency.
    """
    metrics = model.val(
        data=data_yaml,
        iou=0.1,
        single_cls=True,
        device=device,
        verbose=False,
        batch=batch,
        end2end=e2e,
        use_prior=use_prior,
        hm_gate_blend=0.0 if use_prior else 1.0,
    )
    return extract_map(metrics)


def export_profile(
    ckpt: Path, imgsz: int, device: str, fit_dir: str | None = None, cat: str | None = None, cache: str | None = None
) -> dict:
    """Fresh-load, optionally fit, export to ONNX, and profile deploy size + CPU speed.

    A fresh model is loaded per export because ``export()`` fuses layers and mutates state.
    """
    m = YOLOA(str(ckpt), verbose=False)
    buf = 0.0
    if fit_dir:
        m.fit(fit_dir, name=cat, cache=cache, device=device)
        buf = buffer_size_mb(m)
    onnx = m.export(format="onnx", imgsz=imgsz, simplify=True)
    size = Path(onnx).stat().st_size / 1024 / 1024
    mean, std = onnx_speed(onnx)
    Path(onnx).unlink()
    return {"buffer": buf, "size": size, "speed_mean": mean, "speed_std": std}


def print_map(m: dict) -> None:
    """Print a one-line mAP summary for a val pass."""
    print(
        f"  mAP10={m['mAP10']:.4f} mAP25={m['mAP25']:.4f} mAP50={m['mAP50']:.4f} mAP10-50={m['mAP10-50']:.4f}",
        flush=True,
    )


def print_summary(
    params_m: float, flops: float, m_no: dict, m_with: dict, o_no: dict, o_with: dict, device: str
) -> None:
    """Print the final no-memory vs with-memory comparison table."""
    flops_str = "-" if np.isnan(flops) else f"{flops:.1f}G"

    def row(label: str, m: dict, o: dict, buf: str) -> str:
        return (
            f"{label:29s} {params_m:>7.2f}M {flops_str:>8s} {buf:>9s} "
            f"{m.get('mAP10', NAN):>8.4f} {m.get('mAP25', NAN):>8.4f} "
            f"{m.get('mAP50', NAN):>8.4f} {m.get('mAP10-50', NAN):>10.4f} "
            f"{o['speed_mean']:>7.1f}±{o['speed_std']:<4.1f}ms {o['size']:>7.1f}MB"
        )

    print("=" * 112)
    print(
        f"{'':29s} {'params':>8s} {'FLOPs':>8s} {'buffer':>9s} "
        f"{'mAP10':>8s} {'mAP25':>8s} {'mAP50':>8s} {'mAP10-50':>10s} "
        f"{'CPU ONNX':>16s} {'size':>9s}"
    )
    print("-" * 112)
    print(row("YOLOA (no memory)", m_no, o_no, "-"))
    print(row("YOLOA (with memory)", m_with, o_with, f"{o_with['buffer']:.2f}MB"))
    print("=" * 112)
    print("\nNotes:")
    print("  - FLOPs: from model.info() (YAML load only; .pt load shows '-')")
    print("  - buffer: memory-bank register_buffer, stored in checkpoint + ONNX")
    print("  - mAP: *_binary dataset, single_cls=True, IoU grid .10:.50; with-memory gate on (hm_gate_blend=0)")
    print("  - CPU ONNX: onnxruntime CPUExecutionProvider, fp32")
    print(f"  - Device: {device}")


def main():
    ap = argparse.ArgumentParser(description="YOLOA benchmark: no-memory vs with-memory")
    ap.add_argument("--ckpt", required=True, help="path to .pt or .yaml")
    ap.add_argument("--cat", default="bottle", help="MVTec category")
    ap.add_argument("--mvtec-root", required=True, help="MVTec-YOLO root dir")
    ap.add_argument("--device", default="cpu", help="torch device")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--val-batch", type=int, default=8, help="val batch size (lower if OOM)")
    ap.add_argument(
        "--e2e",
        "--end2end",
        action="store_true",
        help="val with the end-to-end (NMS-free) head (default: NMS head, matches run_yoloa.py)",
    )
    ap.add_argument(
        "--score-chunk",
        type=int,
        default=1 << 23,
        help="bank score matmul chunk in elements (~32MB); lower if MPS OOMs. Affects chunking only.",
    )
    ap.add_argument("--cache-dir", default=None, help="bank cache dir (default: runs/temp/bank_cache)")
    ap.add_argument("--no-fit", action="store_true", help="skip fit+mAP (YAML-only: params/FLOPs/speed)")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).resolve()
    root = Path(args.mvtec_root).resolve()
    assert root.is_dir(), f"MVTec root not found: {root}"

    cat = args.cat
    data_yaml = str(root / cat / f"{cat}_binary.yaml")
    fit_dir = root / cat / "train" / "good"
    cache = args.cache_dir or "runs/temp/bank_cache"

    is_yaml = ckpt.suffix in (".yaml", ".yml") or args.no_fit
    can_fit = not is_yaml and fit_dir.is_dir()

    # -- load + params/FLOPs --
    print(f"Loading: {ckpt}", flush=True)
    model = YOLOA(str(ckpt), verbose=False)
    params_m = sum(p.numel() for p in model.model.parameters()) / 1e6
    flops = flops_from_yaml(str(ckpt), args.imgsz) if is_yaml else NAN
    info = f"  params: {params_m:.1f}M" + ("" if np.isnan(flops) else f", FLOPs: {flops:.1f} G")
    print(info + "\n", flush=True)

    # -- val (no-mem before fit) → fit → val (with-mem after fit) --
    metrics_no, metrics_with = {}, {}
    if can_fit:
        print("=== VAL: NO MEMORY (before fit) ===", flush=True)
        free_cache(args.device)
        metrics_no = run_val(model, data_yaml, args.device, args.val_batch, use_prior=False, e2e=args.e2e)
        print_map(metrics_no)

        print("\n=== FIT ===", flush=True)
        free_cache(args.device)
        model.fit(str(fit_dir), name=cat, cache=cache, device=args.device)
        mb = getattr(model.model, "memory_bank", None)
        if mb is not None:
            mb.score_chunk = int(args.score_chunk)  # smaller chunk = tighter peak memory; results unchanged
        print(f"  buffer (memory bank): {buffer_size_mb(model):.2f} MB", flush=True)

        print("\n=== VAL: WITH MEMORY (after fit) ===", flush=True)
        free_cache(args.device)
        metrics_with = run_val(model, data_yaml, args.device, args.val_batch, use_prior=True, e2e=args.e2e)
        print_map(metrics_with)
        print(flush=True)

    # -- ONNX export + CPU speed for both graphs --
    print("=== ONNX: NO MEMORY ===", flush=True)
    onnx_no = export_profile(ckpt, args.imgsz, args.device)
    print(f"  {onnx_no['size']:.1f} MB  |  {onnx_no['speed_mean']:.1f} ± {onnx_no['speed_std']:.1f} ms\n", flush=True)

    print("=== ONNX: WITH MEMORY ===", flush=True)
    if not can_fit:
        print("  (no fit — no train images)", flush=True)
    onnx_with = export_profile(
        ckpt,
        args.imgsz,
        args.device,
        fit_dir=str(fit_dir) if can_fit else None,
        cat=cat,
        cache=cache,
    )
    print(
        f"  {onnx_with['size']:.1f} MB  |  {onnx_with['speed_mean']:.1f} ± {onnx_with['speed_std']:.1f} ms\n",
        flush=True,
    )

    print_summary(params_m, flops, metrics_no, metrics_with, onnx_no, onnx_with, args.device)


if __name__ == "__main__":
    main()
