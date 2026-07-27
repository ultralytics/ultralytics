# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shared T4 latency harness, timed the same way as profile_depth.py and Ultralytics ProfileModels.

Timing comes from the predictor's own ``speed["inference"]``, which brackets ``AutoBackend.forward`` with
``torch.cuda.synchronize`` and ``perf_counter`` and reaches TensorRT through ``execute_v2``. The previous harness
timed ``execute_async_v3`` with CUDA events on a private stream, which read 1.35% low in absolute terms and, more
seriously, shifted cross-architecture ratios by up to 2.9pp because arms self-heat at different rates.

Kept from the previous harness and absent from both colleague harnesses: balanced paired rounds, a named baseline,
paired per-round deltas and A/B win counts. A single-pass mean cannot separate a real win from thermal ordering.

The formats run back to back with nothing between them, so the slower ones preheat the card ahead of the TensorRT
reading exactly as profile_depth.py does. Run telemetry from a separate process, never inside this loop.
"""

import csv
import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.utils.benchmarks import ProfileModels
from ultralytics.utils.torch_utils import get_gpu_info

WARMUP = 10
RUNS = 100
ROUNDS = 8
TIMER = "predictor-speed"  # rows measured before 2026-07-27 used "cuda-events"

# Ordered format table: name -> (Variant attribute holding the artifact, extra predict kwargs). The order is the
# protocol, not a preference, so formats are never selectable and never reordered.
FORMATS = {
    "pt16": ("weights", {"quantize": 16}),
    "pt32": ("weights", {"quantize": 32}),
    "onnx": ("onnx", {}),
    "trt": ("engine", {}),
}


@dataclass(frozen=True)
class Variant:
    """One benchmarked architecture, its prebuilt artifacts, and the facade every one of them loads through.

    Attributes:
        name (str): Short tag used in the CSV and as the baseline key.
        weights (Path): Trained checkpoint, or a model yaml for an untrained architecture-search candidate.
        onnx (Path): FP32 ONNX export of the same architecture.
        engine (Path): FP16 TensorRT engine of the same architecture.
        model_cls (type): Facade that resolves the task, YOLO for Lane A and RTDETR for the DEIM yamls.
        params_m (float): Fused parameter count in millions.
        gflops (float): Fused GFLOPs at the export resolution.
    """

    name: str
    weights: Path
    onnx: Path
    engine: Path
    model_cls: type
    params_m: float
    gflops: float


def env_info(imgsz, variants):
    """Collect the software, hardware and artifact provenance a latency number is only valid under."""
    return {
        "timer": TIMER,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": get_gpu_info(0),
        "ultralytics": ultralytics.__version__,
        "ultralytics_path": ultralytics.__file__,
        "imgsz": imgsz,
        "batch": 1,
        "warmup": WARMUP,
        "timed_runs": RUNS,
        "rounds": ROUNDS,
        **{m: __import__(m).__version__ for m in ("onnxruntime", "tensorrt", "onnx")},
        "artifacts": {v.name: {a: str(getattr(v, a)) for a in ("weights", "onnx", "engine")} for v in variants},
    }


def build_variant(name, weights, outdir, imgsz=640, device="0", model_cls=YOLO):
    """Export the FP32 ONNX and FP16 engine for one architecture, reusing whatever already exists at that imgsz.

    Args:
        name (str): Short tag, also the artifact filename stem.
        weights (str | Path): Trained checkpoint or model yaml.
        outdir (str | Path): Directory the artifacts are written to.
        imgsz (int): Square input size baked into both exports and into the artifact stem.
        device (str): CUDA device index used for the export.
        model_cls (type): Facade that resolves the task, YOLO for Lane A and RTDETR for the DEIM yamls.

    Returns:
        (Variant): The variant with artifact paths, facade and fused model metrics populated.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    onnx, engine = outdir / f"{name}-{imgsz}.onnx", outdir / f"{name}-{imgsz}.engine"
    for path, fmt, extra in ((onnx, "onnx", {}), (engine, "engine", {"quantize": 16})):
        if not path.exists():  # export mutates the model in place, so each one gets a fresh instance
            exported = model_cls(str(weights)).export(
                format=fmt, imgsz=imgsz, device=device, batch=1, verbose=False, **extra
            )
            shutil.move(str(exported), path)
    model = model_cls(str(weights))
    model.fuse()  # so params and GFLOPs describe the deployed graph
    _, params, _, gflops = model.info(imgsz=imgsz)
    return Variant(name, Path(weights), onnx, engine, model_cls, params / 1e6, gflops)


def profile_model(model, image, **predict_kwargs):
    """Warm up, collect RUNS predictor speed records, sigma clip, and return inference and pipeline means.

    Args:
        model (Model): Loaded model in any backend the predictor supports.
        image (np.ndarray): HWC uint8 input reused for every call.
        **predict_kwargs (Any): Forwarded to the predictor, typically imgsz, device and quantize.

    Returns:
        (dict): Sigma-clipped mean inference ms under `inf`, its standard deviation under `inf_std`, and preprocess plus
            inference plus postprocess under `total`.
    """
    for _ in range(WARMUP):
        model(image, verbose=False, **predict_kwargs)
    pre, inf, post = [], [], []
    for _ in range(RUNS):
        speed = model(image, verbose=False, **predict_kwargs)[0].speed
        pre.append(speed["preprocess"])
        inf.append(speed["inference"])
        post.append(speed["postprocess"])
    clipped = ProfileModels.iterative_sigma_clipping(inf)
    mean = float(clipped.mean())
    return {"inf": mean, "inf_std": float(clipped.std()), "total": float(np.mean(pre)) + mean + float(np.mean(post))}


def profile_variant(variant, image, imgsz, device):
    """Profile every format back to back in FORMATS order, through the variant's own facade, with no gap between."""
    return {
        fmt: profile_model(variant.model_cls(str(getattr(variant, attr))), image, imgsz=imgsz, device=device, **extra)
        for fmt, (attr, extra) in FORMATS.items()
    }


def summarize_rounds(per_round, variants, baseline):
    """Reduce per-round records to one row per variant and format, with paired deltas against the baseline."""
    series = {}
    for record in per_round:  # one list per variant and format, in round order, so zip() pairs them correctly
        for fmt in FORMATS:
            series.setdefault((record["variant"], fmt), []).append(record[f"{fmt}_inf"])
    base_median = {fmt: float(np.median(series[baseline, fmt])) for fmt in FORMATS}

    rows = []
    for variant in variants:
        for fmt in FORMATS:
            own, ref = series[variant.name, fmt], series[baseline, fmt]
            median = float(np.median(own))
            rows.append(
                {
                    "variant": variant.name,
                    "format": fmt,
                    "params_M_fused": round(variant.params_m, 2),
                    "gflops_fused": round(variant.gflops, 1),
                    "median_ms": round(median, 4),
                    "delta_vs_base_pct": round(100 * (median - base_median[fmt]) / base_median[fmt], 2),
                    "ab_wins": "" if variant.name == baseline else f"{sum(a < b for a, b in zip(own, ref))}/{ROUNDS}",
                    "baseline": baseline,
                    "timer": TIMER,
                }
            )
    return rows


def write_csv(path, rows):
    """Write a list of uniform dicts as a CSV with a header taken from the first row."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(variants, baseline, out_csv, imgsz=640, device="0"):
    """Run ROUNDS order-balanced rounds over every variant and write the summary, per-round and provenance files.

    Args:
        variants (list[Variant]): Variants to compare, all artifacts already built.
        baseline (str): Name of the variant every delta is taken against.
        out_csv (str | Path): Summary CSV path. Per-round rows and provenance go beside it.
        imgsz (int): Square input size, which must match the size the artifacts were exported at.
        device (str): CUDA device index passed to the predictor.
    """
    out_csv = Path(out_csv)
    image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    print(f"=== timer {TIMER}, {len(variants)} variants, baseline {baseline}", flush=True)

    per_round = []
    for rnd in range(ROUNDS):
        for variant in variants if rnd % 2 == 0 else variants[::-1]:
            timings = profile_variant(variant, image, imgsz, device)
            per_round.append(
                {
                    "round": rnd,
                    "variant": variant.name,
                    **{f"{f}_{k}": round(v, 4) for f, t in timings.items() for k, v in t.items()},
                }
            )
            summary = " ".join(f"{f}={timings[f]['inf']:7.3f}" for f in FORMATS)
            print(f"  round {rnd + 1}/{ROUNDS} {variant.name:<24} {summary}", flush=True)

    rows = summarize_rounds(per_round, variants, baseline)
    write_csv(out_csv, rows)
    write_csv(out_csv.with_suffix(".rounds.csv"), per_round)
    out_csv.with_suffix(".env.json").write_text(json.dumps(env_info(imgsz, variants), indent=2))

    print(f"\n=== medians vs {baseline}", flush=True)
    for row in rows:
        print(
            f"  {row['variant']:<24}{row['format']:<6}{row['median_ms']:9.4f}"
            f"{row['delta_vs_base_pct']:+8.2f}%  {row['ab_wins']}",
            flush=True,
        )
    print(f"\nwrote {out_csv} plus .rounds.csv and .env.json", flush=True)
