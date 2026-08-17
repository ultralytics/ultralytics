# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Shared T4 latency harness, timed the same way as profile_depth.py and Ultralytics ProfileModels.

Timing comes from the predictor's own ``speed["inference"]``, which brackets ``AutoBackend.forward`` with
``torch.cuda.synchronize`` and ``perf_counter`` and reaches TensorRT through ``execute_v2``. The previous harness
timed ``execute_async_v3`` with CUDA events on a private stream, which read 1.35% low in absolute terms and, more
seriously, shifted cross-architecture ratios by up to 2.9pp because arms self-heat at different rates.

Kept from the previous harness and absent from both colleague harnesses: balanced paired rounds, a named baseline,
paired per-round deltas and A/B win counts. A single-pass mean cannot separate a real win from thermal ordering.

The formats run back to back with nothing between them, so the GPU rows preheat the card ahead of the TensorRT
reading. Run telemetry from a separate process, never inside this loop.

An untrained yaml is never benchmarked as built. `materialize_yaml` fills its exactly-zero biases first, without
which TensorRT specializes on them and the resulting ratios are wrong by up to 5.3pp in the flattering direction.

Environment:

- ``onnxruntime-gpu``, not ``onnxruntime``. CPU build is 170x slower and lands in the onnx row. Installing both
  shadows the GPU build, which is how an entire published onnx column was measured on CPU without anyone noticing.
- A ``select_device`` that leaves ``CUDA_VISIBLE_DEVICES`` alone. Older ones blank it for a CPU device, and because
  the CPU row runs first, that masks the GPU for the whole interpreter and every later format raises.
- ``tensorrt==10.11.0.33``. Version alone moves ratios 3pp.
- Build and time engines in one interpreter, they are version-locked.
"""

import csv
import hashlib
import json
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch

import ultralytics
from ultralytics import YOLO
from ultralytics.nn.tasks import yaml_model_load
from ultralytics.utils.benchmarks import ProfileModels
from ultralytics.utils.torch_utils import get_gpu_info

ROUNDS = 8
SEED = 0
BIAS_FILL = 1e-3
TIMER = "predictor-speed"  # rows measured before 2026-07-27 used "cuda-events"

# Format tables: name -> (Variant attribute holding the artifact, extra predict kwargs, timed runs). Warmup defaults to a
# tenth of the runs and only a `warmup=N` sweep changes it.
#
# Only TensorRT is timed in the paired rounds. The other three used to run back to back ahead of it, which cooled
# the card on the CPU row and then reheated it at an architecture-dependent rate, a confound balanced round order
# cannot cancel. They also cost about 25 minutes per suite against 1.5 for TensorRT. They now run once per variant
# after the rounds, into a sidecar, as export health rather than a paired comparison.
TIMED_FORMATS = {"trt": ("engine", {}, 100)}
SIDECAR_FORMATS = {
    "pt_cpu": ("weights", {"device": "cpu", "quantize": 32}, 20),
    "pt16": ("weights", {"quantize": 16}, 100),
    "onnx": ("onnx", {}, 100),
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
        weights_state (str): `trained` or `materialized`, carried into the CSV because a materialized absolute reads 2
            to 3% high and this repo derives every other architecture's absolute from the baseline's.
        source (str): The yaml or checkpoint this was built from, so a row names its own graph rather than a tag.
    """

    name: str
    weights: Path
    onnx: Path
    engine: Path
    model_cls: type
    params_m: float
    gflops: float
    weights_state: str = "trained"
    source: str = ""


def parse_session(argv):
    """Split ``<lane>-<scale> [arm,arm] [warmup=N]`` into the lane, the scale, the session id and the warmup override.

    A warmup override is its own measurement occasion, so it goes into the session id. That keeps an exploratory sweep
    from writing over the standard run's csv, rounds and env files, and keeps its rows visibly outside the standard
    cohort. The suffix lands after the lane and scale are read off, so it cannot be mistaken for the scale.

    Returns:
        (str): Lane name, `lane-a` or `lane-b`.
        (str): Scale letter.
        (str): Session id, carrying a `-w{N}` suffix when an override was given.
        (list): The remaining arguments, at most one comma-joined arm list.
        (int | None): The warmup override, or None to keep the tenth-of-runs default.
    """
    warmup = next((int(a.split("=")[1]) for a in argv if a.startswith("warmup=")), None)
    session, *rest = (a for a in argv if not a.startswith("warmup="))
    lane, scale = session.rsplit("-", 1)
    return lane, scale, session if warmup is None else f"{session}-w{warmup}", rest, warmup


def env_info(imgsz, variants, warmup):
    """Collect the software, hardware and artifact provenance a latency number is only valid under."""
    providers = onnxruntime.get_available_providers()
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
        "timed_runs": {fmt: runs for fmt, (_, _, runs) in {**TIMED_FORMATS, **SIDECAR_FORMATS}.items()},
        # Resolved rather than the override, so a standard session is self-describing. Sidecar formats always take
        # the default, since they are export health and not a paired comparison.
        "timed_warmup": {fmt: runs // 10 if warmup is None else warmup for fmt, (_, _, runs) in TIMED_FORMATS.items()},
        "rounds": ROUNDS,
        "init_seed": SEED,
        "bias_fill": BIAS_FILL,
        **{m: __import__(m).__version__ for m in ("onnxruntime", "tensorrt", "onnx")},
        # Which provider the onnx row actually ran on, recorded rather than required. Installing onnxruntime beside
        # onnxruntime-gpu shadows the GPU build, and the rows measured before 2026-07-27 silently took the CPU path
        # because of it. This mirrors the selection in ultralytics/nn/backends/onnx.py, which asks for CUDA then CPU.
        "onnx_providers_available": providers,
        "onnx_provider_used": "CUDAExecutionProvider"
        if "CUDAExecutionProvider" in providers
        else "CPUExecutionProvider",
        "artifacts": {v.name: {a: str(getattr(v, a)) for a in ("weights", "onnx", "engine")} for v in variants},
    }


def materialize_yaml(name, yaml, outdir, ident, model_cls):
    """Build an untrained yaml into a checkpoint whose biases are never exactly zero.

    Standard PyTorch and DEIM init leaves whole bias tensors at exactly zero, and benchmarking that state is not latency
    equivalent to benchmarking trained weights. The shift is signed and per-architecture (baseline +0.97%, ffnattn2
    -2.35%, attn2 -3.99% against trained), so normalizing a zero-bias arm to a zero-bias baseline amplifies it and
    erased 78% of a real attention penalty. Filling those tensors moved both architecture ratios back toward the trained
    ones, by a paired per-round mean of 0.32pp on ffnattn2 and 1.12pp on attn2.

    Prefer a trained checkpoint whenever one exists. This is a proxy for the yaml-only case, not an equal substitute,
    and it is under-validated: the mechanism is unidentified, the fill is uniform and positive rather than trained-like
    (trained biases are centered and 100x larger, median absolute 0.12 to 0.15), and the 8 rounds behind those figures
    measure timing repeatability only, not seed or engine-builder variance. The ONNX graphs of the two arms are
    near-identical, same `Add`, `Conv` and `Gemm` counts and 1 to 5 `Identity` nodes apart, so whatever TensorRT does
    with a constant-zero weight it does during engine build, not at export.

    Args:
        name (str): Short tag, which alone would not identify the checkpoint.
        yaml (str | Path): Model yaml to build.
        outdir (str | Path): Directory the checkpoint is written to.
        ident (str): Hash of every input this checkpoint depends on, the yaml, the bias fill, the seed and the
            facade. Tags are stable across arm generations, so a tag-keyed name serves a checkpoint built from the
            previous yaml behind that tag.
        model_cls (type): Facade that resolves the task.

    Returns:
        (Path): The written checkpoint.
    """
    ckpt = Path(outdir) / f"{name}-{ident}.pt"
    if ckpt.exists():
        return ckpt
    torch.manual_seed(SEED)
    model = model_cls(str(yaml))
    touched = 0
    with torch.no_grad():
        for param_name, p in model.model.named_parameters():
            if param_name.endswith(".bias") and not torch.count_nonzero(p):
                p.fill_(BIAS_FILL)
                touched += 1
    model.save(ckpt)
    print(f"  {ckpt.name}: filled {touched} wholly-zero bias tensors with {BIAS_FILL}", flush=True)
    return ckpt


def pinned_fp32_attn(onnx, engine):
    """Build the FP16 engine through Esat's builder, with attention softmax and norm internals pinned to fp32.

    This is the Lane B engine build. Without the pin DINOv3 decomposed attention overflows fp16, and the `debug` flag is
    his, so these engines match the ones his published numbers came from. The import is local because `working_dir` only
    exists in the Lane B checkout.
    """
    from working_dir.export_deimv2 import build_engine_fp16

    build_engine_fp16(onnx, engine, half=True, fp32_attn=True, debug=True)


def build_variant(name, weights, outdir, imgsz, device="0", model_cls=YOLO, engine_builder=None):
    """Export the FP32 ONNX and FP16 engine for one architecture, reusing whatever already exists at that imgsz.

    Args:
        name (str): Short tag, also the artifact filename stem, which additionally carries an identity hash.
        weights (str | Path): Trained checkpoint, or a model yaml that `materialize_yaml` turns into one.
        outdir (str | Path): Directory the artifacts are written to.
        imgsz (int): Square input size baked into both exports and into the artifact stem.
        device (str): CUDA device index used for the export.
        model_cls (type): Facade that resolves the task, YOLO for Lane A and RTDETR for the DEIM yamls.
        engine_builder (Callable, optional): Called as ``(onnx_path, engine_path)`` instead of the stock engine
            export. Lane B passes ``pinned_fp32_attn``. Lane A needs no pin.

    Returns:
        (Variant): The variant with artifact paths, facade and fused model metrics populated.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    source = str(weights)
    materialized = Path(weights).suffix == ".yaml"
    # Artifacts are keyed by what produced them, not by tag. A tag-keyed cache silently reuses an engine built
    # from a different yaml, a different bias fill or a different TensorRT, each of which moves the number. A yaml
    # is identified by its resolved config and a trained checkpoint by its bytes. Never a materialized checkpoint's
    # bytes, which carry the save timestamp and so name when it was written rather than what it holds.
    ident = repr(yaml_model_load(weights)).encode() if materialized else Path(weights).read_bytes()
    ident += f"|{BIAS_FILL}|{SEED}|{model_cls.__name__}".encode()  # everything the checkpoint itself depends on
    if materialized:
        weights = materialize_yaml(name, weights, outdir, hashlib.sha256(ident).hexdigest()[:8], model_cls)
    import tensorrt  # local, so a checkout without it can still import this module to inspect the suite table

    key = hashlib.sha256(
        ident + f"|{imgsz}|{tensorrt.__version__}|{getattr(engine_builder, '__name__', 'stock')}".encode()
    ).hexdigest()[:8]
    onnx, engine = outdir / f"{name}-{imgsz}-{key}.onnx", outdir / f"{name}-{imgsz}-{key}.engine"

    def export(fmt, path, **extra):
        """Export one format from a fresh instance, since export mutates the model in place."""
        exported = model_cls(str(weights)).export(
            format=fmt, imgsz=imgsz, device=device, batch=1, verbose=False, **extra
        )
        shutil.move(str(exported), path)

    if not onnx.exists():
        export("onnx", onnx)
    if not engine.exists():
        if engine_builder:
            engine_builder(onnx, engine)
        else:
            export("engine", engine, quantize=16)
    model = model_cls(str(weights))
    model.fuse()  # so params and GFLOPs describe the deployed graph
    _, params, _, gflops = model.info(imgsz=imgsz)
    return Variant(
        name,
        Path(weights),
        onnx,
        engine,
        model_cls,
        params / 1e6,
        gflops,
        "materialized" if materialized else "trained",
        source,
    )


def profile_model(model, image, runs, warmup, **predict_kwargs):
    """Warm up, collect `runs` predictor speed records, sigma clip, and return inference and pipeline means.

    Args:
        model (Model): Loaded model in any backend the predictor supports.
        image (np.ndarray): HWC uint8 input reused for every call.
        runs (int): Timed calls to collect.
        warmup (int): Untimed calls made first, which also set how hot the card is when timing starts.
        **predict_kwargs (Any): Forwarded to the predictor, typically imgsz, device and quantize.

    Returns:
        (dict): Sigma-clipped mean inference ms under `inf`, its standard deviation under `inf_std`, and preprocess plus
            inference plus postprocess under `total`.
    """
    for _ in range(warmup):
        model(image, verbose=False, **predict_kwargs)
    pre, inf, post = [], [], []
    for _ in range(runs):
        speed = model(image, verbose=False, **predict_kwargs)[0].speed
        pre.append(speed["preprocess"])
        inf.append(speed["inference"])
        post.append(speed["postprocess"])
    clipped = ProfileModels.iterative_sigma_clipping(inf)
    mean = float(clipped.mean())
    return {"inf": mean, "inf_std": float(clipped.std()), "total": float(np.mean(pre)) + mean + float(np.mean(post))}


def profile_variant(variant, image, imgsz, device, formats=None, warmup=None):
    """Profile the given formats back to back through the variant's own facade, with no gap between."""
    return {
        fmt: profile_model(
            variant.model_cls(str(getattr(variant, attr))),
            image,
            runs,
            runs // 10 if warmup is None else warmup,
            imgsz=imgsz,
            **{"device": device, **extra},
        )
        for fmt, (attr, extra, runs) in (formats or TIMED_FORMATS).items()
    }


def summarize_rounds(per_round, variants, baselines, session, imgsz, warmup):
    """Reduce per-round records to one row per variant and format, with paired deltas against the variant's baseline."""
    series = {}
    for record in per_round:  # one list per variant and format, in round order, so zip() pairs them correctly
        for fmt in TIMED_FORMATS:
            series.setdefault((record["variant"], fmt), []).append(record[f"{fmt}_inf"])

    rows = []
    for variant in variants:
        baseline = baselines[variant.name]
        for fmt in TIMED_FORMATS:
            own, ref = series[variant.name, fmt], series[baseline, fmt]
            base_median = float(np.median(ref))
            median = float(np.median(own))
            rows.append(
                {
                    "session": session,
                    "variant": variant.name,
                    # The operating size, because a row measured at 512 and one at 640 are otherwise
                    # indistinguishable in the ledger and the two are not comparable.
                    "imgsz": imgsz,
                    "format": fmt,
                    "params_M_fused": round(variant.params_m, 2),
                    "gflops_fused": round(variant.gflops, 1),
                    "median_ms": round(median, 4),
                    # Both medians, because deltas from two sessions may only be chained through their ratios. The
                    # percentage points are not additive: -6.56 against -9.58 is 3.02pp but a 3.34% latency ratio.
                    "base_median_ms": round(base_median, 4),
                    "ratio_vs_base": round(median / base_median, 6),
                    "delta_vs_base_pct": round(100 * (median - base_median) / base_median, 2),
                    "ab_wins": "" if variant.name == baseline else f"{sum(a < b for a, b in zip(own, ref))}/{ROUNDS}",
                    "baseline": baseline,
                    "yaml": variant.source,
                    "engine": variant.engine.name,
                    "weights_state": variant.weights_state,
                    "timer": TIMER,
                    # Beside timer because both name the cohort a row belongs to, and unlike runs and rounds this
                    # one varies per invocation, so a row cannot be placed without it. Resolved, not the override.
                    "warmup": TIMED_FORMATS[fmt][2] // 10 if warmup is None else warmup,
                }
            )
    return rows


def write_csv(path, rows):
    """Write a list of uniform dicts as a CSV with a header taken from the first row."""
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_benchmark(variants, baselines, out_csv, session, imgsz, device="0", warmup=None):
    """Run ROUNDS order-balanced rounds over every variant and write the summary, per-round and provenance files.

    Args:
        variants (list[Variant]): Variants to compare, all artifacts already built.
        baselines (dict): Maps each variant name to the name of the variant its delta is taken against. Keying it per
            variant is what lets one suite span scales, since a row only means architecture when its baseline is at its
            own scale.
        out_csv (str | Path): Summary CSV path. Per-round rows, other formats and provenance go beside it.
        session (str): Identifies this measurement session, since a delta is only exact against arms measured in the
            same one. Chaining across sessions needs a bridge arm present in both.
        imgsz (int): Square input size, which must match the size the artifacts were exported at.
        device (str): CUDA device index passed to the predictor.
        warmup (int, optional): Untimed calls before each timed block, overriding the tenth-of-runs default. Rows
            measured under an override are exploratory and must not be ingested against the standard cohort.
    """
    out_csv = Path(out_csv)
    image = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
    print(f"=== timer {TIMER}, {len(variants)} variants, {len(set(baselines.values()))} baselines", flush=True)

    per_round = []
    for rnd in range(ROUNDS):
        for variant in variants if rnd % 2 == 0 else variants[::-1]:
            timings = profile_variant(variant, image, imgsz, device, warmup=warmup)
            per_round.append(
                {
                    "round": rnd,
                    "variant": variant.name,
                    **{f"{f}_{k}": round(v, 4) for f, t in timings.items() for k, v in t.items()},
                }
            )
            summary = " ".join(f"{f}={timings[f]['inf']:7.3f}" for f in TIMED_FORMATS)
            print(f"  round {rnd + 1}/{ROUNDS} {variant.name:<24} {summary}", flush=True)

    rows = summarize_rounds(per_round, variants, baselines, session, imgsz, warmup)
    write_csv(out_csv, rows)
    write_csv(out_csv.with_suffix(".rounds.csv"), per_round)
    out_csv.with_suffix(".env.json").write_text(json.dumps(env_info(imgsz, variants, warmup), indent=2))

    # Export health, once per variant and outside the rounds so it cannot preheat the card for the timed format.
    print("\n=== other formats, one pass, not paired", flush=True)
    sidecar = []
    for variant in variants:
        timings = profile_variant(variant, image, imgsz, device, SIDECAR_FORMATS)
        sidecar += [
            {"variant": variant.name, "format": f, **{k: round(v, 4) for k, v in t.items()}} for f, t in timings.items()
        ]
        print(f"  {variant.name:<24}" + " ".join(f"{f}={timings[f]['inf']:8.3f}" for f in SIDECAR_FORMATS), flush=True)
    write_csv(out_csv.with_suffix(".formats.csv"), sidecar)

    print("\n=== medians", flush=True)
    for row in rows:
        print(
            f"  {row['variant']:<24}{row['format']:<6}{row['median_ms']:9.4f}"
            f"{row['delta_vs_base_pct']:+8.2f}% vs {row['baseline']:<12}{row['ab_wins']}",
            flush=True,
        )
    print(f"\nwrote {out_csv} plus .rounds.csv, .formats.csv and .env.json", flush=True)
