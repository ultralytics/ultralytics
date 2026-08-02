# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Benchmark 224 image-encoder features with TensorRT on a T4.

Run as ``python bench/t4_encoder_bench.py <session> [model,model]``. With no model list, the suite prepares and times
every row shown on the Orc Backbones page. A model list also includes each requested model's baseline.

The timed graph ends at the unnormalized feature used by ImageNet kNN: pooled Classify-head features for Ultralytics
checkpoints and the released encoder's CLS or pooled feature for references. Image preprocessing and kNN L2
normalization stay outside the engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import ultralytics
from t4_bench_common import BIAS_FILL, ROUNDS, SEED, materialize_yaml, write_csv
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.nn.tasks import load_checkpoint, yaml_model_load
from ultralytics.nn.teacher_model import PIPELINE_IMAGE_MEAN, PIPELINE_IMAGE_STD
from ultralytics.utils import ROOT
from ultralytics.utils.benchmarks import ProfileModels
from ultralytics.utils.export.engine import best_onnx_opset, onnx2engine, torch2onnx
from ultralytics.utils.knn_eval import yolo_cls_features
from ultralytics.utils.torch_utils import get_gpu_info

IMGSZ = 224
WARMUP = 10
TIMED = 100
REFERENCE_BASELINE = "dinov3:vits16"
DEFAULT_WEIGHTS = Path("/root/autodl-tmp/data/encoder-224-weights")
DEFAULT_OUTPUT = Path("/root/autodl-tmp/data/t4-encoder-224")
MODEL_ROOT = ROOT / "cfg/models/26"
SCRIPT_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
GIT_COMMIT = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()

REFERENCE_KEYS = {
    "dinov3:vits16plus": "dinov3:vits16plus",
    "tips:v2b14": "tips:v2b14",
    "dinov3:convnext-small": "dinov3:convnexts",
    "dinov2:vits14": "dinov2:vits14",
    "dinov3:vits16": "dinov3:vits16",
    "dinov3:convnext-tiny": "dinov3:convnextt",
    "eupe:vits16": "eupe:vits16",
    "tips:v1b14": "tips:v1b14",
    "tips:v1s14": "tips:v1s14",
    "mobileclip:s0": "mobileclip:s0",
    "mobileclip:s1": "mobileclip:s1",
    "mobileclip:s2": "mobileclip:s2",
    "mobileclip2:s0": "mobileclip2:s0",
    "mobileclip2:s2": "mobileclip2:s2",
    "dune:vitb14": "dune:vitb14",
    "dune:vits14": "dune:vits14",
}

INTERNAL_RUNS = {
    "yolo26n-sppf-cls": ("ph1-7src-conv-n-sppf-dinov3-vitl16", "yolo26n-sppf-cls"),
    "yolo26s-sppf-cls": ("phase1-12src-conv-s-sppf-dinov3-vitl16", "yolo26s-sppf-cls"),
    "yolo26m-sppf-cls": ("ph1-12src-conv-m-sppf-dinov3-vitl16", "yolo26m-sppf-cls"),
    "yolo26l-sppf-cls": ("phase1-12src-conv-l-sppf-dinov3-vitl16", "yolo26l-sppf-cls"),
    "yolo26x-sppf-cls": ("phase1-12src-conv-x-sppf-dinov3-vitl16", "yolo26x-sppf-cls"),
    "ultravit-n-290726-cls": ("ph1-12src-ultravit-n-290726-dinov3-vitl16", "yolo26n-sppf-cls"),
    "ultravit-s-290726-cls": ("ph1-12src-ultravit-s-290726-dinov3-vitl16", "yolo26s-sppf-cls"),
    "ultravit-m-290726-cls": ("ph1-12src-ultravit-m-290726-dinov3-vitl16", "yolo26m-sppf-cls"),
    "ultravit-l-290726-cls": ("ph1-12src-ultravit-l-290726-dinov3-vitl16", "yolo26l-sppf-cls"),
    "ultravit-x-290726-cls": ("ph1-12src-ultravit-x-290726-dinov3-vitl16", "yolo26x-sppf-cls"),
    "ultravit-l-attn2-cls": (
        "phase1-12src-ultravit-l-repmixer-fastvitffn-attn2-dinov3-vitl16",
        "yolo26l-sppf-cls",
    ),
    "ultravit-x-attn2-cls": ("ph1-12src-ultravit-x-attn2-dinov3-vitl16", "yolo26x-sppf-cls"),
}

EXPLORATORY_YAMLS = {
    "ultravit-n-020826-1-cls": ("yolo26n-ultravit-020826-1-cls.yaml", "yolo26n-sppf-cls"),
    "ultravit-n-020826-2-cls": ("yolo26n-ultravit-020826-2-cls.yaml", "yolo26n-sppf-cls"),
    "ultravit-n-020826-3-cls": ("yolo26n-ultravit-020826-3-cls.yaml", "yolo26n-sppf-cls"),
    "ultravit-s-010826-2a-cls": ("yolo26s-ultravit-010826-2a-cls.yaml", "yolo26s-sppf-cls"),
    "ultravit-s-010826-2b-cls": ("yolo26s-ultravit-010826-2b-cls.yaml", "yolo26s-sppf-cls"),
    "ultravit-s-020826-cls": ("yolo26s-ultravit-020826-cls.yaml", "yolo26s-sppf-cls"),
    "ultravit-s-020826-1-cls": ("yolo26s-ultravit-020826-1-cls.yaml", "yolo26s-sppf-cls"),
    "ultravit-s-020826-2-cls": ("yolo26s-ultravit-020826-2-cls.yaml", "yolo26s-sppf-cls"),
    "ultravit-l-stagematch-cls": ("yolo26l-ultravit-290726-stagematch-cls.yaml", "yolo26l-sppf-cls"),
    "ultravit-l-010826-1-cls": ("yolo26l-ultravit-010826-1-cls.yaml", "yolo26l-sppf-cls"),
    "ultravit-l-010826-2-cls": ("yolo26l-ultravit-010826-2-cls.yaml", "yolo26l-sppf-cls"),
    "ultravit-l-deepbal-cls": ("yolo26l-ultravit-290726-deepbal-cls.yaml", "yolo26l-sppf-cls"),
    "ultravit-l-deepbal-p5lean-cls": (
        "yolo26l-ultravit-290726-deepbal-p5lean-cls.yaml",
        "yolo26l-sppf-cls",
    ),
}


@dataclass(frozen=True)
class ModelSpec:
    """Describe one displayed encoder and its paired baseline.

    Attributes:
        name (str): Display name used by Orc and result files.
        source (str): Teacher registry key or Orc run ID.
        baseline (str): Display name measured as the paired baseline.
        reference (bool): Whether source names a released teacher.
    """

    name: str
    source: str
    baseline: str
    reference: bool


@dataclass(frozen=True)
class EncoderVariant:
    """Carry one prepared feature graph and its benchmark provenance.

    Attributes:
        name (str): Display name used by Orc and result files.
        source (str): Teacher registry key or Orc run ID.
        source_sha256 (str): Hash of checkpoint bytes or released model state.
        baseline (str): Display name measured as the paired baseline.
        params_m (float): Parameters used by the exported feature graph in millions.
        onnx (Path): Fixed-shape FP32 ONNX artifact.
        engine (Path): Fixed-shape FP16 TensorRT artifact.
        sample (torch.Tensor): Deterministic preprocessed input used for parity checks.
        parity (dict): ONNX and TensorRT parity measurements.
    """

    name: str
    source: str
    source_sha256: str
    baseline: str
    params_m: float
    onnx: Path
    engine: Path
    sample: torch.Tensor
    parity: dict


class YOLOFeature(nn.Module):
    """Expose the pooled feature used by ``yolo_cls_features``.

    Attributes:
        model (nn.Module): Classification model consumed by the canonical kNN feature reader.
        params (int): Parameters traversed by the exported feature graph.
    """

    def __init__(self, model: nn.Module):
        """Initialize from a loaded Ultralytics classification model.

        Args:
            model (nn.Module): Loaded classification or image-encoder model.
        """
        super().__init__()
        self.model = model
        self.params = sum(p.numel() for p in model.model[:-1].parameters()) + sum(
            p.numel() for p in model.model[-1].conv.parameters()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into pooled kNN features.

        Args:
            x (torch.Tensor): Preprocessed images shaped (B, 3, 224, 224).

        Returns:
            (torch.Tensor): Unnormalized image features shaped (B, D).
        """
        return yolo_cls_features(self.model, x)


class ReferenceFeature(nn.Module):
    """Expose a released teacher's CLS or pooled kNN feature.

    Attributes:
        teacher (nn.Module): Released teacher wrapper from ``TEACHER_REGISTRY``.
    """

    def __init__(self, teacher: nn.Module):
        """Initialize from a released teacher wrapper.

        Args:
            teacher (nn.Module): Teacher with a meaningful CLS output.
        """
        super().__init__()
        self.teacher = teacher

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode images into released kNN features.

        Args:
            x (torch.Tensor): Images preprocessed for the released encoder.

        Returns:
            (torch.Tensor): Unnormalized image features shaped (B, D).
        """
        return self.teacher.encode(x).cls


def suite() -> list[ModelSpec]:
    """Return the models displayed on the Orc Backbones page.

    Returns:
        (list[ModelSpec]): Full 224 encoder cohort.
    """
    references = [ModelSpec(name, key, REFERENCE_BASELINE, True) for name, key in REFERENCE_KEYS.items()]
    internal = []
    for name, (run_id, baseline) in INTERNAL_RUNS.items():
        internal.append(ModelSpec(name, run_id, baseline, False))
    return references + internal


def hash_state_dict(model: nn.Module) -> str:
    """Hash model tensor names, shapes, dtypes, and bytes.

    Args:
        model (nn.Module): Loaded released model.

    Returns:
        (str): SHA256 digest.
    """
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{tuple(tensor.shape)}|{tensor.dtype}".encode())
        digest.update(tensor.numpy())
    return digest.hexdigest()


def load_feature(spec: ModelSpec, weights_dir: Path) -> tuple[nn.Module, torch.Tensor, str]:
    """Load one feature graph, deterministic input, and source hash.

    Args:
        spec (ModelSpec): Displayed encoder specification.
        weights_dir (Path): Directory containing staged internal checkpoints.

    Returns:
        model (nn.Module): Eval-mode feature graph on CPU.
        sample (torch.Tensor): Deterministic preprocessed input.
        source_sha256 (str): Source checkpoint or released-state digest.
    """
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 3, IMGSZ, IMGSZ, generator=generator)
    if spec.reference:
        from ultralytics.nn.teacher_model import build_teacher_model

        teacher = build_teacher_model(spec.source, device=torch.device("cpu"), normalize_input=False)
        mean = torch.tensor(teacher.IMAGE_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(teacher.IMAGE_STD).view(1, 3, 1, 1)
        model = ReferenceFeature(teacher).eval()
        source_sha256 = hash_state_dict(model)
        if spec.source.startswith("tips:"):
            tips = teacher.model
            patches = (IMGSZ // tips.patch_size) ** 2
            pos_embed = tips.interpolate_pos_encoding(
                torch.empty(1, patches + 1, tips.pos_embed.shape[-1]), IMGSZ, IMGSZ
            ).detach()
            tips.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
        return model, (image - mean) / std, source_sha256

    if spec.source.endswith(".yaml"):
        yaml = MODEL_ROOT / spec.source
        ident = repr(yaml_model_load(yaml)).encode() + f"|{BIAS_FILL}|{SEED}|{YOLO.__name__}".encode()
        checkpoint = materialize_yaml(spec.name, yaml, weights_dir, hashlib.sha256(ident).hexdigest()[:8], YOLO)
    else:
        checkpoint = weights_dir / f"{spec.source}.pt"
    model = YOLOFeature(load_checkpoint(checkpoint, device="cpu", fuse=True)[0]).eval()
    mean = torch.tensor(PIPELINE_IMAGE_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(PIPELINE_IMAGE_STD).view(1, 3, 1, 1)
    digest = hashlib.sha256()
    with checkpoint.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return model, (image - mean) / std, digest.hexdigest()


def parity(model: nn.Module, sample: torch.Tensor, onnx: Path, engine: Path, device: torch.device) -> dict:
    """Compare PyTorch, ONNX Runtime, and TensorRT feature outputs.

    Args:
        model (nn.Module): CPU feature graph used for ONNX export.
        sample (torch.Tensor): Deterministic preprocessed input.
        onnx (Path): Exported ONNX artifact.
        engine (Path): Exported TensorRT artifact.
        device (torch.device): CUDA device used by TensorRT.

    Returns:
        (dict): Output shape, cosine similarity, and relative norm errors.
    """
    with torch.no_grad():
        expected = model(sample).float()
    providers = [
        p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in onnxruntime.get_available_providers()
    ]
    session = onnxruntime.InferenceSession(str(onnx), providers=providers)
    actual_onnx = torch.from_numpy(session.run(None, {session.get_inputs()[0].name: sample.numpy()})[0]).float()
    backend = AutoBackend(str(engine), device=device, verbose=False)
    actual_trt = backend(sample.to(device, dtype=torch.float16 if backend.fp16 else torch.float32)).cpu().float()

    def compare(actual):
        result = {
            "cosine": float(F.cosine_similarity(expected.flatten(1), actual.flatten(1)).mean()),
            "relative_norm_error": float((actual.norm() - expected.norm()).abs() / expected.norm()),
        }
        result["passed"] = result["cosine"] >= 0.999 and result["relative_norm_error"] <= 0.01
        return result

    onnx_result, trt_result = compare(actual_onnx), compare(actual_trt)
    assert expected.shape == actual_onnx.shape == actual_trt.shape
    assert torch.isfinite(actual_onnx).all() and torch.isfinite(actual_trt).all()
    assert onnx_result["passed"]
    return {"shape": list(expected.shape), "onnx": onnx_result, "trt": trt_result}


def prepare(spec: ModelSpec, weights_dir: Path, output_dir: Path, device: torch.device) -> EncoderVariant:
    """Build and validate one fixed-shape encoder engine.

    Args:
        spec (ModelSpec): Displayed encoder specification.
        weights_dir (Path): Directory containing staged internal checkpoints.
        output_dir (Path): Directory for ONNX and TensorRT artifacts.
        device (torch.device): CUDA device used for TensorRT parity.

    Returns:
        (EncoderVariant): Prepared benchmark variant.
    """
    model, sample, source_sha256 = load_feature(spec, weights_dir)
    import onnx
    import tensorrt

    key = hashlib.sha256(
        f"{source_sha256}|{GIT_COMMIT}|{SCRIPT_SHA256}|{IMGSZ}|{onnx.__version__}|{tensorrt.__version__}".encode()
    ).hexdigest()[:8]
    stem = output_dir / f"{spec.name}-{IMGSZ}-{key}"
    onnx_path, engine_path = stem.with_suffix(".onnx"), stem.with_suffix(".engine")
    if not onnx_path.exists():
        torch2onnx(
            model,
            sample,
            onnx_path,
            opset=best_onnx_opset(onnx),
            input_names=["images"],
            output_names=["features"],
        )
    if not engine_path.exists():
        onnx2engine(
            str(onnx_path),
            engine_path,
            quantize=16,
            shape=tuple(sample.shape),
            metadata={"task": "classify", "batch": 1, "imgsz": [IMGSZ, IMGSZ], "names": {0: "feature"}},
        )
    result = parity(model, sample, onnx_path, engine_path, device)
    return EncoderVariant(
        spec.name,
        spec.source,
        source_sha256,
        spec.baseline,
        (model.params if isinstance(model, YOLOFeature) else sum(p.numel() for p in model.parameters())) / 1e6,
        onnx_path,
        engine_path,
        sample,
        result,
    )


def profile(variant: EncoderVariant, device: torch.device) -> dict:
    """Measure one TensorRT engine after ten warmups.

    Args:
        variant (EncoderVariant): Prepared encoder engine.
        device (torch.device): CUDA device used for timing.

    Returns:
        (dict): Sigma-clipped mean inference latency in milliseconds.
    """
    backend = AutoBackend(str(variant.engine), device=device, verbose=False)
    sample = variant.sample.to(device, dtype=torch.float16 if backend.fp16 else torch.float32)
    for _ in range(WARMUP):
        backend(sample)
    values = []
    for _ in range(TIMED):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        backend(sample)
        torch.cuda.synchronize(device)
        values.append(1000 * (time.perf_counter() - start))
    return {"trt_inf": round(float(ProfileModels.iterative_sigma_clipping(values).mean()), 4)}


def run_order_balanced(variants: list[EncoderVariant], device: torch.device) -> list[dict]:
    """Profile encoders for eight rounds, reversing their order on odd rounds.

    Args:
        variants (list[EncoderVariant]): Prepared encoder engines.
        device (torch.device): CUDA device used for timing.

    Returns:
        (list[dict]): Per-round latency records.
    """
    records = []
    for rnd in range(ROUNDS):
        for variant in variants if rnd % 2 == 0 else variants[::-1]:
            values = profile(variant, device)
            records.append({"round": rnd, "variant": variant.name, **values})
            print(f"  round {rnd + 1}/{ROUNDS} {variant.name:<32} trt={values['trt_inf']:7.3f}", flush=True)
    return records


def summarize(records: list[dict], variants: list[EncoderVariant], session: str) -> list[dict]:
    """Reduce per-round records to one same-session row per encoder.

    Args:
        records (list[dict]): Per-round latency records.
        variants (list[EncoderVariant]): Prepared encoders and baselines.
        session (str): Measurement occasion identifier.

    Returns:
        (list[dict]): Median latency and paired comparisons.
    """
    series = {
        variant.name: [row["trt_inf"] for row in records if row["variant"] == variant.name] for variant in variants
    }
    rows = []
    for variant in variants:
        own, baseline = series[variant.name], series[variant.baseline]
        median, base_median = float(np.median(own)), float(np.median(baseline))
        rows.append(
            {
                "cohort": "encoder_trt_only_w10_n100_r8",
                "session": session,
                "model": variant.name,
                "source": variant.source,
                "source_sha256": variant.source_sha256,
                "imgsz": IMGSZ,
                "batch": 1,
                "precision": "fp16",
                "params_M_feature": round(variant.params_m, 2),
                "median_ms": round(median, 4),
                "baseline": variant.baseline,
                "base_median_ms": round(base_median, 4),
                "ratio_vs_base": round(median / base_median, 6),
                "delta_vs_base_pct": round(100 * (median - base_median) / base_median, 2),
                "ab_wins": ""
                if variant.name == variant.baseline
                else f"{sum(a < b for a, b in zip(own, baseline))}/{ROUNDS}",
                "engine": variant.engine.name,
                "timer": "sync-perf-counter",
                "warmup": WARMUP,
            }
        )
    return rows


def main():
    """Prepare, validate, and profile the requested 224 encoder cohort."""
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("models", nargs="?", help="comma-separated display names. Omit for the full suite")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()

    specs = suite()
    available = specs + [
        ModelSpec(name, source, baseline, False) for name, (source, baseline) in EXPLORATORY_YAMLS.items()
    ]
    by_name = {spec.name: spec for spec in available}
    if args.models:
        requested = set(args.models.split(","))
        assert requested <= set(by_name), f"unknown models: {sorted(requested - set(by_name))}"
        requested |= {by_name[name].baseline for name in requested}
        specs = [spec for spec in available if spec.name in requested]
    assert len({spec.name for spec in specs}) == len(specs)
    assert all(spec.baseline in {item.name for item in specs} for spec in specs)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda", args.device)
    variants = []
    for index, spec in enumerate(specs, 1):
        print(f"=== prepare {index}/{len(specs)} {spec.name}", flush=True)
        variants.append(prepare(spec, args.weights_dir, args.output_dir, device))

    stem = args.output_dir / args.session.replace("-", "_")
    write_csv(
        stem.with_suffix(".parity.csv"),
        [
            {
                "model": variant.name,
                "source": variant.source,
                "source_sha256": variant.source_sha256,
                "params_M_feature": round(variant.params_m, 2),
                "shape": "x".join(map(str, variant.parity["shape"])),
                "onnx_cosine": variant.parity["onnx"]["cosine"],
                "onnx_relative_norm_error": variant.parity["onnx"]["relative_norm_error"],
                "trt_cosine": variant.parity["trt"]["cosine"],
                "trt_relative_norm_error": variant.parity["trt"]["relative_norm_error"],
                "trt_parity_passed": variant.parity["trt"]["passed"],
            }
            for variant in variants
        ],
    )
    if args.prepare_only:
        return

    records = run_order_balanced(variants, device)
    rows = summarize(records, variants, args.session)
    write_csv(stem.with_suffix(".csv"), rows)
    write_csv(stem.with_suffix(".rounds.csv"), records)
    stem.with_suffix(".env.json").write_text(
        json.dumps(
            {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "torch_cuda": torch.version.cuda,
                "gpu": get_gpu_info(args.device),
                "ultralytics": ultralytics.__version__,
                "ultralytics_path": ultralytics.__file__,
                "git_commit": GIT_COMMIT,
                "script_sha256": SCRIPT_SHA256,
                "onnxruntime": onnxruntime.__version__,
                "onnx_providers": onnxruntime.get_available_providers(),
                "tensorrt": __import__("tensorrt").__version__,
                "imgsz": IMGSZ,
                "batch": 1,
                "warmup": WARMUP,
                "timed": TIMED,
                "rounds": ROUNDS,
                "artifacts": {
                    variant.name: {
                        "source": variant.source,
                        "source_sha256": variant.source_sha256,
                        "onnx": str(variant.onnx),
                        "engine": str(variant.engine),
                    }
                    for variant in variants
                },
            },
            indent=2,
        )
    )
    for row in rows:
        print(
            f"{row['model']:<32}{row['median_ms']:8.4f} ms {row['delta_vs_base_pct']:+7.2f}% "
            f"vs {row['baseline']} {row['ab_wins']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
