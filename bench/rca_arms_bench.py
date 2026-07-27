"""Do random-init weights reproduce Esat's architecture ratios once the timer is his, or do the zero biases have to go.

Nine variants in one interleaved run: three architectures crossed with trained, random zero-bias and random
nonzero-bias weights. Every engine came from Esat's export_deimv2.py under TensorRT 10.11.0.33, so weight state is
the only thing that varies inside an architecture. Ratios are taken within each weight arm against that arm's own
DINOv3-S+ baseline, which is how Esat's published deltas are defined.
"""

import csv
import sys
from pathlib import Path

sys.path.insert(0, "/root/autodl-tmp/code/ultravit-lane-b")

from ultralytics import RTDETR  # noqa: E402

from t4_bench_common import Variant, run_benchmark  # noqa: E402

ESAT = {"dinov3splus": 0.0, "ffnattn2": -6.38, "attn2": 9.22}
ARCHS = ["dinov3splus", "ffnattn2", "attn2"]
TRAINED_STEM = {
    "dinov3splus": "rtdetr_dinov3sp_deim_deimv2Neck_coco",
    "ffnattn2": "rtdetr_ultravitX_fastVIT_attnv2_deim_deimv2Neck_fatih_coco",
    "attn2": "rtdetr_ultravitX_attnv2_deim_deimv2Neck_fatih_coco",
}
TRAINED_PT = {
    "dinov3splus": "dinov3sp_deim_deimv2Neck_coco.pt",
    "ffnattn2": "ultravitX_fastVIT_attnv2_deim_deimv2Neck_fatih_coco.pt",
    "attn2": "ultravitX_attnv2_deim_deimv2Neck_fatih_coco.pt",
}
SUFFIX = "_op17_nosim_norope_imgsz640_fp32attn_debug_fp16"
ESAT_DIR, RANDOM_DIR = Path("/root/autodl-tmp/esat-models"), Path("/root/autodl-tmp/data/rca-weights")
ENGINE_1011 = Path("/root/autodl-tmp/data/trt1011-engines")
OUT = Path("/root/autodl-tmp/data/t4_rca_arms.csv")


def make(name, weights, onnx, engine):
    """Wrap prebuilt artifacts in a Variant, taking fused metrics from the checkpoint."""
    model = RTDETR(str(weights))
    model.fuse()
    _, params, _, gflops = model.info(imgsz=640)
    return Variant(name, Path(weights), onnx, engine, RTDETR, params / 1e6, gflops)


variants = []
for arch in ARCHS:
    stem = TRAINED_STEM[arch]
    variants.append(
        make(
            f"trained-{arch}",
            ESAT_DIR / TRAINED_PT[arch],
            ESAT_DIR / f"{stem}{SUFFIX}.onnx",
            ENGINE_1011 / f"{stem}{SUFFIX}.engine",
        )
    )
    for arm in ("zerobias", "nonzerobias"):
        weights = RANDOM_DIR / f"{arch}_{arm}.pt"
        onnx = next(RANDOM_DIR.glob(f"*{arch}_{arm}*.onnx"))
        engine = next(RANDOM_DIR.glob(f"*{arch}_{arm}*.engine"))
        variants.append(make(f"{arm}-{arch}", weights, onnx, engine))

for v in variants:
    print(f"  {v.name:<24} {v.params_m:6.2f}M  engine={v.engine.name[:60]}", flush=True)

run_benchmark(variants, "trained-dinov3splus", OUT)

# Ratios are only meaningful inside a weight arm, so recompute them against each arm's own baseline.
med = {}
with OUT.open() as f:
    for row in csv.DictReader(f):
        if row["format"] == "trt":
            med[row["variant"]] = float(row["median_ms"])

print("\n=== trt ratios vs that arm's own DINOv3-S+ baseline", flush=True)
print(f"  {'arch':<14}{'Esat%':>8}{'trained%':>11}{'zerobias%':>12}{'nonzerobias%':>14}", flush=True)
rows = [["arch", "esat_pct", "trained_pct", "zerobias_pct", "nonzerobias_pct"]]
for arch in ARCHS:
    out = [arch, ESAT[arch]]
    for arm in ("trained", "zerobias", "nonzerobias"):
        base, cur = med[f"{arm}-dinov3splus"], med[f"{arm}-{arch}"]
        out.append(round(100 * (cur - base) / base, 2))
    print(f"  {out[0]:<14}{out[1]:>8}{out[2]:>11}{out[3]:>12}{out[4]:>14}", flush=True)
    rows.append(out)

ratio_path = OUT.with_name("t4_rca_arms_ratios.csv")
with ratio_path.open("w", newline="") as f:
    csv.writer(f).writerows(rows)
print(f"\nwrote {ratio_path}", flush=True)
