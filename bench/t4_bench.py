# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 latency for one named suite of architectures, measured in a single paired session.

Run as ``python bench/t4_bench.py <suite>``. Every suite shares the protocol in t4_bench_common, which reads the
predictor's speed["inference"] exactly as profile_depth.py and ProfileModels do, so a new suite is a dict entry
rather than another runner. Suites naming yamls that this checkout does not carry simply are not run from it.

Scale comes from the filename, and a scale-less stem silently resolves to the first scales key, so every entry
carries an explicit size letter. The two-letter xxl needs the widened scale regex in tasks.py to resolve at all.

`delta_vs_base_pct` is an architecture comparison only between rows at the baseline's scale. Rows at another scale
are in the same suite to share its thermal state, and their delta against it is scale, not architecture.
"""

import sys
from pathlib import Path

# Checkout root first, so ultralytics, working_dir and t4_bench_common all come from this commit. sys.path[0] is
# bench/, not the root. Ruff permits a sys.path mutation ahead of imports, so this needs no E402 suppression,
# which the formatter strips anyway.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t4_bench_common import build_variant, pinned_fp32_attn, run_benchmark
from ultralytics import RTDETR, YOLO

DATA = Path("/root/autodl-tmp/data")

# Lane A is the YOLO26 conv-head detector, so the YOLO facade resolves the task and export is stock Ultralytics,
# without Lane B's fp32 attention pin. That is a choice rather than a non-issue since six of these carry MHSA. It
# matches how the detector ships and keeps these rows continuous with earlier Lane A ones, so only within-lane
# ratios travel. Lane B is RTDETR-family and takes the pin, without which DINOv3 attention overflows fp16.
SUITES = {
    # Nine of these eleven cover every phase2 run in the orc db, one per distinct (yaml, scale). The registered
    # numbers they replace were produced across many sessions under a timer and a weight state both now known to be
    # wrong, and two architectures carry two conflicting values each. Depth-matched dinop5 and p4win are new rows.
    "lane-a": (
        "conv-x",
        YOLO,
        None,
        {
            "conv-s": "yolo26s.yaml",
            "uvit-s-attn2-s480": "yolo26s-ultravit-repmixer-fastvitffn-attn2-s480.yaml",
            "uvit-l-attn2": "yolo26l-ultravit-repmixer-fastvitffn-attn2.yaml",
            "conv-x": "yolo26x.yaml",
            "uvit-x-plain": "yolo26x-ultravit.yaml",
            "uvit-x-dinop5": "yolo26x-ultravit-repmixer-fastvitffn-dinop5.yaml",
            "uvit-x-dinop5-depthmatched": "yolo26x-ultravit-repmixer-fastvitffn-dinop5-depthmatched.yaml",
            "uvit-x-attn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2.yaml",
            "uvit-x-attn2-dinoreg": "yolo26x-ultravit-repmixer-fastvitffn-attn2-dinoreg.yaml",
            "uvit-x-attn2-p4pooled": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4pooled.yaml",
            "uvit-x-attn2-p4win": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4win.yaml",
        },
    ),
    # The six yolo27-detr scales from detr_decoder_clean2, the reference arms every Lane B detector row is compared
    # against at its own scale. The family changes across the ladder: n and s carry RTDETRDecoderEfficient on a CSP
    # trunk, m and l swap in DeimDecoder, x and xxl swap the trunk for a plain ViT.
    "laneb-baselines": (
        "yolo27x",
        RTDETR,
        pinned_fp32_attn,
        {
            "yolo27n": "yolo27n-detr.yaml",
            "yolo27s": "yolo27s-detr.yaml",
            "yolo27m": "yolo27m-deim-detr.yaml",
            "yolo27l": "yolo27l-deim-detr.yaml",
            "yolo27x": "yolo27x-vit-detr.yaml",
            "yolo27xxl": "yolo27xxl-vit-detr.yaml",
        },
    ),
    # Five UltraViT backbones on the same DEIMv2 neck, against the DINOv3-S+ arm they were designed to replace.
    "laneb-x": (
        "dinov3splus",
        RTDETR,
        pinned_fp32_attn,
        {
            "dinov3splus": "deim_dinov3splus_sta_l6_xl.yaml",
            "ffnattn2": "yolo26x-ultravit-repmixer-fastvitffn-attn2-deim_mal_deimv2Neck.yaml",
            "fastvitffn-dinop5": "yolo26x-ultravit-repmixer-fastvitffn-dinop5-deim_mal_deimv2Neck.yaml",
            "repmixer-dinop5": "yolo26x-ultravit-repmixer-dinop5-deim_mal_deimv2Neck.yaml",
            "fastvitffn-dinop5-depthmatched": (
                "yolo26x-ultravit-repmixer-fastvitffn-dinop5-depthmatched-deim_mal_deimv2Neck.yaml"
            ),
            "ffnattn2-p4win": "yolo26x-ultravit-repmixer-fastvitffn-attn2-p4win-deim_mal_deimv2Neck.yaml",
        },
    ),
}

suite = sys.argv[1]
baseline, model_cls, engine_builder, yamls = SUITES[suite]
engines = DATA / f"t4-{suite}-engines"
variants = [build_variant(t, y, engines, model_cls=model_cls, engine_builder=engine_builder) for t, y in yamls.items()]
run_benchmark(variants, baseline, DATA / f"t4_{suite.replace('-', '_')}_protocol0727.csv")
