# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""T4 TensorRT latency for one lane at one scale, measured in a single paired session.

Run as ``python bench/t4_bench.py <lane>-<scale> [arm,arm] [warmup=N]``, e.g. `lane-a-x` or `lane-b-n repmixer`.
Given no arm list a session profiles that scale's baseline, its bridge and every arm that exists there. Only
TensorRT is timed, the other formats run once afterwards into a `.formats.csv` sidecar, see t4_bench_common for
why. `warmup=N` overrides the standard conditioning and marks the session exploratory, see run_benchmark.

The bridge is the arm carried by every session at a scale. Lane A uses its YOLO26 Conv baseline itself, so every
candidate transfers through its same-session ratio to Conv. Lane B uses ffnattn2 because its baseline family changes
across scales. A second argument, a comma list of arm tags, times only those arms and the required anchors.

Scale comes from the filename, and a scale-less stem silently resolves to the first scales key, so every entry has
its size letter substituted in. Arms absent at a scale are simply not in that session.

Every arm in a session runs at the baseline's deployed input size, from `IMGSZ` below, since a paired comparison
only holds at one size. Lane A deploys at 640 throughout, Lane B does not.
"""

import sys
from pathlib import Path

# Checkout root first, so ultralytics, working_dir and t4_bench_common all come from this commit. sys.path[0] is
# bench/, not the root. Ruff permits a sys.path mutation ahead of imports, so this needs no E402 suppression,
# which the formatter strips anyway.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from t4_bench_common import build_variant, parse_session, pinned_fp32_attn, run_benchmark
from ultralytics import RTDETR, YOLO

DATA = Path("/root/autodl-tmp/data")

# The size each lane's baseline deploys at, per scale, defaulting to 640. yolo27-detr trains n at 480 and s and m
# at 512, so timing those at 640 measures an operating point nobody ships.
IMGSZ = {("lane-b", "n"): 480, ("lane-b", "s"): 512, ("lane-b", "m"): 512}

# lane -> (facade, engine builder, baseline tag prefix, bridge tag, {arm: (yaml template, scales)}).
#
# Lane A exports stock, Lane B with the fp32 attention pin DINOv3 needs to survive fp16, so only within-lane ratios
# travel.
#
LANES = {
    "lane-a": (
        YOLO,
        None,
        "conv",
        "conv",
        {
            "conv": ("yolo26{s}.yaml", "nsmlx"),
            "attn2": ("yolo26{s}-ultravit-attn2.yaml", "nsmlx"),
            "base": ("yolo26{s}-ultravit.yaml", "nsmlx"),
            "dinoreg": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg.yaml", "smlx"),
            "dinorope": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-dinorope.yaml", "smlx"),
            "fastvitffn": ("yolo26{s}-ultravit-repmixer-fastvitffn.yaml", "nsmlx"),
            "ffnattn2": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2.yaml", "nsmlx"),
            "mixedrope": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-mixedrope.yaml", "smlx"),
            "p4pooled": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4pooled.yaml", "nsmlx"),
            "p4win": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4win.yaml", "x"),
            "repmixer": ("yolo26{s}-ultravit-repmixer.yaml", "nsmlx"),
            "s480": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-s480.yaml", "s"),
            "tokenmlp-stagefloor": ("yolo26{s}-ultravit-290726.yaml", "nl"),
            "tokenmlp-stagebalance": ("yolo26{s}-ultravit-290726.yaml", "s"),
            "tokenmlp-p2wide": ("yolo26{s}-ultravit-290726.yaml", "x"),
            "ultravit-010826-2a": ("yolo26{s}-ultravit-010826-2a.yaml", "s"),
            "ultravit-010826-2b": ("yolo26{s}-ultravit-010826-2b.yaml", "s"),
            "ultravit-010826-1": ("yolo26{s}-ultravit-010826-1.yaml", "l"),
            "ultravit-010826-2": ("yolo26{s}-ultravit-010826-2.yaml", "l"),
            "attn2-090826-1": ("yolo26{s}-ultravit-attn2-090826-1.yaml", "l"),
            "attn2-090826-2": ("yolo26{s}-ultravit-attn2-090826-2.yaml", "l"),
            "attn2-090826-3": ("yolo26{s}-ultravit-attn2-090826-3.yaml", "l"),
            "attn2-090826-4": ("yolo26{s}-ultravit-attn2-090826-4.yaml", "l"),
            "ultravit-020826": ("yolo26{s}-ultravit-020826.yaml", "s"),
            "ultravit-020826-1": ("yolo26{s}-ultravit-020826-1.yaml", "ns"),
            "ultravit-020826-2": ("yolo26{s}-ultravit-020826-2.yaml", "ns"),
            "ultravit-020826-3": ("yolo26{s}-ultravit-020826-3.yaml", "n"),
            "ultravit-040826-1": ("yolo26{s}-ultravit-040826-1.yaml", "nsm"),
            "ultravit-040826-2": ("yolo26{s}-ultravit-040826-2.yaml", "s"),
            "ultravit-060826-1": ("yolo26{s}-ultravit-060826-1.yaml", "n"),
            "deep16-sni-uvit-290726": ("yolo26{s}-p4p5-wide-deep16-sni-ultravit.yaml", "sm"),
            "slim19-uvit-290726": ("yolo26{s}-p4p5-wide-slim19-ultravit.yaml", "sm"),
            "deep16-sni-uvit-020826-1": ("yolo26{s}-p4p5-wide-deep16-sni-ultravit-020826-1.yaml", "s"),
            "slim19-uvit-020826-1": ("yolo26{s}-p4p5-wide-slim19-ultravit-020826-1.yaml", "s"),
            "deep16-sni-uvit-040826-1": ("yolo26{s}-p4p5-wide-deep16-sni-ultravit-040826-1.yaml", "s"),
            "slim19-uvit-040826-1": ("yolo26{s}-p4p5-wide-slim19-ultravit-040826-1.yaml", "s"),
            "deep16-sni-conv": ("yolo26{s}-p4p5-wide-deep16-sni.yaml", "ns"),
            "slim19-conv": ("yolo26{s}-p4p5-wide-slim19.yaml", "s"),
            "y27-p3-170826-1": ("yolo27{s}-p3-170826-1.yaml", "ns"),
            "y27-p3-170826-2": ("yolo27{s}-p3-170826-2.yaml", "ns"),
            "y27-p2lite-190826-1": ("yolo27{s}-p2lite-190826-1.yaml", "ns"),
            "y27-p3-repcib-190826-2": ("yolo27{s}-p3-repcib-190826-2.yaml", "ns"),
            "y27-p2wide-190826-3": ("yolo27{s}-p2wide-190826-3.yaml", "n"),
            "y27-p2wide-repcib-190826-4": ("yolo27{s}-p2wide-repcib-190826-4.yaml", "n"),
            "y27-ultravit-170826-1": ("yolo27{s}-ultravit-170826-1.yaml", "ns"),
            "y27-ultravit-170826-2": ("yolo27{s}-ultravit-170826-2.yaml", "ns"),
            "y27-ultravit-170826-3": ("yolo27{s}-ultravit-170826-3.yaml", "ns"),
            "y27-ultravit-170826-4": ("yolo27{s}-ultravit-170826-4.yaml", "ns"),
        },
    ),
    "lane-b": (
        RTDETR,
        pinned_fp32_attn,
        "yolo27",
        "ffnattn2",
        {
            # The detr_decoder_clean2 reference arms. The family changes along the ladder: CSP trunk with
            # RTDETRDecoderEfficient at n and s, DeimDecoder at m and l, plain ViT trunk at x.
            "yolo27": ("yolo27{s}-detr.yaml", "ns"),
            "yolo27-deim": ("yolo27{s}-deim-detr.yaml", "ml"),
            "yolo27-vit": ("yolo27{s}-vit-detr.yaml", "x"),
            "dinov3splus": ("deim_dinov3splus_sta_l6_xl.yaml", "x"),
            "attn2": ("yolo26{s}-ultravit-attn2-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "base": ("yolo26{s}-ultravit-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "dinop5": ("yolo26{s}-ultravit-repmixer-dinop5-deim_mal_deimv2Neck.yaml", "smlx"),
            "dinoreg": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-deim_mal_deimv2Neck.yaml", "x"),
            "dinorope": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-dinorope-deim_mal_deimv2Neck.yaml",
                "x",
            ),
            "fastvitffn": ("yolo26{s}-ultravit-repmixer-fastvitffn-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "fastvitffn-dinop5": ("yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-deim_mal_deimv2Neck.yaml", "smlx"),
            "fastvitffn-dinop5-mixedrope": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-mixedrope-deim_mal_deimv2Neck.yaml",
                "smlx",
            ),
            "fastvitffn-dinop5-depthmatched": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-dinop5-depthmatched-deim_mal_deimv2Neck.yaml",
                "x",
            ),
            "ffnattn2": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "mixedrope": (
                "yolo26{s}-ultravit-repmixer-fastvitffn-attn2-dinoreg-mixedrope-deim_mal_deimv2Neck.yaml",
                "x",
            ),
            "p4deep": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4deep-deim_mal_deimv2Neck.yaml", "x"),
            "p4pooled": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4pooled-deim_mal_deimv2Neck.yaml", "x"),
            "p4win": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-p4win-deim_mal_deimv2Neck.yaml", "x"),
            "repmixer": ("yolo26{s}-ultravit-repmixer-deim_mal_deimv2Neck.yaml", "nsmlx"),
            "s480": ("yolo26{s}-ultravit-repmixer-fastvitffn-attn2-s480-deim_mal_deimv2Neck.yaml", "s"),
        },
    ),
}

lane, scale, session, only, warmup = parse_session(sys.argv[1:])
model_cls, engine_builder, base_prefix, bridge, arms = LANES[lane]
yamls = {tag: t.format(s=scale) for tag, (t, scales) in arms.items() if scale in scales}
# Lane B's baseline changes file along the ladder, so it is whichever arm carries the baseline prefix at this scale.
baseline = next(tag for tag in yamls if tag.startswith(base_prefix))
assert bridge in yamls, f"{session} has no bridge arm, so its ratios cannot be anchored to another session"
if only:  # append session, carrying the named arms plus the baseline and bridge that anchor them
    (arg,) = only  # unpack, so a stray extra argument fails here rather than being ignored
    want = set(arg.split(","))
    assert want <= set(yamls), f"{sorted(want - set(yamls))} absent from {session}, which would bench anchors alone"
    yamls = {t: y for t, y in yamls.items() if t in want | {baseline, bridge}}
    session += "-" + arg  # its own session id, since it is its own measurement occasion

imgsz = IMGSZ.get((lane, scale), 640)
engines = DATA / f"t4-{lane}-{scale}-engines"
variants = [
    build_variant(t, y, engines, imgsz, model_cls=model_cls, engine_builder=engine_builder) for t, y in yamls.items()
]
run_benchmark(
    variants,
    dict.fromkeys(yamls, baseline),
    DATA / f"t4_{session.replace('-', '_')}.csv",
    session,
    imgsz,
    warmup=warmup,
)
