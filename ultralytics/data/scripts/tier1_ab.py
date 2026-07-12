# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Tier-1 s3d A/B harness: projected-center + depth-uncertainty.

Runs 4 training arms (T0 baseline, Tc +center, Tsigma +uncertainty, Tcsigma +both) from the
base yolo26-s3d.yaml with train-time flags injected, then evaluates 8 decode arms (A0..A7)
by toggling the decode knobs (use_proj_center / ivw_fusion / score_weight) at val time, and
prints the KITTI Car AP3D table vs the baseline.

The 4 arm configs are generated at runtime from the base YAML (no committed duplication).

Usage (on the weste box, 8 GPUs):
    python -m ultralytics.data.scripts.tier1_ab --data kitti-stereo.yaml --epochs 200 --batch 32 \
        --devices 0,1,2,3 --project /root/autodl-tmp/s3d/runs/tier1_ab
Smoke (2 epochs on the mini dataset):
    python -m ultralytics.data.scripts.tier1_ab --data kitti-stereo8.yaml --epochs 2 --batch 2 \
        --devices 0 --project /tmp/tier1_smoke
Add --skip-train to only run the eval matrix over existing checkpoints.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml

from ultralytics import YOLO
from ultralytics.utils import LOGGER, ROOT

# Training arms: name -> (use_proj_center, use_depth_uncertainty)
TRAIN_ARMS = {
    "T0": (False, False),
    "Tc": (True, False),
    "Tsigma": (False, True),
    "Tcsigma": (True, True),
}

# Eval arms: name -> (checkpoint arm, use_proj_center, ivw_fusion, score_weight)
EVAL_ARMS = [
    ("A0_baseline", "T0", False, False, False),
    ("A1_center", "Tc", True, False, False),
    ("A2_fusion", "Tsigma", False, True, False),
    ("A3_score", "Tsigma", False, False, True),
    ("A4_fusion_score", "Tsigma", False, True, True),
    ("A5_full", "Tcsigma", True, True, True),
    ("A6_center_fusion", "Tcsigma", True, True, False),
    ("A7_center_score", "Tcsigma", True, False, True),
]


def gen_configs(cfg_dir: Path) -> dict[str, Path]:
    """Generate the 4 arm model-configs from the base yolo26-s3d.yaml."""
    base = yaml.safe_load((ROOT / "cfg/models/26/yolo26-s3d.yaml").read_text())
    cfg_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (proj, unc) in TRAIN_ARMS.items():
        cfg = yaml.safe_load(yaml.safe_dump(base))  # deep copy
        tr = cfg.setdefault("training", {})
        tr["use_proj_center"] = proj
        tr["use_depth_uncertainty"] = unc
        if proj:
            tr.setdefault("loss_weights", {})["proj_center"] = 1.0
        # filename carries the scale letter 'n' so guess_model_scale picks it up
        out = cfg_dir / f"yolo26n-s3d-{name}.yaml"
        out.write_text(yaml.safe_dump(cfg, sort_keys=False))
        paths[name] = out
    return paths


def train_arms(configs: dict[str, Path], data: str, epochs: int, batch: int, imgsz: str, devices: list[str], project: str):
    """Train each arm as a parallel subprocess, one per device (round-robin if fewer devices).

    imgsz is passed as the raw CLI string (e.g. "384,1248"); the yolo CLI parses it to the
    rectangular (H, W) tuple. This MUST match the eval imgsz — a train/eval imgsz mismatch
    silently zeroes every AP3D.
    """
    procs = []
    for i, (name, cfg) in enumerate(configs.items()):
        dev = devices[i % len(devices)]
        cmd = ["yolo", "train", "task=s3d", f"model={cfg}", f"data={data}", f"epochs={epochs}",
               f"batch={batch}", f"imgsz={imgsz}", "val=False", "amp=False", f"device={dev}",
               f"project={project}", f"name={name}"]
        LOGGER.info(f"[tier1] launching train {name} on device {dev}")
        procs.append((name, subprocess.Popen(cmd)))
    failed = [name for name, p in procs if p.wait() != 0]
    if failed:
        raise RuntimeError(f"training arms failed: {failed}")


def eval_matrix(project: str, data: str, batch: int, imgsz, device: str) -> list[dict]:
    """Run the 8 eval arms; each re-runs inference (never reuse a preds dict — NMS mutates it)."""
    rows = []
    for name, ckpt_arm, proj, ivw, score in EVAL_ARMS:
        ckpt = Path(project) / ckpt_arm / "weights" / "best.pt"
        if not ckpt.exists():
            LOGGER.warning(f"[tier1] missing checkpoint {ckpt}; skipping {name}")
            continue
        m = YOLO(str(ckpt))
        res = m.val(
            task="s3d", data=data, batch=batch, imgsz=imgsz, device=device,
            project=project, name=f"eval/{name}",
            use_proj_center=proj, ivw_fusion=ivw, score_weight=score,
        )
        rd = getattr(res, "results_dict", {}) or {}
        rows.append({"arm": name, "ckpt": ckpt_arm, "proj": proj, "ivw": ivw, "score": score, **rd})
    return rows


# Summary metrics to tabulate (the primary A/B signal is ap3d_70 = Car Moderate AP3D@0.7).
SUMMARY_KEYS = ["ap3d_50", "ap3d_70", "apbev_50", "apbev_70", "aos_50", "aos_70", "fitness"]


def print_table(rows: list[dict]):
    """Print the collected AP3D summary table (primary signal: ap3d_70)."""
    if not rows:
        LOGGER.warning("[tier1] no eval rows collected")
        return
    keys = [k for k in SUMMARY_KEYS if k in rows[0]] or [
        k for k in rows[0] if k not in ("arm", "ckpt", "proj", "ivw", "score")
    ]
    header = f"{'arm':<18}{'ckpt':<9}{'proj':<6}{'ivw':<6}{'score':<6}" + "".join(f"{k:>12}" for k in keys)
    LOGGER.info(header)
    for r in rows:
        line = f"{r['arm']:<18}{r['ckpt']:<9}{str(r['proj']):<6}{str(r['ivw']):<6}{str(r['score']):<6}"
        line += "".join(f"{r.get(k, float('nan')):>12.4f}" for k in keys)
        LOGGER.info(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="kitti-stereo.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--devices", default="0", help="comma-separated GPU ids for parallel training")
    ap.add_argument("--imgsz", default="384,1248", help="aspect-preserving letterbox (H,W) for KITTI")
    ap.add_argument("--project", default="runs/tier1_ab")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    devices = [d.strip() for d in str(args.devices).split(",") if d.strip() != ""]
    imgsz = [int(x) for x in str(args.imgsz).split(",")] if "," in str(args.imgsz) else int(args.imgsz)

    configs = gen_configs(Path(args.project) / "configs")
    if not args.skip_train:
        train_arms(configs, args.data, args.epochs, args.batch, str(args.imgsz), devices, args.project)
    rows = eval_matrix(args.project, args.data, args.batch, imgsz, devices[0])
    print_table(rows)


if __name__ == "__main__":
    main()
