"""Stage 6 of h8 — validate the s5_decision hypothesis with 3 seeds on seetacloud.

Reads:  to_human/s5_decision.md (must contain a `diff` fenced block applying to the champion yaml)
Writes: to_human/EXPERIMENT.md
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
OUT = ROOT / "to_human"

# Frozen pass criterion per spec.
PASS_CRITERION = "median(new R1) > median(champion R1)"


def _extract_recipe_diff(decision_md: Path) -> str:
    text = decision_md.read_text()
    m = re.search(r"```diff\n(.*?)\n```", text, re.DOTALL)
    if not m:
        raise SystemExit("No `diff` fenced block in s5_decision.md — fill it in before running Stage 6.")
    return m.group(1)


def _apply_diff(diff_text: str, target_yaml: Path) -> Path:
    """Apply diff_text to target_yaml -> write to a sibling .h8-variant.yaml file."""
    variant = target_yaml.with_name(target_yaml.stem + ".h8-variant.yaml")
    shutil.copy(target_yaml, variant)
    proc = subprocess.run(["patch", str(variant), "-i", "-"], input=diff_text, text=True, capture_output=True)
    if proc.returncode != 0:
        raise SystemExit(f"patch failed:\n{proc.stdout}\n{proc.stderr}")
    return variant


def _run_seed(variant_yaml: Path, seed: int, gpu: int, base_args: dict) -> dict:
    """Launch one training seed on `gpu`. Returns metrics dict on completion."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cmd = [
        "python", "-c",
        f"""
from ultralytics import YOLO
m = YOLO('{variant_yaml}', task='reid')
m.train(seed={seed}, **{base_args!r})
res = m.val(reid_tta=True, reid_reranking=True)
print('SEED_RESULT', {{ 'seed': {seed}, 'r1': float(res.rank1), 'mAP': float(res.mAP) }})
"""
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        return {"seed": seed, "status": "crashed", "stderr_tail": proc.stderr[-2000:]}
    m = re.search(r"SEED_RESULT (\{.*\})", proc.stdout)
    if not m:
        return {"seed": seed, "status": "no_result", "stdout_tail": proc.stdout[-2000:]}
    return eval(m.group(1)) | {"status": "ok"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--champion-yaml", required=True, help="path to champion config yaml (the diff base)")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/Market-1501.yaml")
    parser.add_argument("--epochs", type=int, default=635)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--seeds", type=int, nargs=3, default=[0, 1, 2])
    parser.add_argument("--gpus", type=int, nargs=4, default=[0, 1, 2, 3])
    parser.add_argument("--champion-baseline-only", action="store_true",
                        help="just run the env-drift baseline and exit (Step 2 below)")
    args = parser.parse_args()

    decision = OUT / "s5_decision.md"
    if not decision.exists():
        raise SystemExit("Run Stage 5 and edit s5_decision.md first.")
    diff_text = _extract_recipe_diff(decision)

    base_args = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    # 1) Env-drift baseline on the 4th GPU.
    print(">>> env-drift baseline (champion, 1 seed) on GPU", args.gpus[3])
    baseline = _run_seed(Path(args.champion_yaml), seed=args.seeds[0], gpu=args.gpus[3], base_args=base_args)
    if baseline.get("status") != "ok":
        raise SystemExit(f"env-drift baseline failed: {baseline}")
    if not (0.925 <= baseline["r1"] <= 0.928):
        raise SystemExit(f"env drift: baseline R1={baseline['r1']:.4f} not in [0.925, 0.928]")
    print(f"    baseline R1={baseline['r1']:.4f} ✓")
    if args.champion_baseline_only:
        return

    # 2) Apply diff → variant yaml.
    variant = _apply_diff(diff_text, Path(args.champion_yaml))
    print(f">>> applied diff to {variant}")

    # 3) 3 seeds in parallel on GPUs 0,1,2.
    import concurrent.futures as cf
    results = []
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_run_seed, variant, seed, args.gpus[i], base_args): seed
                   for i, seed in enumerate(args.seeds)}
        for fut in cf.as_completed(futures):
            results.append(fut.result())

    r1s = [r["r1"] for r in results if r.get("status") == "ok"]
    if not r1s:
        raise SystemExit(f"all seeds failed: {results}")
    median_new = float(np.median(r1s))
    verdict = "confirmed" if median_new > baseline["r1"] else "null"

    # 4) Write EXPERIMENT.md.
    lines = [
        "# h8 Validation Experiment\n",
        f"\n## Pass criterion (frozen pre-run)\n\n`{PASS_CRITERION}`\n",
        f"\n## Env-drift baseline\n\nchampion @ seed={args.seeds[0]} on this seetacloud image: R1={baseline['r1']:.4f}, mAP={baseline['mAP']:.4f}\n",
        f"\n## Recipe diff\n\n```diff\n{diff_text}\n```\n",
        "\n## Per-seed results\n\n",
        pd.DataFrame(results).to_markdown(index=False),
        f"\n\n## Median delta\n\nmedian(new R1) = {median_new:.4f}  vs  baseline R1 = {baseline['r1']:.4f}  →  Δ = {median_new - baseline['r1']:+.4f}\n",
        f"\n## Verdict\n\n**{verdict.upper()}**\n",
    ]
    (OUT / "EXPERIMENT.md").write_text("\n".join(lines))
    print(f"verdict = {verdict}, median R1 = {median_new:.4f}, baseline = {baseline['r1']:.4f}")


if __name__ == "__main__":
    main()
