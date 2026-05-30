# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Re-benchmark Stereo 3D (s3d) models with the corrected KITTI metrics.

Vals one or more trained ``*-s3d`` checkpoints on the KITTI Stereo validation
split and prints docs-ready Markdown tables for AP3D, AP_BEV and AOS at the
Moderate difficulty (the KITTI headline), at IoU 0.5 and 0.7.

This regenerates the numbers in ``docs/en/tasks/s3d.md`` after the 3D IoU was
fixed from an axis-aligned approximation to true rotated IoU (those old numbers
were inflated and not leaderboard-comparable).

Example (run on the GPU box):
    python -m ultralytics.data.scripts.benchmark_s3d \
        --weights yolo26n-s3d.pt yolo26s-s3d.pt yolo26m-s3d.pt yolo26l-s3d.pt yolo26x-s3d.pt \
        --data kitti-stereo.yaml --imgsz 384 1248 --device 0 --out s3d_benchmark.md

Notes:
- Pass checkpoint paths (or names resolvable by YOLO) via --weights.
- The "Car (Mod)" columns are the KITTI headline; the "mean (Mod)" columns
  average over the model's real classes (Aux_ pseudo-classes excluded), matching
  the validator's ap3d_*/apbev_*/aos_* summaries.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def _param_count(model: YOLO) -> float:
    """Return parameter count in millions."""
    return sum(p.numel() for p in model.model.parameters()) / 1e6


def _get(rd: dict, *keys: str) -> float | None:
    """Return the first present key from results_dict, else None."""
    for k in keys:
        if k in rd:
            return float(rd[k])
    return None


def _fmt(v: float | None) -> str:
    """Format a metric as a percentage string, or '-' when absent."""
    return f"{v * 100:.1f}" if v is not None else "-"


def benchmark(weights: list[str], data: str, imgsz: list[int], batch: int, device: str) -> list[dict]:
    """Val each checkpoint and collect its results_dict plus parameter count."""
    rows = []
    for w in weights:
        model = YOLO(w)
        metrics = model.val(data=data, imgsz=imgsz, batch=batch, device=device, plots=False, verbose=False)
        rd = metrics.results_dict
        rows.append({"name": Path(w).stem, "params": _param_count(model), "rd": rd})
    return rows


def render_tables(rows: list[dict]) -> str:
    """Render the docs-ready Markdown tables from collected results."""
    out = []

    # Headline table (mirrors docs/en/tasks/s3d.md), now true rotated IoU.
    out.append("### AP3D (KITTI R40, Moderate) — true rotated 3D IoU\n")
    out.append("| Model | Params | AP3D@0.5 (Car) | AP3D@0.7 (Car) | AP3D@0.5 (mean) | AP3D@0.7 (mean) |")
    out.append("|-------|--------|----------------|----------------|-----------------|-----------------|")
    for r in rows:
        rd = r["rd"]
        out.append(
            f"| {r['name']} | {r['params']:.1f}M "
            f"| {_fmt(_get(rd, 'AP3D_Car_Mod_50'))} | {_fmt(_get(rd, 'AP3D_Car_Mod_70'))} "
            f"| {_fmt(_get(rd, 'ap3d_50'))} | {_fmt(_get(rd, 'ap3d_70'))} |"
        )

    # Extended table: BEV and orientation (AOS) at Moderate.
    out.append("\n### AP_BEV and AOS (KITTI R40, Moderate)\n")
    out.append("| Model | APBEV@0.5 (Car) | APBEV@0.7 (Car) | AOS@0.5 (Car) | AOS@0.7 (Car) | AOS@0.7 (mean) |")
    out.append("|-------|-----------------|-----------------|---------------|---------------|----------------|")
    for r in rows:
        rd = r["rd"]
        out.append(
            f"| {r['name']} "
            f"| {_fmt(_get(rd, 'APBEV_Car_Mod_50'))} | {_fmt(_get(rd, 'APBEV_Car_Mod_70'))} "
            f"| {_fmt(_get(rd, 'AOS_Car_Mod_50'))} | {_fmt(_get(rd, 'AOS_Car_Mod_70'))} "
            f"| {_fmt(_get(rd, 'aos_70'))} |"
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    """Parse arguments, run the benchmark and write/print the Markdown tables."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument(
        "--weights",
        nargs="+",
        default=["yolo26n-s3d.pt", "yolo26s-s3d.pt", "yolo26m-s3d.pt", "yolo26l-s3d.pt", "yolo26x-s3d.pt"],
        help="Checkpoint paths or names to validate.",
    )
    p.add_argument("--data", default="kitti-stereo.yaml", help="Dataset YAML.")
    p.add_argument("--imgsz", nargs=2, type=int, default=[384, 1248], help="Validation image size [H, W].")
    p.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    p.add_argument("--device", default="0", help="CUDA device, e.g. '0' or 'cpu'.")
    p.add_argument("--out", default="s3d_benchmark.md", help="Output Markdown file.")
    args = p.parse_args()

    rows = benchmark(args.weights, args.data, args.imgsz, args.batch, args.device)
    tables = render_tables(rows)
    Path(args.out).write_text(tables)
    print(tables)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
