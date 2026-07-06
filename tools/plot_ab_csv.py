#!/usr/bin/env python
"""Generate per-category bar charts from a multi-method comparison CSV.

Reads a CSV with columns like ``none_mAP10``, ``heatmap_mAP10``,
``mask_mAP10``, etc. and produces one figure per category + one averages
figure + a grid.  Supports 1–N methods (e.g.  none-only,  none+heatmap,
none+heatmap+mask).

Usage:
  # Standalone
  python tools/plot_ab_csv.py runs/temp/yoloa/.../val_none,heatmap_all.csv
  python tools/plot_ab_csv.py val.csv --metrics mAP10,mAP50 --out my_plots/

  # From run_yoloa.py
  from tools.plot_ab_csv import plot_ab_from_csv
  plot_ab_from_csv(csv_path, out_dir)
"""

import argparse
import csv
import math
import os

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Constants
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_METRICS = None  # means auto-detect from CSV
_METRIC_ORDER = ("image_auroc", "pixel_auroc", "mAP10", "mAP25", "mAP50", "mAP10_50", "P", "R")

# Colours for up to 6 methods (enough for none / heatmap / mask / segment / ...).
# Light-to-dark progression; if more than 6, matplotlib cycles.
_METHOD_COLORS = [
    "#90caf9",  # light blue
    "#42a5f5",  # medium blue
    "#1565c0",  # dark blue
    "#7e57c2",  # purple
    "#e57373",  # red
    "#66bb6a",  # green
]


# ──────────────────────────────────────────────────────────────────────────────
#  CSV reader
# ──────────────────────────────────────────────────────────────────────────────

def _read_csv(csv_path):
    """Parse a multi-method comparison CSV.

    Returns:
        categories:      list of category name strings (no AVERAGE)
        methods:         list of method prefix strings, in column order
        available_metrics: list of metric suffix strings
        raw_rows:        all CSV rows as dicts
        avg_row:         the AVERAGE row (or None)
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        raw_rows = list(reader)

    # Detect methods and metrics from column names
    methods = []
    suffixes = set()
    for fn in fieldnames:
        if fn == "category":
            continue
        parts = fn.split("_", 1)
        if len(parts) == 2:
            method, suffix = parts[0], parts[1]
            if method not in methods:
                methods.append(method)
            suffixes.add(suffix)

    if not methods:
        raise ValueError(f"No method columns found in CSV. Columns: {fieldnames}")

    available_metrics = sorted(suffixes)

    categories = []
    avg_row = None
    for r in raw_rows:
        cat = r.get("category", "").strip()
        if cat == "AVERAGE":
            avg_row = r
        elif cat:
            categories.append(cat)

    return categories, methods, available_metrics, raw_rows, avg_row


# ──────────────────────────────────────────────────────────────────────────────
#  Bar helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bar_positions(n_methods, i_method, bar_w=0.3, gap=0.02):
    """Return the x-offset for each method's bar group."""
    total_w = n_methods * bar_w + (n_methods - 1) * gap
    start = -total_w / 2 + bar_w / 2
    return start + i_method * (bar_w + gap)


def _method_color(i, n):
    """Return a colour for method index *i* out of *n*."""
    if i < len(_METHOD_COLORS):
        return _METHOD_COLORS[i]
    # fallback: matplotlib default cycle
    import matplotlib as mpl
    return mpl.rcParams["axes.prop_cycle"].by_key()["color"][i % 10]


# ──────────────────────────────────────────────────────────────────────────────
#  Per-category figure
# ──────────────────────────────────────────────────────────────────────────────

def _plot_one(cat, row, methods, metrics, out_dir):
    import matplotlib.pyplot as plt

    n_methods = len(methods)
    bar_w = max(0.15, 0.55 / n_methods)

    # Gather values per method
    vals_raw = []
    vals_plot = []
    for m in methods:
        vr = [float(row.get(f"{m}_{k}", math.nan)) for k in metrics]
        vals_raw.append(vr)
        vals_plot.append([0.0 if math.isnan(v) else v for v in vr])

    fw = max(5, len(metrics) * 1.0)
    fig, ax = plt.subplots(figsize=(fw, 4.5))
    title = f"{cat} — " + " vs ".join(methods)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    x = np.arange(len(metrics))

    for i_m, m in enumerate(methods):
        off = _bar_positions(n_methods, i_m, bar_w)
        color = _method_color(i_m, n_methods)
        ax.bar(x + off, vals_plot[i_m], bar_w, color=color, edgecolor="white", label=m)

    for i_k in range(len(metrics)):
        for i_m in range(n_methods):
            off = _bar_positions(n_methods, i_m, bar_w)
            v = vals_raw[i_m][i_k]
            vp = vals_plot[i_m][i_k]
            if math.isnan(v):
                ax.text(x[i_k] + off, 0.02, "N/A", ha="center", fontsize=6,
                        color="#999", rotation=90, va="bottom")
            else:
                ax.text(x[i_k] + off, vp / 2, f"{v:.3f}", ha="center", va="center",
                        fontsize=6.5, color="black")

    # Delta annotations (only for exactly 2 methods)
    if n_methods == 2:
        for i_k in range(len(metrics)):
            va = vals_raw[0][i_k]
            vb = vals_raw[1][i_k]
            if not math.isnan(va) and not math.isnan(vb):
                peak = max(vals_plot[0][i_k], vals_plot[1][i_k])
                delta = vb - va
                sign = "+" if delta >= 0 else ""
                ax.text(x[i_k], peak + 0.015, f"{sign}{delta:.3f}",
                        ha="center", fontsize=6.5, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, f"{cat}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  Averages figure
# ──────────────────────────────────────────────────────────────────────────────

def _plot_average(avg_row, methods, metrics, n_cats, out_dir):
    import matplotlib.pyplot as plt

    n_methods = len(methods)
    bar_w = max(0.15, 0.55 / n_methods)

    vals_raw = []
    vals_plot = []
    for m in methods:
        vr = [float(avg_row.get(f"{m}_{k}", math.nan)) for k in metrics]
        vals_raw.append(vr)
        vals_plot.append([0.0 if math.isnan(v) else v for v in vr])

    fw = max(5, len(metrics) * 1.0)
    fig, ax = plt.subplots(figsize=(fw, 4.5))
    title = f"Average ({n_cats} categories) — " + " vs ".join(methods)
    fig.suptitle(title, fontsize=12, fontweight="bold")

    x = np.arange(len(metrics))

    for i_m, m in enumerate(methods):
        off = _bar_positions(n_methods, i_m, bar_w)
        color = _method_color(i_m, n_methods)
        ax.bar(x + off, vals_plot[i_m], bar_w, color=color, edgecolor="white", label=m)

    for i_k in range(len(metrics)):
        for i_m in range(n_methods):
            off = _bar_positions(n_methods, i_m, bar_w)
            v = vals_raw[i_m][i_k]
            vp = vals_plot[i_m][i_k]
            if math.isnan(v):
                ax.text(x[i_k] + off, 0.02, "N/A", ha="center", fontsize=6,
                        color="#999", rotation=90, va="bottom")
            else:
                ax.text(x[i_k] + off, vp / 2, f"{v:.3f}", ha="center", va="center",
                        fontsize=7, color="black")

    if n_methods == 2:
        for i_k in range(len(metrics)):
            va = vals_raw[0][i_k]
            vb = vals_raw[1][i_k]
            if not math.isnan(va) and not math.isnan(vb):
                peak = max(vals_plot[0][i_k], vals_plot[1][i_k])
                delta = vb - va
                sign = "+" if delta >= 0 else ""
                ax.text(x[i_k], peak + 0.015, f"{sign}{delta:.3f}",
                        ha="center", fontsize=7, color="#666")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.10)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    path = os.path.join(out_dir, "_average.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  Grid (4×4)
# ──────────────────────────────────────────────────────────────────────────────

def _plot_grid(categories, cat_rows, avg_row, methods, metrics, out_dir):
    import matplotlib.pyplot as plt

    panels = list(categories) + ["AVERAGE"]
    n = len(panels)
    ncols = 4
    nrows = int(math.ceil(n / ncols))

    n_methods = len(methods)
    bar_w = max(0.12, 0.50 / n_methods)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 4 * nrows))
    fig.suptitle(" vs ".join(methods), fontsize=14, fontweight="bold", y=0.98)

    x = np.arange(len(metrics))

    for idx, (ax, panel_name) in enumerate(zip(axes.flat, panels)):
        if panel_name == "AVERAGE":
            r = avg_row
        else:
            r = cat_rows.get(panel_name)

        if r is None:
            ax.set_visible(False)
            continue

        vals_raw = []
        vals_plot = []
        for m in methods:
            vr = [float(r.get(f"{m}_{k}", math.nan)) for k in metrics]
            vals_raw.append(vr)
            vals_plot.append([0.0 if math.isnan(v) else v for v in vr])

        for i_m, m in enumerate(methods):
            off = _bar_positions(n_methods, i_m, bar_w)
            color = _method_color(i_m, n_methods)
            ax.bar(x + off, vals_plot[i_m], bar_w, color=color, edgecolor="white")

        for i_k in range(len(metrics)):
            for i_m in range(n_methods):
                off = _bar_positions(n_methods, i_m, bar_w)
                v = vals_raw[i_m][i_k]
                vp = vals_plot[i_m][i_k]
                if not math.isnan(v):
                    ax.text(x[i_k] + off, vp / 2, f"{v:.2f}", ha="center", va="center",
                            fontsize=5.5, color="black")

        # Delta (2 methods only)
        if n_methods == 2:
            for i_k in range(len(metrics)):
                va = vals_raw[0][i_k]
                vb = vals_raw[1][i_k]
                if not math.isnan(va) and not math.isnan(vb):
                    peak = max(vals_plot[0][i_k], vals_plot[1][i_k])
                    delta = vb - va
                    sign = "+" if delta >= 0 else ""
                    ax.text(x[i_k], peak + 0.03, f"{sign}{delta:.2f}",
                            ha="center", fontsize=5, color="#666")

        title = "Average" if panel_name == "AVERAGE" else panel_name
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, fontsize=6)
        ax.set_ylim(0, 1.15)
        ax.tick_params(axis="y", labelsize=6)

    # Turn off unused subplots
    for idx in range(n, nrows * ncols):
        axes.flat[idx].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, fc=_method_color(i, n_methods), edgecolor="white")
               for i in range(n_methods)]
    fig.legend(handles, methods, loc="upper right",
               fontsize=10, framealpha=0.95, bbox_to_anchor=(0.99, 0.96))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(out_dir, "_grid.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    return path


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def plot_ab_from_csv(csv_path, out_dir=None, metrics=DEFAULT_METRICS):
    """Read a multi-method comparison CSV and generate per-category + average + grid figures.

    Args:
        csv_path: Path to the CSV file.
        out_dir:  Output directory (default: ``<csv_dir>/plots/``).
        metrics:  Metric suffixes to plot, or None to auto-detect all.

    Returns:
        List of generated PNG paths.
    """
    categories, methods, available_metrics, rows, avg_row = _read_csv(csv_path)

    # --- resolve metrics ---
    if metrics is None:
        metrics = tuple(m for m in _METRIC_ORDER if m in available_metrics)
        for m in available_metrics:
            if m not in metrics:
                metrics = metrics + (m,)
        if not metrics:
            raise ValueError(f"No usable metrics in CSV. Columns: {available_metrics}")

    for m in metrics:
        if m not in available_metrics:
            raise ValueError(f"Metric '{m}' not found in CSV. Available: {available_metrics}")

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(csv_path), "plots")
    os.makedirs(out_dir, exist_ok=True)

    cat_rows = {r["category"].strip(): r for r in rows}

    paths = []
    for cat in categories:
        r = cat_rows.get(cat)
        if r is None:
            print(f"  SKIP {cat}: not in CSV")
            continue
        p = _plot_one(cat, r, methods, metrics, out_dir)
        paths.append(p)
        print(f"  {cat}")

    if avg_row:
        p = _plot_average(avg_row, methods, metrics, len(categories), out_dir)
        paths.append(p)
        print(f"  AVERAGE ({len(categories)} cats)")

    p = _plot_grid(categories, cat_rows, avg_row, methods, metrics, out_dir)
    paths.append(p)
    print(f"  GRID ({len(categories) + 1} panels)")

    print(f"\n{len(paths)} figures -> {out_dir}/")
    return paths


# ──────────────────────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate per-category bar charts from a multi-method comparison CSV.")
    ap.add_argument("csv", help="Path to the comparison CSV")
    ap.add_argument("--metrics", default=None,
                    help="Comma-separated metric suffixes (default: all metrics in CSV)")
    ap.add_argument("--out", default=None, help="Output directory (default: <csv_dir>/plots/)")
    args = ap.parse_args()

    metrics = None
    if args.metrics is not None:
        metrics = tuple(m.strip() for m in args.metrics.split(","))
    plot_ab_from_csv(args.csv, out_dir=args.out, metrics=metrics)


if __name__ == "__main__":
    main()
