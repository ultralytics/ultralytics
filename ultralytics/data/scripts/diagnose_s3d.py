# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Network-level diagnostics for s3d checkpoints.

Runs the val split once, then computes: a MonoDLE/TIDE3D oracle-swap ladder (per-component AP headroom incl. a
score-ranking oracle), the per-component error profile with a depth bias/variance split, bootstrap AP confidence
intervals (the metric-noise floor for small val sets), DepthDFL and cost-volume distribution probes at GT centers,
a counterfactual right-image-shift stereo probe (is depth stereo-driven or a monocular shortcut?), and an optional
per-branch gradient-conflict probe. Outputs <out-dir>/report.md plus CSVs.

Example:
    python -m ultralytics.data.scripts.diagnose_s3d --weights best.pt --data cube-s3d.yaml \
        --imgsz 768 1216 --device 0 --out-dir diagnosis/cube_best
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ultralytics import YOLO
from ultralytics.models.yolo.s3d.diagnose import (
    ORACLE_COMPONENTS,
    bootstrap_ap,
    depth_bias_fit,
    dist_stats,
    error_records,
    grad_diag_step,
    oracle_ladder,
    stereo_sensitivity,
    summarize_errors,
)
from ultralytics.utils import LOGGER

LADDER_KEYS = ("ap3d_50", "ap3d_70", "apbev_50", "apbev_70")


def rank_levers(ladder: dict[str, dict[str, float]], key: str = "ap3d_70") -> list[tuple[str, float]]:
    """Sort oracle components by ΔAP over baseline (descending) — the ranked lever list."""
    base = ladder.get("baseline", {}).get(key, 0.0)
    return sorted(((c, v.get(key, 0.0) - base) for c, v in ladder.items() if c != "baseline"), key=lambda t: -t[1])


def render_report(sections: dict[str, str]) -> str:
    """Assemble named markdown sections into the final report."""
    out = ["# s3d network diagnostics\n"]
    for title, body in sections.items():
        out += [f"## {title}\n", body, ""]
    return "\n".join(out)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Render a simple markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _error_profile_section(stats, out_dir: Path) -> str:
    """Per-component error profile + depth bias/variance split + error-vs-distance buckets."""
    recs = error_records(stats)
    if not recs:
        return "_No matched predictions — nothing to profile._"
    with open(out_dir / "error_records.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    s = summarize_errors(recs)
    a, b, resid = depth_bias_fit(recs)
    rows = [
        ["matched objects", str(s["n"])],
        ["MAE x / y / z (m)", f"{s['mae_x']:.3f} / {s['mae_y']:.3f} / {s['mae_z']:.3f}"],
        ["MAE orientation (rad)", f"{s['mae_theta']:.3f}"],
        ["mean IoU3D / IoU_BEV", f"{s['mean_iou3d']:.3f} / {s['mean_ioubev']:.3f}"],
        ["frac IoU3D ≥ 0.5 / ≥ 0.7", f"{s['frac_iou3d_ge_50']:.3f} / {s['frac_iou3d_ge_70']:.3f}"],
        ["depth bias fit dz = a·z + b", f"a={a:.4f}, b={b:.3f} m, residual σ={resid:.3f} m"],
    ]
    # Error vs distance (MonoDLE): MAE_z per z_gt quartile
    z = np.array([r["z_gt"] for r in recs])
    dz = np.abs([r["dz"] for r in recs])
    qs = np.quantile(z, [0.25, 0.5, 0.75])
    buckets = ["z<q1", "q1-q2", "q2-q3", "z>q3"]
    edges = [-np.inf, *qs, np.inf]
    for name, lo, hi in zip(buckets, edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        if m.any():
            rows.append(
                [f"MAE z, {name} (z∈[{max(lo, z.min()):.1f},{min(hi, z.max()):.1f}] m)", f"{dz[m].mean():.3f} m"]
            )
    note = (
        "\nInterpretation: a significant slope `a` is a systematic depth **scale bias** (calibration/fusion bug "
        "signature); near-zero slope with large residual σ is a **resolution/matching limit**."
    )
    return _md_table(["quantity", "value"], rows) + note


def _oracle_section(stats, names) -> tuple[str, dict]:
    """One-at-a-time GT-substitution ladder + ranked levers."""
    ladder = oracle_ladder(stats, names, components=ORACLE_COMPONENTS, keys=LADDER_KEYS)
    base = ladder["baseline"]
    rows = [["baseline"] + [f"{base[k]:.4f}" for k in LADDER_KEYS] + ["—"]]
    for comp in ORACLE_COMPONENTS:
        v = ladder[comp]
        rows.append([comp] + [f"{v[k]:.4f}" for k in LADDER_KEYS] + [f"{v['ap3d_70'] - base['ap3d_70']:+.4f}"])
    table = _md_table(["oracle", *LADDER_KEYS, "ΔAP3D@.7"], rows)
    levers = rank_levers(ladder)
    ranked = "\n".join(f"{i + 1}. **{c}**: ΔAP3D@0.7 = {d:+.4f}" for i, (c, d) in enumerate(levers))
    return table + "\n\n### Ranked levers (headroom at strict IoU)\n\n" + ranked, ladder


def _bootstrap_section(stats, names, n: int) -> str:
    """Bootstrap CI section."""
    ci = bootstrap_ap(stats, names, n=n)
    rows = [[k, f"{med:.4f}", f"[{lo:.4f}, {hi:.4f}]", f"{hi - lo:.4f}"] for k, (lo, med, hi) in ci.items()]
    return (
        _md_table(["metric", "median", "95% CI", "width"], rows)
        + "\n\nA/B deltas smaller than the CI width are metric noise, not model differences."
    )


def _probe_hooks(core: torch.nn.Module) -> tuple[dict, list]:
    """Register forward hooks capturing DepthDFL input logits and the raw cost volume."""
    from ultralytics.models.yolo.s3d.head import Stereo3DDetHead
    from ultralytics.nn.modules.block import StereoCostVolume

    captured: dict[str, torch.Tensor] = {}
    hooks = []
    head = next((m for m in core.modules() if isinstance(m, Stereo3DDetHead)), None)
    if head is not None:
        hooks.append(
            head.depth_dfl.register_forward_pre_hook(lambda m, inp: captured.update(depth_logits=inp[0].detach()))
        )
    costvol = next((m for m in core.modules() if isinstance(m, StereoCostVolume)), None)
    if costvol is not None:
        hooks.append(costvol.refine.register_forward_pre_hook(lambda m, inp: captured.update(cost_vol=inp[0].detach())))
    return captured, hooks


def _gt_probe_rows(validator, captured: dict) -> list[dict]:
    """Per-GT distribution-shape stats at the GT's P3 cell (DepthDFL) and cost-volume cell."""
    from ultralytics.data.stereo.box3d import Box3D
    from ultralytics.models.yolo.s3d.preprocess import compute_letterbox_params
    from ultralytics.models.yolo.s3d.val import _reverse_letterbox_calib

    batch = validator._current_batch
    imgsz = validator.args.imgsz
    in_h, in_w = (imgsz, imgsz) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
    p3_hw = (in_h // 8) * (in_w // 8)
    rows = []
    depth_logits = captured.get("depth_logits")  # [B, n_bins, HW_total]
    cost_vol = captured.get("cost_vol")  # [B, n_disp_bins, Hc, Wc]
    for si, labels in enumerate(batch.get("labels", [])):
        calib_lb = batch["calib"][si] if si < len(batch.get("calib", [])) else None
        ori = batch["ori_shape"][si] if si < len(batch.get("ori_shape", [])) else None
        if not isinstance(calib_lb, dict) or ori is None:
            continue
        scale, pad_left, pad_top = compute_letterbox_params(int(ori[0]), int(ori[1]), imgsz)
        calib = _reverse_letterbox_calib(calib_lb, scale, pad_left, pad_top, int(ori[1]), int(ori[0]))
        gt_boxes = [b for lab in labels if (b := Box3D.from_label(lab, calib, class_names=validator.names)) is not None]
        for gb in gt_boxes:
            x, y, z = gb.center_3d
            if z <= 0:
                continue
            u = (calib["fx"] * x / z + calib["cx"]) * scale + pad_left
            v = (calib["fy"] * y / z + calib["cy"]) * scale + pad_top
            row: dict[str, float] = {"z_gt": float(z), "img_key": f"{validator.batch_i}_{si}"}
            if depth_logits is not None:
                flat = int(v // 8) * (in_w // 8) + int(u // 8)
                if 0 <= flat < min(p3_hw, depth_logits.shape[2]):
                    ds = dist_stats(depth_logits[si, :, flat][None])
                    row.update(dfl_entropy=float(ds["entropy"][0]), dfl_top2=float(ds["top2_mass"][0]))
            if cost_vol is not None:
                stride_cv = in_h / cost_vol.shape[2]
                cu, cv_ = int(u / stride_cv), int(v / stride_cv)
                if 0 <= cv_ < cost_vol.shape[2] and 0 <= cu < cost_vol.shape[3]:
                    ds = dist_stats(cost_vol[si, :, cv_, cu][None])
                    gt_disp_bins = (calib["fx"] * scale) * calib["baseline"] / z / stride_cv
                    row.update(
                        cv_entropy=float(ds["entropy"][0]),
                        cv_peak_err_bins=float(abs(float(ds["argmax"][0]) - gt_disp_bins)),
                    )
            rows.append(row)
    return rows


def _shift_probe(core, validator, shift_px: int) -> list[tuple[float, float]]:
    """Counterfactual right-image shift on the current batch: (base_z, shifted_z) per matched det pair."""
    batch = validator._current_batch
    decode_kw = dict(
        batch=batch,
        args=validator.args,
        use_geometric=False,
        use_dense_alignment=False,
        conf_threshold=0.25,
        top_k=100,
        iou_thres=getattr(validator.args, "iou", 0.45),
        imgsz=validator.args.imgsz,
        mean_dims=getattr(validator, "mean_dims", None),
        std_dims=getattr(validator, "std_dims", None),
        class_names=getattr(validator, "names", None),
        score_k=getattr(validator.args, "score_k", 0.5),
    )
    with torch.no_grad():
        base = _decode_forward(core, batch["img"], decode_kw)
        img2 = batch["img"].clone()
        # Roll right-image channels rightward: content at u moves to u+k → measured disparity u_L−u_R
        # DECREASES by k → dpx = −shift_px in expected_dz.
        img2[:, 3:6] = torch.roll(img2[:, 3:6], shifts=shift_px, dims=3)
        shifted = _decode_forward(core, img2, decode_kw)
    pairs = []
    for boxes_a, boxes_b in zip(base, shifted):
        for ba in boxes_a:
            if not boxes_b:
                continue
            bb = min(boxes_b, key=lambda o: np.linalg.norm(np.array(o.center_3d) - np.array(ba.center_3d)))
            if np.linalg.norm(np.array(bb.center_3d) - np.array(ba.center_3d)) < max(1.0, 0.25 * ba.center_3d[2]):
                pairs.append((float(ba.center_3d[2]), float(bb.center_3d[2])))
    return pairs


def _decode_forward(core, img, decode_kw):
    """Forward the raw module and decode to Box3D lists."""
    from ultralytics.models.yolo.s3d.preprocess import decode_and_refine_predictions

    out = core(img)
    if isinstance(out, (tuple, list)) and len(out) == 2 and isinstance(out[1], dict):
        preds = {**out[1], "det": out[0]}
    else:
        preds = out
    return decode_and_refine_predictions(preds=preds, **decode_kw)


def _grad_section(weights: str, validator, n_batches: int, device: str) -> str:
    """Offline gradient probe: per-branch norms + pairwise cosine conflict averaged over N val batches."""
    from ultralytics.cfg import get_cfg

    fresh = YOLO(weights)  # unfused copy — val fuses Conv+BN, which would distort training-time gradients
    core = fresh.model.to(validator.device)
    if not hasattr(core, "args") or isinstance(getattr(core, "args", None), dict):
        core.args = get_cfg()
    if getattr(core, "criterion", None) is None:
        core.criterion = core.init_criterion()
    norms_acc: dict[str, list[float]] = {}
    cos_acc: dict[tuple[str, str], list[float]] = {}
    for bi, batch in enumerate(validator.dataloader):
        if bi >= n_batches:
            break
        batch = validator.preprocess(batch)
        out = grad_diag_step(core, batch)
        for k, v in out["norms"].items():
            norms_acc.setdefault(k, []).append(v)
        for k, v in out["cosine"].items():
            if not np.isnan(v):
                cos_acc.setdefault(k, []).append(v)
    if not norms_acc:
        return "_No gradient data collected._"
    mean_norms = {k: float(np.mean(v)) for k, v in norms_acc.items()}
    gbar = np.mean(list(mean_norms.values())) or 1.0
    rows = [[k, f"{v:.4g}", f"{v / gbar:.2f}"] for k, v in sorted(mean_norms.items(), key=lambda t: -t[1])]
    table = _md_table(["branch", "‖g‖ on shared neck", "ratio vs mean"], rows)
    crows = [
        [f"{a} ↔ {b}", f"{np.mean(v):+.3f}", "**CONFLICT**" if np.mean(v) < 0 else ""]
        for (a, b), v in sorted(cos_acc.items(), key=lambda t: np.mean(t[1]))
    ]
    ctable = _md_table(["branch pair", "mean cos(g_i, g_j)", ""], crows)
    return (
        table
        + "\n\nGradNorm ratio ≫ 1 = branch dominates shared features; ≪ 1 = starved.\n\n"
        + ctable
        + "\n\nPCGrad: persistent cos < 0 means the pair fights on shared parameters — evidence for "
        "uncertainty weighting or gradient surgery over hand-tuned loss weights."
    )


def main() -> None:
    """Parse args, run the val pass with probes attached, compute analyses, write report.md."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", required=True, help="Checkpoint path or resolvable name.")
    p.add_argument("--data", default="kitti-stereo.yaml", help="Dataset YAML.")
    p.add_argument("--imgsz", nargs=2, type=int, default=[384, 1248], help="Validation image size [H, W].")
    p.add_argument("--batch", type=int, default=8, help="Validation batch size.")
    p.add_argument("--device", default="0", help="CUDA device, e.g. '0' or 'cpu'.")
    p.add_argument("--out-dir", default="diagnosis", help="Output directory.")
    p.add_argument("--bootstrap", type=int, default=200, help="Bootstrap resamples (0 = skip).")
    p.add_argument("--no-oracle", action="store_true", help="Skip the oracle-swap ladder.")
    p.add_argument("--no-probes", action="store_true", help="Skip DFL/cost-volume/shift probes.")
    p.add_argument("--shift-px", type=int, default=2, help="Right-image shift in px for the stereo probe.")
    p.add_argument("--shift-batches", type=int, default=4, help="Batches to run the shift probe on.")
    p.add_argument("--grad-batches", type=int, default=0, help="Batches for the gradient probe (0 = skip).")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)

    # Build and drive the validator directly (mirrors Model.val) so we keep a handle on it for
    # the dataloader, hooks, and per-batch probe callback — everything happens in ONE val pass.
    v_args = {
        **model.overrides,
        "rect": True,
        "data": args.data,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "plots": False,
        "verbose": False,
        "mode": "val",
    }
    validator = model._smart_load("validator")(args=v_args, _callbacks=model.callbacks)

    probe_rows: list[dict] = []
    shift_pairs: list[tuple[float, float]] = []
    fx0, b0 = [None], [None]
    captured, hooks = ({}, [])
    if not args.no_probes:
        captured, hooks = _probe_hooks(model.model)

        def _probe_cb(v):
            try:
                probe_rows.extend(_gt_probe_rows(v, captured))
                if v.batch_i < args.shift_batches:
                    shift_pairs.extend(_shift_probe(model.model, v, args.shift_px))
                    if fx0[0] is None and v._current_batch.get("calib"):
                        c0 = v._current_batch["calib"][0]
                        ori0 = v._current_batch["ori_shape"][0]
                        from ultralytics.models.yolo.s3d.preprocess import compute_letterbox_params

                        s0, _, _ = compute_letterbox_params(int(ori0[0]), int(ori0[1]), v.args.imgsz)
                        fx0[0] = float(c0["fx"]) / s0  # original-image fx
                        b0[0] = float(c0.get("baseline", 0.54))
            except Exception as e:  # probes must never break the val pass
                LOGGER.warning("s3d diagnose probe failed on batch %s: %s", v.batch_i, e)
            finally:
                captured.clear()

        validator.add_callback("on_val_batch_end", _probe_cb)

    try:
        validator(model=model.model)
    finally:
        for h in hooks:
            h.remove()

    stats, names = validator.metrics.stats, validator.metrics.names
    if not stats:
        raise SystemExit("No validation stats collected — check --data/--weights/--imgsz.")

    sections: dict[str, str] = {}
    sections["Error profile (stable per-component eval)"] = _error_profile_section(stats, out_dir)

    ladder = None
    if not args.no_oracle:
        sections["Oracle-swap ladder (GT substitution, MonoDLE/TIDE3D)"], ladder = _oracle_section(stats, names)
    if args.bootstrap > 0:
        sections[f"Bootstrap AP CIs ({args.bootstrap} resamples)"] = _bootstrap_section(stats, names, args.bootstrap)

    if probe_rows:
        with open(out_dir / "probe_records.csv", "w", newline="") as f:
            keys = sorted({k for r in probe_rows for k in r})
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(probe_rows)
        lines = [f"GT objects probed: {len(probe_rows)}"]
        for col, label in (("dfl_entropy", "DepthDFL entropy"), ("cv_entropy", "cost-volume entropy")):
            vals = [r[col] for r in probe_rows if col in r]
            if vals:
                lines.append(f"- mean {label}: {np.mean(vals):.3f} (0 = one-hot, 1 = uniform)")
        errs = [r["cv_peak_err_bins"] for r in probe_rows if "cv_peak_err_bins" in r]
        if errs:
            lines.append(f"- cost-volume peak vs GT disparity: median |err| = {np.median(errs):.2f} bins")
        lines.append("Per-object records in probe_records.csv — correlate entropy vs |dz| (GFLv2/AcfNet).")
        sections["Representation probes (GT-anchored)"] = "\n".join(lines)

    if shift_pairs and fx0[0]:
        base_z = np.array([a for a, _ in shift_pairs])
        shifted_z = np.array([b for _, b in shift_pairs])
        sens = stereo_sensitivity(base_z, shifted_z, base_z, dpx=-args.shift_px, fx=fx0[0], baseline=b0[0])
        sections["Counterfactual stereo probe"] = (
            f"Right image rolled by +{args.shift_px} px on {len(shift_pairs)} matched detections → "
            f"**stereo sensitivity = {sens:.2f}**.\n\n"
            "1.0 = depth fully stereo-driven; 0.0 = stereo ignored (monocular shortcut, e.g. apparent-size cue "
            "on constant-dimension objects); intermediate = fused."
        )

    if args.grad_batches > 0:
        try:
            sections["Gradient diagnostics (per-branch, shared neck)"] = _grad_section(
                args.weights, validator, args.grad_batches, args.device
            )
        except Exception as e:
            LOGGER.warning("s3d diagnose gradient probe failed: %s", e)

    report = render_report(sections)
    (out_dir / "report.md").write_text(report)
    print(report)
    if ladder:
        print("\nRanked levers (ΔAP3D@0.7):", rank_levers(ladder))
    print(f"\nSaved to {out_dir / 'report.md'}")


if __name__ == "__main__":
    main()
