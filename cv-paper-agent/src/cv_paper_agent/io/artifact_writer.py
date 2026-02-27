import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def bootstrap_paper_project(templates_root: Path, paper_project: Path) -> None:
    if (paper_project / "main.tex").exists():
        return
    shutil.copytree(templates_root, paper_project, dirs_exist_ok=True)
    (paper_project / "tables").mkdir(parents=True, exist_ok=True)
    (paper_project / "figures").mkdir(parents=True, exist_ok=True)


def _tex_escape(s: str) -> str:
    for ch in ("_", "&", "%", "#"):
        s = s.replace(ch, f"\\{ch}")
    return s


def _fmt_metric(v: Any, decimals: int = 3) -> str:
    if v is None:
        return "--"
    try:
        return f"{float(v):.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)


def _short_name(exp_id: str) -> str:
    """Extract a human-readable short name from experiment id."""
    name = exp_id.rsplit("/", 1)[-1]
    name = re.sub(r"^SC-ELAN-VisDrone_", "", name)
    name = re.sub(r"^yolo11-scelan-?", "SC-ELAN-", name)
    name = name.replace("detect-cai", "DetCAI")
    name = name.replace("-tscg", "-TSCG")
    name = name.replace("-lska", "-LSKA")
    if name == "SC-ELAN-":
        name = "SC-ELAN (base)"
    return name


def _get_metric(exp: Dict[str, Any], *keys: str) -> Optional[float]:
    bm = exp.get("best_metrics", {})
    if not isinstance(bm, dict):
        return None
    for k in keys:
        v = bm.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def _classify_group(exp_id: str) -> str:
    if "SC-ELAN-v2" in exp_id:
        return "v2"
    return "v1"


def _classify_v2_phase(exp_id: str) -> str:
    name = exp_id.rsplit("/", 1)[-1]
    if "-p1" in name:
        return "Phase 1"
    if "-p2" in name:
        return "Phase 2"
    if "-p3" in name:
        return "Phase 3"
    return "Other"


def _sort_key(exp: Dict[str, Any]) -> Tuple[int, str]:
    eid = exp.get("exp_id", "")
    group = 0 if "SC-ELAN-v2" not in eid else 1
    return (group, eid)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _parse_markdown_table(lines: List[str], start_idx: int) -> Tuple[List[str], List[List[str]]]:
    """
    Parse a GitHub-flavored markdown table.
    Expect:
      header row: | a | b |
      separator:  |---|---|
      data rows...
    """
    header_line = lines[start_idx].strip()
    if "|" not in header_line:
        return [], []
    header = [c.strip() for c in header_line.strip("|").split("|")]

    rows: List[List[str]] = []
    i = start_idx + 2  # skip separator
    while i < len(lines):
        raw = lines[i].rstrip("\n")
        if raw.strip() == "" or "|" not in raw:
            break
        cols = [c.strip() for c in raw.strip().strip("|").split("|")]
        if len(cols) >= len(header):
            rows.append(cols[: len(header)])
        i += 1
    return header, rows


def _md_unbold(s: str) -> str:
    return s.replace("**", "").strip()


def _to_float_or_none(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _load_sc_elan_summary(repo_root: Path, paper_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Load summary tables from sc-elan.md (or a configured summary_doc).
    Returns a dict with optional keys:
      - testdev_overall: list[dict]
      - infer_overall: list[dict]
      - v2_overall: list[dict]
      - v2_perclass_map50: list[dict]
    """
    summary_rel = paper_cfg.get("summary_doc", "sc-elan.md")
    if not isinstance(summary_rel, str) or not summary_rel:
        summary_rel = "sc-elan.md"
    md_path = (repo_root / summary_rel).resolve()
    text = _read_text(md_path)
    if not text:
        return {"_path": md_path.as_posix(), "_error": "missing_or_unreadable"}

    lines = text.splitlines()
    out: Dict[str, Any] = {"_path": md_path.as_posix()}

    def find_header(target: str) -> int:
        for idx, ln in enumerate(lines):
            if ln.strip() == target:
                return idx
        return -1

    # 7.1 overall test-dev table
    idx = find_header("| Model Variant | Parameters | GFLOPs | mAP50 | mAP50-95 | Speed (ms) |")
    if idx != -1:
        header, rows = _parse_markdown_table(lines, idx)
        parsed = []
        for r in rows:
            model = _md_unbold(r[0])
            map50_95 = _to_float_or_none(_md_unbold(r[4]))
            # Filter out separator/comment rows inside markdown tables.
            if not model or map50_95 is None:
                continue
            parsed.append(
                {
                    "model": model,
                    "params_m": _md_unbold(r[1]),
                    "gflops": _to_float_or_none(_md_unbold(r[2])),
                    "map50": _to_float_or_none(_md_unbold(r[3])),
                    "map50_95": map50_95,
                    "speed_ms": _to_float_or_none(_md_unbold(r[5])),
                }
            )
        out["testdev_overall"] = parsed

    # 7.3 inference table
    idx = find_header("| Model | Preprocess (ms) | Inference (ms) | Postprocess (ms) | Total (ms) |")
    if idx != -1:
        header, rows = _parse_markdown_table(lines, idx)
        parsed = []
        for r in rows:
            model = _md_unbold(r[0])
            parsed.append(
                {
                    "model": model,
                    "pre_ms": _to_float_or_none(_md_unbold(r[1])),
                    "infer_ms": _to_float_or_none(_md_unbold(r[2])),
                    "post_ms": _to_float_or_none(_md_unbold(r[3])),
                    "total_ms": _to_float_or_none(_md_unbold(r[4])),
                }
            )
        out["infer_overall"] = parsed

    # 8.2 v2 overall table
    idx = find_header("| Model | Config | Params | GFLOPs | P | R | mAP50 | mAP50-95 | Total (ms) |")
    if idx != -1:
        header, rows = _parse_markdown_table(lines, idx)
        parsed = []
        for r in rows:
            parsed.append(
                {
                    "model": _md_unbold(r[0]),
                    "config": _md_unbold(r[1]),
                    "params_m": _md_unbold(r[2]),
                    "gflops": _to_float_or_none(_md_unbold(r[3])),
                    "p": _to_float_or_none(_md_unbold(r[4])),
                    "r": _to_float_or_none(_md_unbold(r[5])),
                    "map50": _to_float_or_none(_md_unbold(r[6])),
                    "map50_95": _to_float_or_none(_md_unbold(r[7])),
                    "total_ms": _to_float_or_none(_md_unbold(r[8])),
                }
            )
        out["v2_overall"] = parsed

    # 8.7 v2 per-class mAP50 table
    idx = find_header("| Model | pedestrian | people | bicycle | tricycle | awning-tri | motor |")
    if idx != -1:
        header, rows = _parse_markdown_table(lines, idx)
        parsed = []
        for r in rows:
            parsed.append(
                {
                    "model": _md_unbold(r[0]),
                    "pedestrian": _to_float_or_none(_md_unbold(r[1])),
                    "people": _to_float_or_none(_md_unbold(r[2])),
                    "bicycle": _to_float_or_none(_md_unbold(r[3])),
                    "tricycle": _to_float_or_none(_md_unbold(r[4])),
                    "awning_tricycle": _to_float_or_none(_md_unbold(r[5])),
                    "motor": _to_float_or_none(_md_unbold(r[6])),
                }
            )
        out["v2_perclass_map50"] = parsed

    return out


# ---------------------------------------------------------------------------
# experiments.tex
# ---------------------------------------------------------------------------

def write_experiments_section(
    path: Path,
    experiments: List[Dict[str, Any]],
    *,
    repo_root: Path,
    paper_cfg: Dict[str, Any],
) -> None:
    v1_exps = [e for e in experiments if _classify_group(e["exp_id"]) == "v1"]
    v2_exps = [e for e in experiments if _classify_group(e["exp_id"]) == "v2"]
    summary = _load_sc_elan_summary(repo_root, paper_cfg)

    lines: List[str] = []
    lines.append("\\section{Experiments}")
    lines.append("")
    lines.append("\\subsection{Experimental Setup}")
    lines.append(
        "All models are trained on VisDrone2019-DET with identical settings: "
        "300 epochs, batch=16, imgsz=640, seed=0, patience=50, "
        "pretrained=true, optimizer=auto. "
        f"A total of {len(experiments)} training runs were discovered and parsed."
    )

    if v1_exps:
        lines.append("")
        lines.append("\\subsection{Architecture Variant Comparison}")
        lines.append(_build_main_table(v1_exps, "tab:v1-compare",
                                       "Validation-best metrics for SC-ELAN architecture variants (VisDrone val set)."))

    if v2_exps:
        lines.append("")
        lines.append("\\subsection{SC-ELAN v2 Phased Pipeline}")
        lines.append(_build_main_table(v2_exps, "tab:v2-compare",
                                       "Validation-best metrics for SC-ELAN v2 phased experiments."))

    if v2_exps:
        lines.append("")
        lines.append(_build_v2_phase_summary(v2_exps))

    # Test-dev summary tables from sc-elan.md
    testdev = summary.get("testdev_overall")
    if isinstance(testdev, list) and testdev:
        lines.append("")
        lines.append("\\subsection{Test-dev Benchmark Results}")
        lines.append(
            "Table~\\ref{tab:testdev-overall} reports official VisDrone test-dev results "
            "summarized from the project experiment report (sc-elan.md)."
        )
        lines.append(
            "Note that Tables~\\ref{tab:v1-compare} and~\\ref{tab:v2-compare} are based on "
            "validation-best checkpoints recorded in training logs, while the test-dev table "
            "is the primary comparison for final claims."
        )
        lines.append(_build_testdev_overall_table(testdev))

    infer = summary.get("infer_overall")
    if isinstance(infer, list) and infer:
        lines.append("")
        lines.append("\\subsection{Inference Efficiency}")
        lines.append(
            "Table~\\ref{tab:infer-overall} summarizes preprocessing/inference/postprocessing "
            "latency measured on RTX~4090 as reported in sc-elan.md."
        )
        lines.append(_build_infer_table(infer))

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_main_table(exps: List[Dict[str, Any]], label: str, caption: str) -> str:
    rows: List[str] = []
    rows.append("\\begin{table}[htbp]")
    rows.append("\\centering")
    rows.append(f"\\caption{{{caption}}}")
    rows.append(f"\\label{{{label}}}")
    rows.append("\\small")
    rows.append("\\begin{tabular}{lcccccc}")
    rows.append("\\toprule")
    rows.append("Model & Epoch & P & R & mAP50 & mAP50-95 & Best Epoch \\\\")
    rows.append("\\midrule")

    best_map5095 = max(
        (_get_metric(e, "metrics/mAP50-95(B)") or 0.0 for e in exps),
        default=0.0,
    )

    for exp in sorted(exps, key=_sort_key):
        name = _tex_escape(_short_name(exp["exp_id"]))
        args = exp.get("args", {})
        epochs = args.get("epochs", "--") if isinstance(args, dict) else "--"
        p = _get_metric(exp, "metrics/precision(B)")
        r = _get_metric(exp, "metrics/recall(B)")
        m50 = _get_metric(exp, "metrics/mAP50(B)")
        m5095 = _get_metric(exp, "metrics/mAP50-95(B)")
        best_ep = _get_metric(exp, "epoch")

        m5095_str = _fmt_metric(m5095)
        if m5095 is not None and abs(m5095 - best_map5095) < 1e-6:
            m5095_str = f"\\textbf{{{m5095_str}}}"

        rows.append(
            f"{name} & {epochs} & {_fmt_metric(p)} & {_fmt_metric(r)} "
            f"& {_fmt_metric(m50)} & {m5095_str} & {_fmt_metric(best_ep, 0)} \\\\"
        )

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")
    return "\n".join(rows)


def _build_v2_phase_summary(v2_exps: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("\\subsubsection{Phase Summary}")

    phase_groups: Dict[str, List[Dict[str, Any]]] = {}
    for exp in v2_exps:
        phase = _classify_v2_phase(exp["exp_id"])
        phase_groups.setdefault(phase, []).append(exp)

    for phase_name in ["Phase 1", "Phase 2", "Phase 3"]:
        group = phase_groups.get(phase_name, [])
        if not group:
            continue
        best = max(group, key=lambda e: _get_metric(e, "metrics/mAP50-95(B)") or 0.0)
        bname = _short_name(best["exp_id"])
        bval = _get_metric(best, "metrics/mAP50-95(B)")
        lines.append(
            f"\\textbf{{{phase_name}}}: {len(group)} runs, "
            f"best = {_tex_escape(bname)} (mAP50-95 = {_fmt_metric(bval)})."
        )
        lines.append("")

    return "\n".join(lines)


def _build_testdev_overall_table(rows_in: List[Dict[str, Any]]) -> str:
    # keep a stable order: sort by mAP50-95 desc then model name
    rows_sorted = sorted(
        rows_in,
        key=lambda r: (-(r.get("map50_95") or -1.0), str(r.get("model", ""))),
    )
    best = max((r.get("map50_95") or 0.0 for r in rows_sorted), default=0.0)

    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Overall performance on VisDrone2019-DET test-dev (from sc-elan.md).}")
    lines.append("\\label{tab:testdev-overall}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("Model & Params (M) & GFLOPs & mAP50 & mAP50-95 & Speed (ms) \\\\")
    lines.append("\\midrule")
    for r in rows_sorted:
        model = _tex_escape(str(r.get("model", "N/A")))
        params_m = _tex_escape(str(r.get("params_m", "N/A")))
        gflops = _fmt_metric(r.get("gflops"), 1)
        map50 = _fmt_metric(r.get("map50"))
        map50_95 = r.get("map50_95")
        map50_95_str = _fmt_metric(map50_95)
        if map50_95 is not None and abs(float(map50_95) - float(best)) < 1e-9:
            map50_95_str = f"\\textbf{{{map50_95_str}}}"
        speed = _fmt_metric(r.get("speed_ms"), 1)
        lines.append(f"{model} & {params_m} & {gflops} & {map50} & {map50_95_str} & {speed} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _build_infer_table(rows_in: List[Dict[str, Any]]) -> str:
    rows_sorted = sorted(rows_in, key=lambda r: str(r.get("model", "")))
    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Latency breakdown on RTX~4090 (from sc-elan.md).}")
    lines.append("\\label{tab:infer-overall}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccc}")
    lines.append("\\toprule")
    lines.append("Model & Pre (ms) & Infer (ms) & Post (ms) & Total (ms) \\\\")
    lines.append("\\midrule")
    for r in rows_sorted:
        model = _tex_escape(str(r.get("model", "N/A")))
        lines.append(
            f"{model} & {_fmt_metric(r.get('pre_ms'), 1)} & {_fmt_metric(r.get('infer_ms'), 1)} "
            f"& {_fmt_metric(r.get('post_ms'), 1)} & {_fmt_metric(r.get('total_ms'), 1)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# method.tex
# ---------------------------------------------------------------------------

def write_method_section(
    path: Path,
    method_spec: Dict[str, Any],
    *,
    repo_root: Path,
    paper_cfg: Dict[str, Any],
) -> None:
    name = method_spec.get("method_name", "SC-ELAN")
    backbone = method_spec.get("backbone", "YOLO11")
    training = method_spec.get("training", {})
    summary = _load_sc_elan_summary(repo_root, paper_cfg)
    v2_overall = summary.get("v2_overall") if isinstance(summary, dict) else None

    lines: List[str] = []
    lines.append("\\section{Method}")
    lines.append("")
    lines.append(f"\\subsection{{Overview}}")
    lines.append(
        f"{_tex_escape(name)} integrates large-kernel spatial attention (LSKA), "
        f"two-stage context gating (TSCG), and context-aware RepConv blocks into "
        f"the {_tex_escape(backbone)} detection framework. "
        f"The design targets three principles: context awareness, feature fidelity, "
        f"and attentional interaction."
    )

    if training:
        lines.append("")
        lines.append("\\subsection{Training Configuration}")
        lines.append("\\begin{itemize}")
        for key in ("epochs", "batch", "imgsz", "optimizer", "lr0", "patience", "seed"):
            val = training.get(key)
            if val is not None:
                lines.append(f"  \\item \\texttt{{{_tex_escape(key)}}}: {_tex_escape(str(val))}")
        lines.append("\\end{itemize}")

    models = method_spec.get("model_variants", [])
    if models:
        lines.append("")
        lines.append("\\subsection{Model Variants}")
        lines.append(f"A total of {len(models)} model configurations were evaluated:")
        lines.append("\\begin{itemize}")
        for m in models[:15]:
            lines.append(f"  \\item \\texttt{{{_tex_escape(m)}}}")
        if len(models) > 15:
            lines.append(f"  \\item ... and {len(models) - 15} more")
        lines.append("\\end{itemize}")

    if isinstance(v2_overall, list) and v2_overall:
        # Show the best reported v2 setting from test-dev summary, if available.
        best = max(v2_overall, key=lambda r: r.get("map50_95") or 0.0)
        lines.append("")
        lines.append("\\subsection{Recommended Configuration}")
        lines.append(
            "Based on the phased study summarized in sc-elan.md, the recommended setting is "
            f"\\textbf{{{_tex_escape(str(best.get('model', 'N/A')))}}} "
            f"(mAP50-95 = {_fmt_metric(best.get('map50_95'))})."
        )

    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# ablation.tex
# ---------------------------------------------------------------------------

def write_ablation_section(
    path: Path,
    experiments: List[Dict[str, Any]],
    *,
    repo_root: Path,
    paper_cfg: Dict[str, Any],
) -> None:
    v2_exps = [e for e in experiments if _classify_group(e["exp_id"]) == "v2"]
    v1_exps = [e for e in experiments if _classify_group(e["exp_id"]) == "v1"]
    summary = _load_sc_elan_summary(repo_root, paper_cfg)

    lines: List[str] = []
    lines.append("\\section{Ablation Study}")
    lines.append("")

    if v2_exps:
        v2_overall = summary.get("v2_overall")
        if isinstance(v2_overall, list) and v2_overall:
            lines.append("\\subsection{v2 Test-dev Summary (from sc-elan.md)}")
            lines.append(_build_v2_testdev_table(v2_overall))
            lines.append("")

        v2_pc = summary.get("v2_perclass_map50")
        if isinstance(v2_pc, list) and v2_pc:
            lines.append("\\subsection{v2 Per-class mAP50 (from sc-elan.md)}")
            lines.append(_build_v2_perclass_table(v2_pc))
            lines.append("")

        p1 = [e for e in v2_exps if _classify_v2_phase(e["exp_id"]) == "Phase 1"]
        p2 = [e for e in v2_exps if _classify_v2_phase(e["exp_id"]) == "Phase 2"]
        p3 = [e for e in v2_exps if _classify_v2_phase(e["exp_id"]) == "Phase 3"]

        if p1:
            lines.append("\\subsection{Phase 1: CAI Parameter Sweep}")
            lines.append(_build_ablation_table(
                p1, "tab:abl-p1",
                "Phase 1 CAI $\\alpha$/$\\beta$ parameter sweep results (val-best)."))
            lines.append("")

        if p2:
            lines.append("\\subsection{Phase 2: SA-LSKA Kernel Ablation}")
            lines.append(_build_ablation_table(
                p2, "tab:abl-p2",
                "Phase 2 SA-LSKA kernel size and TSCG variant ablation (val-best)."))
            lines.append("")

        if p3:
            lines.append("\\subsection{Phase 3: P3-FRM Integration}")
            lines.append(_build_ablation_table(
                p3, "tab:abl-p3",
                "Phase 3 P3-FRM feature reuse module integration (val-best)."))
            lines.append("")

    if v1_exps:
        efficiency = [e for e in v1_exps if any(
            k in e["exp_id"] for k in ("efficient", "repadd", "repexact", "slim", "mixed")
        )]
        context = [e for e in v1_exps if e not in efficiency]

        if context:
            lines.append("\\subsection{Context Module Variants}")
            lines.append(_build_ablation_table(
                context, "tab:abl-context",
                "Architecture variants: context and attention module comparison (val-best)."))
            lines.append("")

        if efficiency:
            lines.append("\\subsection{Efficiency Variants}")
            lines.append(_build_ablation_table(
                efficiency, "tab:abl-efficiency",
                "Lightweight/efficiency-oriented architecture variants (val-best)."))
            lines.append("")

    if not lines[-1]:
        lines.pop()
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_v2_testdev_table(rows_in: List[Dict[str, Any]]) -> str:
    rows_sorted = sorted(
        rows_in,
        key=lambda r: (-(r.get("map50_95") or -1.0), str(r.get("model", ""))),
    )
    best = max((r.get("map50_95") or 0.0 for r in rows_sorted), default=0.0)

    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{SC-ELAN v2 phased results on test-dev (from sc-elan.md).}")
    lines.append("\\label{tab:v2-testdev}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & Config & GFLOPs & P & R & mAP50 & mAP50-95 \\\\")
    lines.append("\\midrule")
    for r in rows_sorted:
        model = _tex_escape(str(r.get("model", "N/A")))
        config = _tex_escape(str(r.get("config", "N/A")))
        gflops = _fmt_metric(r.get("gflops"), 1)
        p = _fmt_metric(r.get("p"))
        rr = _fmt_metric(r.get("r"))
        m50 = _fmt_metric(r.get("map50"))
        m5095 = r.get("map50_95")
        m5095_str = _fmt_metric(m5095)
        if m5095 is not None and abs(float(m5095) - float(best)) < 1e-9:
            m5095_str = f"\\textbf{{{m5095_str}}}"
        lines.append(f"{model} & {config} & {gflops} & {p} & {rr} & {m50} & {m5095_str} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)


def _build_v2_perclass_table(rows_in: List[Dict[str, Any]]) -> str:
    # Stable order: keep as sc-elan.md order if possible by sorting on model string.
    rows_sorted = sorted(rows_in, key=lambda r: str(r.get("model", "")))

    def best_of(key: str) -> float:
        return max((r.get(key) or 0.0 for r in rows_sorted), default=0.0)

    b_ped = best_of("pedestrian")
    b_people = best_of("people")
    b_bic = best_of("bicycle")
    b_tri = best_of("tricycle")
    b_aw = best_of("awning_tricycle")
    b_mot = best_of("motor")

    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Per-class mAP50 for SC-ELAN v2 variants (from sc-elan.md).}")
    lines.append("\\label{tab:v2-perclass}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lcccccc}")
    lines.append("\\toprule")
    lines.append("Model & ped. & people & bicycle & tricycle & awn.-tri. & motor \\\\")
    lines.append("\\midrule")
    for r in rows_sorted:
        model = _tex_escape(str(r.get("model", "N/A")))

        def fmt(key: str, best: float) -> str:
            v = r.get(key)
            s = _fmt_metric(v)
            if v is not None and abs(float(v) - float(best)) < 1e-9:
                return f"\\textbf{{{s}}}"
            return s

        lines.append(
            f"{model} & {fmt('pedestrian', b_ped)} & {fmt('people', b_people)} & {fmt('bicycle', b_bic)} "
            f"& {fmt('tricycle', b_tri)} & {fmt('awning_tricycle', b_aw)} & {fmt('motor', b_mot)} \\\\"
        )
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    return "\n".join(lines)

def _build_ablation_table(exps: List[Dict[str, Any]], label: str, caption: str) -> str:
    rows: List[str] = []
    rows.append("\\begin{table}[htbp]")
    rows.append("\\centering")
    rows.append(f"\\caption{{{caption}}}")
    rows.append(f"\\label{{{label}}}")
    rows.append("\\small")
    rows.append("\\begin{tabular}{lcccc}")
    rows.append("\\toprule")
    rows.append("Variant & P & R & mAP50 & mAP50-95 \\\\")
    rows.append("\\midrule")

    best_val = max(
        (_get_metric(e, "metrics/mAP50-95(B)") or 0.0 for e in exps),
        default=0.0,
    )

    for exp in sorted(exps, key=_sort_key):
        name = _tex_escape(_short_name(exp["exp_id"]))
        p = _get_metric(exp, "metrics/precision(B)")
        r = _get_metric(exp, "metrics/recall(B)")
        m50 = _get_metric(exp, "metrics/mAP50(B)")
        m5095 = _get_metric(exp, "metrics/mAP50-95(B)")

        m5095_str = _fmt_metric(m5095)
        if m5095 is not None and abs(m5095 - best_val) < 1e-6:
            m5095_str = f"\\textbf{{{m5095_str}}}"

        rows.append(
            f"{name} & {_fmt_metric(p)} & {_fmt_metric(r)} "
            f"& {_fmt_metric(m50)} & {m5095_str} \\\\"
        )

    rows.append("\\bottomrule")
    rows.append("\\end{tabular}")
    rows.append("\\end{table}")
    return "\n".join(rows)
