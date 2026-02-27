from pathlib import Path
from typing import Any, Dict, List

from ..io.artifact_writer import (
    bootstrap_paper_project,
    write_ablation_section,
    write_experiments_section,
    write_method_section,
)
from ..io.runs_reader import index_artifacts
from ..parsing.csv_discovery import discover_experiments
from ..parsing.csv_parser import parse_best_metrics
from ..parsing.yaml_parser import load_yaml
from ..utils.fingerprint import experiment_fingerprint


def stage_bootstrap(workspace: Path, templates_root: Path) -> Dict[str, Path]:
    workdir = workspace / "workdir"
    cache_dir = workdir / "cache"
    paper_project = workspace / "paper_project"
    workdir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_paper_project(templates_root, paper_project)
    return {"workdir": workdir, "cache_dir": cache_dir, "paper_project": paper_project}


def stage_scan(runs_root: Path, marker_rules: List[List[str]] | None = None) -> List[Path]:
    return discover_experiments(runs_root, marker_rules=marker_rules)


def _select_best_metric(best_metrics: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    selection = schema.get("selection", {}) if isinstance(schema, dict) else {}
    best_by = selection.get("best_by", []) if isinstance(selection, dict) else []
    if not isinstance(best_by, list):
        best_by = []

    for key in best_by:
        if key in best_metrics:
            return {"metric": key, "value": best_metrics.get(key)}

    if best_metrics:
        first_key = next(iter(best_metrics.keys()))
        return {"metric": first_key, "value": best_metrics.get(first_key)}

    return {"metric": "N/A", "value": None}


def stage_parse(
    exps: List[Path],
    runs_root: Path,
    old_cache: Dict[str, Dict[str, Any]],
    old_fingerprints: Dict[str, str],
    schema: Dict[str, Any],
    resume: bool,
) -> tuple[List[Dict[str, Any]], Dict[str, str]]:
    rows: List[Dict[str, Any]] = []
    fps: Dict[str, str] = {}
    for exp_dir in exps:
        rel = exp_dir.relative_to(runs_root).as_posix()
        exp_id = rel
        fp = experiment_fingerprint(exp_dir, runs_root)
        fps[exp_id] = fp
        if resume and old_fingerprints.get(exp_id) == fp and exp_id in old_cache:
            rows.append(old_cache[exp_id])
            continue

        args = load_yaml(exp_dir / "args.yaml")
        best_metrics = parse_best_metrics(exp_dir / "results.csv", schema)
        selected_best = _select_best_metric(best_metrics, schema)
        artifacts = index_artifacts(exp_dir, runs_root)
        rows.append(
            {
                "exp_id": exp_id,
                "rel_path": rel,
                "fingerprint": fp,
                "args": args,
                "best_metrics": best_metrics,
                "selected_best": selected_best,
                "artifacts": artifacts,
            }
        )
    return rows, fps


def _extract_method_spec(experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build method spec from the collection of parsed experiments."""
    training: Dict[str, Any] = {}
    model_variants: List[str] = []

    for exp in experiments:
        args = exp.get("args", {})
        if not isinstance(args, dict):
            continue
        model = args.get("model", "")
        if isinstance(model, str) and model:
            short = model.rsplit("/", 1)[-1]
            if short not in model_variants:
                model_variants.append(short)
        if not training:
            for key in ("epochs", "batch", "imgsz", "optimizer", "lr0", "patience", "seed",
                        "momentum", "weight_decay", "warmup_epochs", "close_mosaic"):
                val = args.get(key)
                if val is not None:
                    training[key] = val

    return {
        "method_name": "SC-ELAN",
        "backbone": "YOLO11",
        "training": training,
        "model_variants": sorted(model_variants),
    }


def stage_write_sections(
    workspace: Path,
    experiments: List[Dict[str, Any]],
    *,
    repo_root: Path,
    paper_cfg: Dict[str, Any],
) -> None:
    pp = workspace / "paper_project" / "sections"
    write_experiments_section(pp / "experiments.tex", experiments, repo_root=repo_root, paper_cfg=paper_cfg)
    method_spec = _extract_method_spec(experiments)
    write_method_section(pp / "method.tex", method_spec, repo_root=repo_root, paper_cfg=paper_cfg)
    write_ablation_section(pp / "ablation.tex", experiments, repo_root=repo_root, paper_cfg=paper_cfg)
