from pathlib import Path

from ..io.repo_reader import ensure_repo_exists
from ..parsing.yaml_parser import load_yaml
from ..storage.cache_store import atomic_write_jsonl, load_jsonl
from ..storage.state_store import atomic_write_json, load_json
from ..storage.trace_logger import TraceLogger
from .stages import stage_bootstrap, stage_parse, stage_scan, stage_write_sections


def run_pipeline(repo_root: Path, runs_root: Path, workspace: Path, resume: bool = True) -> None:
    ensure_repo_exists(repo_root)
    templates = Path(__file__).resolve().parents[3] / "templates" / "latex"
    config_root = Path(__file__).resolve().parents[3] / "configs"
    paper_cfg = load_yaml(config_root / "paper.yaml")
    if not isinstance(paper_cfg, dict):
        paper_cfg = {}

    logger = TraceLogger(workspace / "workdir" / "trace.jsonl")
    logger.log("stage_start", stage="S0_bootstrap")
    paths = stage_bootstrap(workspace, templates)
    workdir = paths["workdir"]
    cache_dir = paths["cache_dir"]

    state_path = workdir / "state.json"
    fp_path = workdir / "fingerprints.json"
    cache_path = cache_dir / "experiments.jsonl"
    layout = load_yaml(config_root / "runs_layout.yaml")
    if not isinstance(layout, dict):
        layout = {}
    marker_rules = layout.get("marker_rules", [["results.csv", "args.yaml"]])
    if not isinstance(marker_rules, list):
        marker_rules = [["results.csv", "args.yaml"]]

    state = load_json(state_path, default={"stage": "init"})
    state["stage"] = "S0_bootstrap"
    atomic_write_json(state_path, state)

    logger.log("stage_start", stage="S1_scan")
    exps = stage_scan(runs_root, marker_rules=marker_rules)
    state["stage"] = "S1_scan"
    state["num_experiments"] = len(exps)
    atomic_write_json(state_path, state)

    old_rows = load_jsonl(cache_path)
    old_cache = {r["exp_id"]: r for r in old_rows}
    old_fps = load_json(fp_path, default={})
    schema = load_yaml(config_root / "runs_schema.yaml")
    if not isinstance(schema, dict):
        schema = {}

    logger.log("stage_start", stage="S2_parse")
    rows, fps = stage_parse(exps, runs_root, old_cache, old_fps, schema, resume=resume)
    atomic_write_jsonl(cache_path, rows)
    atomic_write_json(fp_path, fps)
    state["stage"] = "S2_parse"
    state["cached_experiments"] = len(rows)
    atomic_write_json(state_path, state)

    logger.log("stage_start", stage="S3_draft_experiments")
    logger.log("stage_start", stage="S4_draft_method")
    stage_write_sections(workspace, rows, repo_root=repo_root, paper_cfg=paper_cfg)
    state["stage"] = "S4_draft_method"
    atomic_write_json(state_path, state)
    logger.log("done", experiments=len(rows))
