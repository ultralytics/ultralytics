import argparse
from pathlib import Path

from .pipeline.orchestrator import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline CV paper-writing agent")
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_pipeline(
        repo_root=args.repo_root,
        runs_root=args.runs_root,
        workspace=args.workspace,
        resume=args.resume,
    )
    return 0
