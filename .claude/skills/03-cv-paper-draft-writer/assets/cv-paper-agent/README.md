# CV Paper Agent

Offline, cross-platform agent for scanning CV experiment runs and drafting a LaTeX paper skeleton with resume-safe caching.

## Usage

```bash
python -m cv_paper_agent \
  --repo-root /path/to/repo \
  --runs-root /path/to/runs \
  --workspace cv-paper-agent/workspace \
  --resume
```

## Outputs

- `workspace/workdir/cache/experiments.jsonl`
- `workspace/paper_project/sections/experiments.tex`
- `workspace/workdir/state.json`

## Safety defaults

- Read-only intent for repo/runs input roots.
- No training execution.
- No external network calls.
