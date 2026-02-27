# CV Paper Agent — SC-ELAN

Offline agent for scanning SC-ELAN experiment runs and drafting a LaTeX paper skeleton with resume-safe caching.

## Usage

```bash
cd cv-paper-agent
python -m cv_paper_agent \
  --repo-root .. \
  --runs-root ../runs \
  --workspace workspace \
  --resume
```

## Project Structure

```
cv-paper-agent/
  configs/          # agent, paper, runs layout/schema configs
  src/              # Python source (cv_paper_agent package)
  templates/latex/  # IEEEtran LaTeX template + section stubs
  workspace/        # Generated outputs (git-ignored except .gitkeep)
```

## Outputs

- `workspace/workdir/cache/experiments.jsonl` — parsed experiment cache
- `workspace/paper_project/sections/experiments.tex` — auto-generated experiments table
- `workspace/workdir/state.json` — pipeline state checkpoint

## Data Sources

- **Runs**: `../runs/detect/` (SC-ELAN-VisDrone + SC-ELAN-v2)
- **Summary**: `../sc-elan.md`
