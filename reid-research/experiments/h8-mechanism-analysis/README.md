# h8 — ReID Mechanism Analysis

Diagnostic post-mortem of the R1=0.927 champion ReID model, plus one validated experiment.
See `docs/superpowers/specs/2026-05-29-reid-mechanism-analysis-design.md` for the design.

## Run order

All commands assume CWD = repo root.

1. **Stage 1** — extract artifacts (one-time, ~hours, GPU):
   `PYTHONPATH=reid-research/experiments/h8-mechanism-analysis python reid-research/experiments/h8-mechanism-analysis/extract.py`

2. **Stage 2** — failure taxonomy:
   `PYTHONPATH=reid-research/experiments/h8-mechanism-analysis python reid-research/experiments/h8-mechanism-analysis/s2_failure_taxonomy.py`

3. **Stage 3** — champion vs SOLIDER gap:
   `PYTHONPATH=reid-research/experiments/h8-mechanism-analysis python reid-research/experiments/h8-mechanism-analysis/s3_solider_gap.py`

4. **Stage 4** — training dynamics:
   `PYTHONPATH=reid-research/experiments/h8-mechanism-analysis python reid-research/experiments/h8-mechanism-analysis/s4_training_dynamics.py`

5. **Stage 5** — synthesize (writes `to_human/s5_decision.md`):
   `PYTHONPATH=reid-research/experiments/h8-mechanism-analysis python reid-research/experiments/h8-mechanism-analysis/s5_synthesize.py`

6. **Stage 6** — dispatch validation on seetacloud:
   `PYTHONPATH=reid-research/experiments/h8-mechanism-analysis python reid-research/experiments/h8-mechanism-analysis/s6_validate.py`

## Layout
- `artifacts/` — shared inputs for s2/s3/s4 (gitignored, regen via `extract.py`)
- `figures/` — per-stage figures (gitignored)
- `to_human/` — committed: ANALYSIS.md, EXPERIMENT.md, REPRO.md, s5_decision.md
