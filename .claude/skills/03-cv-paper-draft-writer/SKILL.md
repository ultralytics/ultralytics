---
name: 03-cv-paper-draft-writer
description: Generate and incrementally update LaTeX draft outputs from cached experiment data in cv-paper-agent. Use when asked to draft or refresh paper_project content, especially sections/method.tex and sections/experiments.tex tables from discovered experiments.
---

# 03 CV Paper Draft Writer

Use this skill to maintain S3/S4 draft generation from the bundled copy at `assets/cv-paper-agent/`.

## Bundled Runtime

1. Treat `assets/cv-paper-agent/` as the primary runtime source for this skill.
2. Copy this bundled project to a target workspace when the user requests standalone execution.
3. Do not rely on repository-root `cv-paper-agent/` paths.

## Execute

1. Bootstrap `workspace/paper_project` from `templates/latex/` on first run.
2. Ensure output structure exists: `sections/`, `tables/`, `figures/`.
3. Regenerate `sections/experiments.tex` from cached experiments:
   - Include experiment count.
   - Render a LaTeX table with experiment path, selected metric, best value.
   - Use `N/A` for missing values.
4. Regenerate `sections/method.tex` as placeholder using available method fields.
5. Keep writes deterministic and idempotent across reruns.

## Verify

1. Confirm outputs:
   - `workspace/paper_project/sections/experiments.tex`
   - `workspace/paper_project/sections/method.tex`
2. Confirm no fabricated metrics; missing values remain `N/A`.
3. Keep compatibility with IEEE template imports in `main.tex`.
