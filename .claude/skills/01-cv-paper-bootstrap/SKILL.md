---
name: 01-cv-paper-bootstrap
description: Bootstrap and maintain the cv-paper-agent project skeleton, configs, and LaTeX templates for offline CV paper drafting. Use when asked to initialize the agent structure, sync template files, fix template encoding/compatibility, or prepare workspace layout before parsing runs.
---

# 01 CV Paper Bootstrap

Use this skill to set up and maintain the project baseline from the bundled copy at `assets/cv-paper-agent/`.

## Bundled Runtime

1. Treat `assets/cv-paper-agent/` as the primary runtime source for this skill.
2. If the user wants a working project in another location, copy `assets/cv-paper-agent/` to the target path first, then edit the copied files.
3. Do not depend on repository-root `cv-paper-agent/` paths when executing this skill.

## Execute

1. Ensure required layout exists: `configs/`, `templates/latex/`, `src/cv_paper_agent/`, `workspace/`.
2. Keep `.gitignore` rule to retain only `workspace/.gitkeep` in git.
3. Keep LaTeX templates in `templates/latex/` compilable:
   - Keep `main.tex` plus `sections/*.tex`.
   - Keep `IEEEtran.cls` available in the template root when IEEE style is used.
   - Save all `.tex` as UTF-8 **without BOM**.
4. Keep `pyproject.toml` dependencies minimal (`pyyaml`, `pandas`, `matplotlib`, `pillow`).
5. Avoid adding network-dependent or training-execution behavior.

## Verify

1. Confirm `templates/latex/main.tex` and all section files exist.
2. Byte-check `.tex` headers to ensure no BOM (`EF BB BF`).
3. Confirm project still runs from the copied bundle with `python -m cv_paper_agent --help`.
