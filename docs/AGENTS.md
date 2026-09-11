# AGENTS.md

Instructions for `docs/`; `CLAUDE.md` points here. Follow the root `AGENTS.md` for repository-wide rules, including the pinned prettier command for markdown formatting (`npx prettier@3.8.5 --tab-width 4 --print-width 120 --write` for `docs/**/*.md`).

## Ownership

- Pages live in `docs/en/`; the root `mkdocs.yml` owns navigation and Markdown configuration. Relative `.md` cross-file links are the correct convention in `docs/en/` — production is rendered by the centralized publisher, and the local build intentionally omits production-owned site chrome.
- Shared tables live in `docs/macros/` (`augmentation-args.md`, `export-args.md`, `export-table.md`, `platform-*.md`, ...) and are included into pages such as `docs/en/usage/cfg.md`; edit the macro owner, never copy a table into a consuming page. Jinja default values come from `ultralytics/cfg/default.yaml`.
- `docs/en/reference/` is generated from Python docstrings — API prose belongs in the docstring, not in the reference page. `reference/index.md` is the one hand-written page there.
- Comparison pages are imported from the `ultralytics/docs` GitHub repository during the full build.

## Commands

Run from the repository root. Full validation needs Python 3.10+ and `uv pip install -e ".[dev]"` (installs `zensical` and `minijinja`), plus network access for the comparison-page clone.

```bash
# Regenerate docs/en/reference/ after adding/removing/renaming public APIs (docs.yml runs this and pushes the result)
python docs/build_reference.py

# Prepare the complete docs tree (macros, references, comparison pages) and validate it with `zensical build --strict`
python docs/build_docs.py

# Preview the completed build
python -m http.server --directory site
```

- `build_reference.py` rewrites the tracked reference stubs, deletes orphan stubs, preserves `reference/index.md`, and updates the reference navigation in `mkdocs.yml`. It fails when a parameter type is missing from both the signature and the docstring. Newly created reference files are `git add`ed by `create_markdown()`, so inspect `git diff --cached` as well as `git diff` before committing.
- `build_docs.py` temporarily renders full references and macros into `docs/en` and `docs/macros`, builds `site/`, then restores both directories. `zensical serve` is fine for simple page previews but performs none of that preparation, so it cannot validate macros, references, or comparison pages.
