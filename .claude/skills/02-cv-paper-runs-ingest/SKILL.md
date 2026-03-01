---
name: 02-cv-paper-runs-ingest
description: Discover and parse experiment runs for cv-paper-agent with resume-safe incremental caching. Use when asked to scan runs directories, apply marker rules, parse results.csv and args.yaml, compute fingerprints, update experiments.jsonl/fingerprints.json/state.json, or debug resume behavior.
---

# 02 CV Paper Runs Ingest

Use this skill to maintain S1/S2 data ingestion from the bundled copy at `assets/cv-paper-agent/`.

## Bundled Runtime

1. Treat `assets/cv-paper-agent/` as the primary runtime source for this skill.
2. If edits are requested, modify files under the bundled copy or a user-requested copy destination created from it.
3. Do not rely on repository-root `cv-paper-agent/` paths.

## Execute

1. Use `configs/runs_layout.yaml` marker rules to discover experiment dirs recursively.
2. Treat experiment id as run-relative POSIX path.
3. Compute fingerprint from relative path + `results.csv` and `args.yaml` metadata (`size`, `mtime`).
4. On `--resume`:
   - Reuse cached row when fingerprint unchanged.
   - Re-parse and replace row when fingerprint changed.
5. Parse:
   - `results.csv` with pandas and schema mapping from `configs/runs_schema.yaml`.
   - `args.yaml` as dict.
   - PNG/JPG artifacts list with relative POSIX paths and sizes.
6. Persist incrementally:
   - `workspace/workdir/fingerprints.json`
   - `workspace/workdir/cache/experiments.jsonl`
   - `workspace/workdir/state.json`
   - Use atomic write for JSON/JSONL replacements.

## Verify

1. Running twice with `--resume` must not duplicate experiments in JSONL.
2. Touching/changing one `results.csv` must only refresh that experiment row.
3. Serialized paths must use `/` separators.
