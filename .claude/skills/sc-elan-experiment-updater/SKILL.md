---
name: sc-elan-experiment-updater
description: Update experiment records in `sc-elan.md` for the SC-ELAN project. Use when adding a new model variant, a new validation run, or new log-derived metrics, and when section-level consistency is required across Variants (Section 6) and Experimental Results (Sections 7.1-7.7).
---

# SC-ELAN Experiment Updater

## Overview

Apply a deterministic update workflow for `sc-elan.md` so one new experiment is reflected consistently in all dependent sections.
Use this skill whenever a new run is added and the document must remain queryable and internally consistent.

## Workflow

1. Collect source metrics from logs.
- Extract: model name, params, GFLOPs, `all` row (`P/R/mAP50/mAP50-95`), per-class rows, and speed split.
- Record source file names for traceability.

2. Update variant definition in Section 6 when architecture/head changes are new.
- Add a new `Variant N` subsection with: model file, mechanism, expected outcomes, risks, and comparison protocol.
- Skip this step for pure re-runs of an existing variant.

3. Update `7.1 Overall Performance Comparison`.
- Add one row to the overall table.
- Include `Parameters`, `GFLOPs`, `mAP50`, `mAP50-95`, `Speed (ms)`.
- Refresh `Key Observations` so top-line statements match the current best results.

4. Update `7.2 Per-Class Performance Analysis`.
- Add a new `7.2.xx` subsection with the full per-class block.
- Keep numbering continuous and model naming consistent with logs and 7.1.

5. Update `7.3 Inference Performance`.
- Add the run to the main speed table.
- If it belongs to the val4 batch, also add it to `7.3.1 val4 Complete Inference Table`.

6. Update `7.6 Validation Batch Summary` when batch-level comparisons are affected.
- Add the new run to the summary table.
- Refresh `Rigorous Comparison` and `Per-Class Signals` conclusions.

7. Update global conclusions and targets.
- Sync `7.4 Conclusions and Recommendations` with the newest ranking.
- Sync `7.5 Summary and Future Work` causal statements with latest evidence.
- Sync `7.7 Complete Summary` priority roadmap and baseline to beat.
- Keep `Quantitative Targets` aligned with the latest best `mAP50-95`.

## Consistency Rules

- Do not keep conflicting best-model statements across 7.1, 7.4, 7.5, 7.6, and 7.7.
- Ensure metric values are identical wherever the same run is referenced.
- Distinguish historical baselines vs latest batch results explicitly.
- Use exact model identifiers consistently (table rows, subsection titles, and analysis bullets).

## Quick Validation

Run targeted checks after editing `sc-elan.md`:

- `rg -n "^### 7\\.1|^### 7\\.2|^### 7\\.3|^### 7\\.4|^### 7\\.5|^### 7\\.6|^### 7\\.7" sc-elan.md`
- `rg -n "best|mAP50-95|baseline to beat|Quantitative Targets" sc-elan.md`
- `rg -n "^#### 7\\.2\\." sc-elan.md`

If a new best score appears, update all sections that mention the previous best.

## References

- For a full section-by-section checklist, read `references/update-checklist.md`.