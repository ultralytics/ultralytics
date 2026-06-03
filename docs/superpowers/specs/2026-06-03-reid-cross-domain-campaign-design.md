# ReID Cross-Domain Generalization Campaign — Design

**Date:** 2026-06-03
**Branch:** reid-task-official-clean
**Workspace:** `reid-research/experiments/h13-cross-domain/`
**Compute:** ultra1 (8× RTX PRO 6000 Blackwell), strictly sequential single-GPU runs

## Goal

Market-1501 is saturated (verified SOTA R1 96–98 / mAP 91–95 *without* rerank; our family's
no-rerank L is R1 0.906 / mAP 0.768). The open, unsaturated frontier is **cross-domain /
domain-generalizable (DG) ReID**, where no model achieves both strong in-domain and cross-domain
performance. Our differentiator — a lightweight, deployable single-network family
`yolo26{n,s,m,l,x}-reid` + a LUPerson-NL pretrain pipeline — is positioned to answer the literature's
open question: *do DG gains transfer to a lightweight CNN/YOLO backbone?*

This campaign measures (Phase 1) the existing family's zero-shot cross-domain transfer, then
(Phase 2) whether multi-source pooled retraining with the **existing recipe** (no new losses)
improves generalization. Scope = **Phase 1 + Phase 2 only** (no new DG method / loss functions).

## Non-Goals

- No new loss functions, architecture changes, or DG-specific training methods (deferred Phase 3).
- No cross-box metric comparisons (everything on ultra1, one eval config). Per prior finding,
  ultra1 eval reads ~12pp lower mAP than the old seetacloud headline — never mix.
- Not pushing Market-1501 SOTA further.

## Architecture: two artifacts, one guarantee

The "run all experiments in sequence without stop" requirement is met by separating fallible work
from compute work:

### PREP (interactive, run now with the user)
All networked/fallible steps happen here, where failure is visible and recoverable:
downloads, dataset path reconciles, writing yamls + filename parsers, building pooled training
dirs, and `dataset-check` validation. **Output: `validated.yaml`** — a manifest listing exactly
which datasets and models passed validation. The campaign only ever touches validated entries.

### CAMPAIGN driver (`campaign.py`, unattended)
Pure compute, network-free. Reads `validated.yaml`, runs every stage inside `try/except`
(failure → log + continue to next stage). Idempotent: each completed stage writes a results-row
keyed by (phase, model, fold, target, metric-config); on re-launch, stages whose row already
exists are skipped. Runs under `setsid`/`nohup` on ultra1 with a heartbeat log + PID file.
Strictly sequential — one single-GPU run at a time (8-GPU DDP is known to cost ~1pp on small
data; remaining GPUs idle by the user's choice).

## Environment facts (verified 2026-06-03)

- `datasets_dir` = `/home/rick/.cache/autoresearch`
- Present: `Market-1501-v15.09.15` ✅; `MSMT17_V1` (yaml expects `MSMT17_V2` / `mask_*_v2` layout —
  reconcile in prep); `~/datasets/DukeMTMC-reID` (outside datasets_dir — symlink in).
- Absent (acquire in prep, best-effort): Occluded-Duke, CUHK03, CUHK-SYSU, PRID, GRID, VIPeR, iLIDS.
- cls seeds: `/home/rick/yolo26{n,s,m,l,x}-cls.pt` (all present).
- LUPerson-NL pretrain ckpts: `…/h11_luperson_nl_pretrain/weights/best.pt` (L). Per-size LUPerson
  (or LUPerson→MSMT) checkpoints from the h12 size sweep (`reseed_ft_{n,s,m,l,x}` lineage) are the
  intended per-size Phase-2 seeds — **confirm exact locations in prep**; fall back to
  `cls→LUPerson` for any size whose pretrain ckpt is missing.
- Existing weights: `~/reid_weights/{champion_best.pt, msmt_pretrain_best.pt, tx2_best.pt}`.
  Published family `yolo26{n,s,m,l,x}-reid.pt` — confirm location / auto-download in prep.
- Run command: `PYTHONPATH=/home/rick/ultralytics /home/rick/ultralytics/.venv/bin/python <script>`.
- Remote helper: `python3 reid-research/remote_ultra1.py run "true; <cmd>"` (lead with `true;`;
  login shell is zsh — quote any `===`/`=`-leading echo strings).
- Eval harness: `ultralytics/models/yolo/reid/val.py` reads `val:` (query) + `gallery:` from the
  `data=*.yaml`, supports `reranking`, `scales`, `tta` knobs. Cross-domain eval = run val with a
  different dataset's yaml against a model trained elsewhere. No new eval code needed for the
  standard single-query protocol.

## Datasets

| Dataset | Status | Role | Acquisition |
|---|---|---|---|
| Market-1501 | present | source + in-domain ref | none |
| MSMT17 | present (V1) | source + target | relayout to yaml splits |
| DukeMTMC-reID | present | source + target (deprecated — note caveat) | symlink |
| Occluded-Duke | acquire | eval-only target | download + yaml (reuse Duke parser) |
| CUHK03, CUHK-SYSU | best-effort | eval-only target | download + yaml + parser |
| PRID/GRID/VIPeR/iLIDS | best-effort | eval-only target (DG Protocol-1) | gated/manual; needs 10-split eval |

The three sets with training splits (Market, MSMT17, Duke) are the **leave-one-out training
sources**. All others are **eval-only targets**.

## CAMPAIGN — Phase 1: zero-shot cross-domain eval (no training)

- **Models:** `yolo26{n,s,m,l,x}-reid` (5) — gives a size-vs-generalization curve.
- **Targets:** Market (in-domain reference), MSMT17, DukeMTMC, Occluded-Duke [+ validated small sets].
- **Metrics:** std **and** rerank, mAP **and** R1, in **separate columns** (pitfall fix). BoT
  94.5/85.9 (no rerank) recorded as a baseline reference row.
- ~5 × 4 = 20 fast eval runs → `phase1_crossdomain_results.tsv`.

## CAMPAIGN — Phase 2: multi-source retrain (existing recipe, no new losses)

Leave-one-out DG: seed from the per-size LUPerson-NL pretrain checkpoint, train on **pooled
sources** (folder-per-identity, namespaced PIDs), eval the **held-out target**.

- **Folds:** {MSMT+Duke}→Market, {Market+Duke}→MSMT17, {Market+MSMT}→Duke.
- **Sizes:** all 5 (`n/s/m/l/x`). 3 folds × 5 sizes = **15 training runs**, strictly sequential.
  Each size seeds from **its own** per-size LUPerson ckpt (per the width-mismatch lesson: never
  cross-width seed; fall back to `cls→LUPerson` if a per-size pretrain is missing).
- Each fold's trained model is also evaluated on the eval-only targets (Occluded-Duke, small sets).
- Output: `phase2_multisource_results.tsv`, compared directly against Phase 1 single-source numbers.

## New code inventory (honest)

1. Dataset yamls + filename parsers — Occluded-Duke (certain; reuses Duke format), others best-effort. **Data plumbing.**
2. Multi-source pooling converter (prep) → `combined_<sources>/train/<src>_<pid>/…`. **Data plumbing.**
3. `campaign.py` orchestrator (sequential, try/except, idempotent rows, heartbeat). **Orchestration.**
4. *Only if small sets validate:* 10-split averaged eval protocol — the **one** piece of new eval
   code; skipped entirely otherwise.

No new loss functions or architecture changes.

## Error handling / no-stop guarantees

- Every stage wrapped in `try/except` → log full traceback to `logs/<stage>.log`, append a
  `status=FAILED` row, continue.
- Manifest-driven: only validated datasets/models run; a missing dataset can never halt the run.
- Idempotent re-launch: completed rows are skipped, so a crash + relaunch resumes.
- Driver under `setsid`/`nohup`; heartbeat line every N minutes; PID file for monitoring.
- Strictly sequential single-GPU; no DDP, no concurrency.

## Outputs

- `phase1_crossdomain_results.tsv`, `phase2_multisource_results.tsv`
- `logs/<stage>.log` per stage; `heartbeat.log`; `campaign.pid`
- `REPORT.md`: with/without-rerank columns, BoT reference row, Phase-1-vs-Phase-2 generalization
  delta, DukeMTMC deprecation caveat, and a note on which best-effort datasets were dropped.

## Compute estimate

Phase 1 ≈ minutes–1h (20 fast evals). Phase 2 ≈ 15 sequential training runs × ~1–4h each →
**~1–3 days** unattended. Acceptable given the no-stop design.

## Success criteria

1. Campaign runs end-to-end on ultra1 without manual intervention; any failed stage is logged and
   skipped, not fatal.
2. `phase1` + `phase2` TSVs are complete for every validated (model × target) and (fold × size).
3. `REPORT.md` answers: (a) how far the family's cross-domain transfer falls below in-domain, and
   (b) whether multi-source pooling improves it on a lightweight CNN — with/without rerank reported
   separately and benchmarked against BoT.
