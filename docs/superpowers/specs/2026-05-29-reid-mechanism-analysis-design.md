# ReID Mechanism Analysis — Design Spec

**Date:** 2026-05-29
**Topic:** Diagnostic post-mortem of the saturated R1=0.927 champion ReID model
**Outcome target:** A diagnosed mechanism that explains the R1 ceiling, plus one validated experiment that tests the most promising lever surfaced by the diagnosis.

---

## 1. Context

The Ultralytics ReID research effort has tested nine single-model attack vectors against a champion checkpoint (yolo26l-2psa + MSMT-pretrain + Market FT, imgsz=384, k-reciprocal rerank + flip TTA) and concluded the stack saturates at **R1=0.9267 / mAP=0.8844** on Market-1501. Outstanding levers (LUPerson pretrain, cross-architecture distillation) are either storage-blocked or have already failed in four distillation variants.

This study shifts from *hypothesis-trial* to *post-mortem mechanism analysis*. Inputs are the existing weights and training logs already on disk; outputs are a diagnostic understanding of the ceiling and one validated experiment grounded in that understanding.

**Available weights** (all already staged on the westd analysis box and on seetacloud):
- `champion` — yolo26l-2psa, MSMT-pretrain + Market FT (R1=0.9267)
- `mgn-t3` — MGN head, backbone-only MSMT pretrain + Market FT (R1=0.9219)
- `mgn-t4` — MGN head, full-model MSMT pretrain + Market FT, catastrophic (R1=0.8477)
- `t5fix` — RKD distillation from SOLIDER (R1=0.7530)
- `solider` — SOLIDER Swin-Base teacher reference (R1=0.9680)
- Optional control: `h1-ls1-clean` (last_stride=1 clean baseline, R1=0.9264)

**Compute split:**
- **westd** (1 GPU): all analysis stages (1–5). One GPU is sufficient because nothing trains.
- **seetacloud** (4 GPUs): Stage 6 validation training, 3 seeds dispatched in parallel.

---

## 2. Architecture

The whole study lives in `reid-research/experiments/h8-mechanism-analysis/` with one shared artifact directory all stages read from:

```
h8-mechanism-analysis/
  extract.py              # Stage 1: writes artifacts/ once, never rerun
  s2_failure_taxonomy.py  # Stage 2
  s3_solider_gap.py       # Stage 3
  s4_training_dynamics.py # Stage 4
  s5_synthesize.py        # Stage 5: cross-references s2/s3/s4 -> 1 hypothesis
  s6_validate.py          # Stage 6: dispatches the 3-seed validation on seetacloud
  artifacts/
    extraction_manifest.json
    market_meta.parquet
    {model_tag}/
      embeddings.pt
      retrieval.parquet
      feats_p4.pt
      feats_p5.pt
      saliency/{query_id}.npy
  figures/{s2,s3,s4,s5,s6}/
  to_human/
    ANALYSIS.md       # diagnostic narrative (stages 1-4) with embedded figures
    EXPERIMENT.md     # the one validation run + result + verdict
    REPRO.md          # commit SHA, env versions, dataset checksums, seeds
```

**Artifact contract:** every stage reads `artifacts/{model_tag}/*` and `market_meta.parquet`, never reruns inference, never modifies anything inside `artifacts/`. Re-running a stage is idempotent and cheap.

---

## 3. Stage 1 — Extraction (`extract.py`)

One-time pass over all 5 (or 6) models on Market-1501 query + gallery. For each image: forward pass with hooks capturing `feat_p4` (stride-16 features), `feat_p5` (stride-32 features), and the final BN+L2-normed `embedding`. For each *query* image (~3368), also compute an **Integrated Gradients** saliency map on the P5 feature map, with the target signal being the cosine similarity of the query embedding against its closest correct gallery match (50-step Riemann approximation; clamp/skip-and-log on NaN).

For SOLIDER (Swin-Base, different topology), the analogous taps are block-3 (stride-16-equivalent) and block-4 (stride-32-equivalent). The per-model tap registry is documented in `extract.py`.

**`retrieval.parquet` (one row per query):** `query_id, true_pid, true_camid, top50_gallery_ids, top50_distances, top50_pids, top50_camids, r1, r5, r10, mAP_q`. Junk-filter applied (exclude same-pid-same-cam gallery hits).

**`market_meta.parquet` (one row per Market image, query+gallery):** `image_id, split, pid, camid, img_path, aspect_ratio, mean_brightness, occlusion_score, pid_gallery_count`. `occlusion_score = 1 - person_mask_fraction` from a `yolo11n-seg` segmentation pass.

**Sanity gate (load-bearing):** after extracting champion, recompute R1/mAP from `retrieval.parquet` and assert R1 ∈ [0.925, 0.928]. After SOLIDER, assert R1 ≈ 0.968. Any failure halts the study; usually means a forward-mode or normalization regression.

---

## 4. Stage 2 — Failure taxonomy (`s2_failure_taxonomy.py`)

Scope: the ~246 champion R1-failures (3368 × 7.3%).

**Auto-tagging.** Each failure gets a row with: `cross_camera` flag, `confusion_type` (hard_neg / no_good_match / missing), `margin_to_truth` (signed cosine margin), `occlusion_bin` and `brightness_bin` (quartile-based, normalized over the full query set), `pose_bin` (aspect-ratio quartile, dropped if it doesn't separate failures), and `pid_rarity` (low if pid_gallery_count ≤ 3).

**Cross-tab.** The headline plot is the 2×3 `cross_camera × confusion_type` heatmap. For each binned axis, failure rate per bin (with bootstrap CI on the rate-ratio). Bins with n<20 are marked underpowered, not statistically significant.

**Residual-direction clustering.** For each failure, the *failure embedding residual* = `embedding(query) − embedding(best_true_match_in_gallery)` — the direction the model would need to move to fix it. UMAP → HDBSCAN, sweeping `min_cluster_size ∈ {5,10,20}` and picking the value that leaves <30% noise. If all three leave >50% noise, the residuals don't cluster and the stage reports that honestly, without fabricating a taxonomy.

**Outputs.** `figures/s2/failure_crosstab.png`, `failure_rate_by_{axis}.png`, `residual_umap.png`, `contact_sheet_cluster_{k}.png` (10 worst queries × top-5 retrieved per cluster), and `s2_findings.md` with the written summary.

---

## 5. Stage 3 — Champion-vs-SOLIDER gap (`s3_solider_gap.py`)

**Set construction.** `W` = {q : champion wrong, SOLIDER right} — the winnable set, expected ~138 queries. `S` = both right. `H` = both wrong (the irreducible ceiling). All quantitative claims over `W` report `n` and bootstrap CI. If `|W| < 50`, the stage downgrades to qualitative case study.

**Question 1 — Margin geometry on W.** For each model and each `q ∈ W`: `cos_to_true`, `cos_to_top1_wrong`, `margin = cos_to_true − cos_to_top1_wrong`. Joint scatter `(margin_champion, margin_solider)` over `W`. Small positive SOLIDER margins mean the gap is fragile (small push could close it); large positive margins mean categorically different geometry.

**Question 2 — Where in the network does the representation diverge?** Linear-CKA between champion P4/P5 features and SOLIDER block-3/block-4 features on a 2k-image subsample. Two heatmaps side-by-side (`S` set vs `W` set). **Caveat documented at top of `s3_findings.md`:** cross-architecture CKA (15M CNN vs 88M Swin) is a localization hint only, not a content explanation; Swin tokenization breaks the spatial alignment CNN features have. Interpret as "where they differ", never as "what they encode".

**Question 3 — Where on the image does SOLIDER look that champion doesn't?** For each `q ∈ W`, both models' IG saliency maps from Stage 1, resized to input resolution. Per-query saliency divergence = `1 − cosine_sim(sal_champion.flatten(), sal_solider.flatten())`. Bucket `W` by divergence quartile. Top-quartile contact sheets (manual qualitative tagging — no automatic part-segmenter classifier).

**Cross-reference back to Stage 2.** Tag each `q ∈ W` with its Stage 2 cluster id and failure tags. Cross-tab `W vs Stage 2 clusters` — which named failure modes is SOLIDER actually fixing? This is the bridge that turns "SOLIDER does better" into "SOLIDER fixes cluster K of failure mode Y".

**Outputs.** `margin_scatter_W.png`, `cka_S_vs_W.png`, `saliency_divergence_hist.png`, `contact_sheet_high_divergence.png`, `W_vs_s2_clusters_crosstab.png`, `s3_findings.md`.

---

## 6. Stage 4 — Training dynamics (`s4_training_dynamics.py`)

Inputs are the existing training logs on the remote box and the 285-run `results.tsv` — no new training. The first task of the stage is a **log-shape audit**: enumerate what each run logged, write the gaps to `s4_findings.md`. Missing per-component losses are reported as "no data" cells, never fabricated.

**Runs.** `champion` (Ultralytics trainer, periodic checkpoints), `mgn-t3` and `mgn-t4` (custom Python loop, checkpoint policy verified in audit), `t5fix` (custom loop, only `best.pt` + `last.pt` saved every 50 epochs — no early-FT snapshot), optionally `h1-ls1-clean` as a control.

**Analysis 1 — Loss/R1 decoupling.** Per-run twin-axis plot of loss components and val R1 vs epoch. Mark `epoch_of_max_R1`. Compute **post-saturation slack** = `(epochs_total − epoch_of_max_R1) / epochs_total`. Mark each loss component's own plateau epoch. If CE plateaus early but triplet keeps falling past the R1 peak, metric losses are over-fitting train margins; if triplet plateaus first, hard-mining is exhausted.

**Analysis 2 — MSMT pretrain transfer.** Take the MSMT-pretrain endpoint checkpoint for each pretrained run; extract embeddings on Market query+gallery directly (no FT); compute zero-shot R1/mAP. Then track R1/mAP evolution over FT epochs. The slope from `pretrain_zero_shot_R1` to `final_R1` quantifies what FT actually adds vs what pretrain donated. For mgn-t4, this should expose pathologically MSMT-overfit pretrain endpoint, confirming the report's hypothesis quantitatively.

**Analysis 3 — t5fix dominance check.** Static evidence (load-bearing): the loss formula `loss = reid_l + 50.0 * distill_loss` in `t5fix_distill.py` is itself strong evidence of distillation dominance. Corroborating evidence (sanity-only): on the final `t5fix` checkpoint, one forward+backward on a Market batch, log per-loss-term gradient L2 norm at the head and at the backbone's last conv. If `distill_rkd_grad ≫ ce_grad + triplet_grad`, dominance is confirmed at convergence. If the gradient pattern is reversed (post-convergence equilibrium), we rely on the static formula evidence and note the equilibrium in `s4_findings.md`.

**Analysis 4 — Champion saturation fit.** Fit `R1(epoch) = R1_∞ − A·exp(−epoch/τ)` to champion's R1 trajectory (smoothed with 10-epoch moving average). If R² < 0.7, the fit is unreliable and we report only the raw curve and the epoch of max R1.

**Outputs.** `figures/s4/loss_r1_decoupling_{run}.png` × N, `pretrain_transfer_table.md`, `t5fix_grad_attribution.png`, `champion_saturation_fit.png`, `s4_findings.md`.

---

## 7. Stage 5 — Synthesis (`s5_synthesize.py`)

Stages 2–4 each surface candidate hypotheses for what's actually capping R1. Stage 5 scores them on a 4-cell card and selects exactly one for validation.

| Cell | Source | Test |
|---|---|---|
| **Failure slice attacked** | Stage 2 | Which named cluster does it target? What fraction of the 7.3% champion gap does that cluster represent? |
| **SOLIDER mechanism aligned** | Stage 3 | Does SOLIDER's saliency/margin pattern on `W` match what this hypothesis would produce in the student? |
| **Recipe-side feasibility** | Stage 4 | Does training-dynamics evidence say the lever is reachable — unused post-saturation epochs, a still-descending loss term, a known dominance pattern to flip? |
| **Cost on seetacloud (4 GPUs)** | implementation | GPU-hours, dataset prep, code surface. Must fit one 3-seed validation in reasonable budget. |

**Selection rule.** Highest `gap-fraction × mechanism-alignment`, subject to fitting the validation budget. Ties break toward the cheapest test.

**Output — `s5_decision.md`** (committed before `s6_validate.py` starts):
- Named hypothesis
- Predicted R1 lift band (e.g. "+1.0pp to +2.5pp median")
- Explicit falsification condition
- Frozen recipe diff vs champion (as a unified diff against the champion config yaml)
- The second-best hypothesis named as next-round candidate

**Candidate hypothesis types** (the data picks; we do not pre-commit):
- Targeted augmentation against the dominant failure cluster (e.g. occlusion-CutMix if occlusion dominates `W`)
- Loss recipe shift (harder triplet mining / weighted sampling)
- Architectural micro-change motivated by saliency (e.g. an attention block at the layer where Stage 3 CKA diverges)
- Resampling-only retrain if Stage 4 shows champion saturated under its current data distribution

---

## 8. Stage 6 — Validation (`s6_validate.py`)

Exactly one training experiment, executed on **seetacloud (4 GPUs, 3 seeds in parallel)**.

**Environment-drift baseline.** Before validation seeds start, re-run champion at one seed on seetacloud's current image. Assert R1 ∈ [0.925, 0.928]. If not, environment has drifted since the published 0.9267 and the comparison is halted until reconciled.

**Recipe diff.** One change from champion, frozen by `s5_decision.md`'s unified diff. No co-changes.

**Seeds.** 3 seeds — champion's original seed + 2 new. Headline = median R1. Per-seed table and bootstrap CIs reported in `EXPERIMENT.md` for transparency.

**Eval.** Identical to champion: `val.py` with imgsz=384, flip TTA, k-reciprocal rerank. No new eval-side tricks (those would be a confound for the recipe change).

**Pass criterion (frozen before run).** `median(new R1) > median(champion R1)`. Documented in `s5_decision.md`. Diffing the criterion after seeing results is out of bounds; if we want to re-examine, that's a new hypothesis in a follow-up plan.

**Compute cap.** Each seed occupies one GPU; the 3 seeds run on 3 of seetacloud's 4 GPUs simultaneously, so wall-clock ≈ one champion-equivalent run (~10h). The 4th GPU stays free for the environment-drift baseline above. If predicted recipe is more expensive than champion, cut to 2 seeds with explicit caveat, or run in the imgsz=288 / ~200-epoch proxy regime the 285-run frontier uses.

**Output — `to_human/EXPERIMENT.md`:**
- Frozen recipe diff
- 3-seed table (R1, mAP per seed)
- Median delta vs champion baseline + 95% bootstrap CI
- Signed verdict: **confirmed / null / refuted**, by the pre-frozen criterion
- Second-best hypothesis named for next round

---

## 9. Error handling & reproducibility

**Stage 1.**
- Embedding-normalization regression / junk-id handling: caught by the champion R1 ∈ [0.925, 0.928] assert.
- SOLIDER topology: caught by SOLIDER R1 ≈ 0.968 assert.
- IG NaN on zero-activation channels: skip-and-log; warn if >1% of queries are bad (means integration baseline is wrong).
- Seeds: extraction is deterministic, but versions and recipe params are logged in `artifacts/extraction_manifest.json`.

**Stage 2.**
- HDBSCAN noise floor: sweep `min_cluster_size ∈ {5,10,20}`, pick <30% noise, otherwise report no taxonomy.
- Small per-bin counts: bootstrap CI; bins with n<20 marked underpowered.

**Stage 3.**
- Small `W`: report `n` and CI on all quantitative claims; downgrade to qualitative if `|W| < 50`.
- CKA pitfall: caveat documented at top of `s3_findings.md`, interpreted as localization only.

**Stage 4.**
- Log gaps: per-run audit first, missing data reported as "no data" cells.
- Saturation fit failure (R² < 0.7): report raw curve, no derived `τ`.
- t5fix grad attribution may show equilibrium pattern at convergence; static formula evidence (`DISTILL_W=50.0`) is the load-bearing claim, gradients are corroborating only.

**Stage 6.**
- Environment-drift assert before any new training.
- Mid-run crash: standard checkpoint-resume; surviving seeds reported with explicit crash note.
- "Moved goalposts": pass criterion committed before run; no post-hoc diffing.

**Reproducibility — `to_human/REPRO.md`:**
- Exact commit SHA, env versions (torch, cuda, cudnn, ultralytics), dataset checksums
- Model checkpoint SHA256s
- Random seeds for Stage 6
- Per-stage runtime + GPU model

---

## 10. Out of scope

- LUPerson(-NL) pretrain (storage-blocked; if it surfaces as Stage 5's top hypothesis, becomes a follow-up plan)
- New cross-architecture distillation variants (four already negative; we are not running a fifth)
- Architecture sweeps beyond what Stage 5's hypothesis specifies (single recipe diff only)
- Ensembling or test-time tricks (single model, single forward pass per the project constraint)
- The next-round experiment after Stage 6 (named in `EXPERIMENT.md` but not executed in this plan)
