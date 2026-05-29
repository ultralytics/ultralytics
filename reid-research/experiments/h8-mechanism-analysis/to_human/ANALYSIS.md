# h8 Mechanism Analysis — Findings (no-IG, no-occlusion run)

**Pipeline:** Stage 1 extraction on westd, 4 models (champion / mgn-t3 / mgn-t4 / solider — t5fix not on this box). Saliency and occlusion-segmentation skipped to keep wall-clock manageable. Stage 2 (failure taxonomy) and Stage 3 (champion-vs-SOLIDER gap) completed. Stage 4 attempted but Ultralytics CSVs only logged val at the final epoch — only loss-plateau analysis usable.

## Reproduced metrics (no TTA, no rerank)

| Model | R1 | R5 | R10 | mAP |
|---|---|---|---|---|
| champion (yolo26l-2psa, arch31) | 0.8996 | 0.9656 | 0.9795 | 0.7404 |
| mgn-t3 (MGN head, backbone-only MSMT pretrain) | 0.8928 | 0.9644 | 0.9777 | 0.7345 |
| **mgn-t4** (MGN head + full MSMT pretrain) | **0.7999** | 0.9213 | 0.9504 | 0.5792 |
| **solider** (Swin-Base teacher) | **0.9682** | 0.9893 | 0.9932 | 0.9355 |

Headlines:
- Champion matches Ultralytics' built-in `m.val()` to 4 decimals — pipeline verified.
- SOLIDER R1 = 0.9682 matches published 0.968 — sanity gate passed.
- **mgn-t4 confirmed catastrophic (-10pp R1 vs champion)** — full-model MSMT pretrain over-fits MSMT identities and Market FT can't recover.
- Champion-SOLIDER gap = **6.86pp R1** at no-TTA (published 4.1pp with TTA+rerank).

## Stage 2 — Failure taxonomy on champion's 338 R1-failures

**Cross-tab `cross_camera × confusion_type`:**

| | hard_neg_distractor | no_good_match |
|---|---|---|
| same_cam | 131 | 9 |
| cross_cam | 187 | 11 |

Two load-bearing observations:

1. **94% of failures are `hard_neg_distractor`** (318/338) — the true PID *exists* in the model's top-50, but a wrong PID was retrieved at top-1. This is a **fine-grained discrimination problem**, not a "missing match" problem. The retrieval candidate set is fine; the ranking head is misordering near-duplicates.
2. **58% cross-camera, 42% same-camera failures.** Cross-camera dominates but same-cam failures are surprisingly prevalent — they shouldn't be "easy" cases, so the model is genuinely confused on perceptually-near gallery items even within a single camera view.

**Residual-direction clustering (UMAP → HDBSCAN, min_cluster_size=5, 29.6% noise):** 3 clusters. Cluster 2 contains the bulk of the failures. Brightness/aspect-ratio bins show flat ~9-11% failure rate — neither axis stratifies failures meaningfully.

## Stage 3 — Champion vs SOLIDER on the winnable set

**Set construction:**
- |W| = **261** (champion-wrong ∧ SOLIDER-right) — the "winnable" recoverable failures
- |S| = 3000 (both right)
- |H| = **77** (both wrong) — the irreducible Market-1501 ceiling for this query set

So **77% (261/338) of champion's R1-failures are recoverable**: SOLIDER demonstrates they're not intrinsic.

**Margin geometry on W (figures/s3/margin_scatter_W.png):**
- Champion margin median = **−0.0379**, **97% of W queries have negative champion margin**
- Champion margin range: [−0.25, 0]; mass concentrated in [−0.10, 0]
- SOLIDER margin = 0 (by construction — SOLIDER's top-1 IS the true match on this set)

**This is the most important finding of the study.** The 261 recoverable failures are *near-misses* — champion ranks the true match at a cosine distance only ~0.04 worse than the wrong top-1. A tiny calibration push (~0.05 cosine units) could close most of them. The geometry is not catastrophically wrong; it's slightly miscalibrated on a specific, identifiable slice.

**CKA on S vs W (caveat: cross-architecture, localization-only):**

| | sol_stage3 | sol_stage4 |
|---|---|---|
| champ_p4 (S / W) | 0.33 / 0.36 | 0.21 / 0.27 |
| champ_p5 (S / W) | 0.49 / 0.52 | 0.40 / 0.45 |

CKA is *slightly higher on W than on S* at every stage pair. Interpretation: champion and SOLIDER produce more-similar representations on the hard queries than on the easy ones — i.e., champion's failures aren't from a representation that's wildly different from SOLIDER's. The disagreement that matters lives in the small last-mile of the embedding head, not in the backbone alignment. Consistent with the small-margin finding.

**Bridge to Stage 2 clusters:**

| Stage-2 cluster | W-queries in cluster |
|---|---|
| -1 (noise) | 76 |
| 0 | 5 |
| 1 | 5 |
| **2** | **167** |

**64% of winnable failures (167/261) live in Stage-2 cluster 2** — a single dominant failure-residual direction. This is the highest-leverage attack target: if cluster 2 has a coherent visual signature, a single intervention could recover ~50% of champion's gap to SOLIDER.

## Stage 4 — Training dynamics (partial)

Ultralytics' results.csv only logged val metrics at the final epoch, so we cannot trace R1 saturation. The loss-plateau analysis still surfaces a consistent pattern:

| Run | epochs | CE plateau | Triplet plateau | Post-plateau slack |
|---|---|---|---|---|
| champion | 635 | ~69 | ~68 | 89% |
| mgn-t3 | 635 | ~138 | ~146 | 78% |
| mgn-t4 | 635 | ~139 | ~119 | 78% |

All three runs spent **78–89% of their epoch budget after CE + triplet losses had effectively converged**. With cosine LR decaying to ~3.5e-6 by the end, those tail epochs are slow fine-tuning of the embedding manifold without further objective progress. We don't have per-epoch val to tell whether R1 actually kept improving in this tail or just held — but the very small margins we see in Stage 3 suggest the tail was *not* fully exploited for discriminative calibration.

## Synthesis — the actionable mechanism

Joining the three pieces:

1. Champion's R1=0.8996 leaves 338 failures. 261 are recoverable, 77 are intrinsic.
2. The 261 recoverable failures are **fine-grained near-misses** with a tiny but systematic negative margin (median −0.04 cosine, 97% negative).
3. 64% of those 261 collapse into a single Stage-2 cluster (cluster 2) — one dominant failure mode, not a long tail.
4. Backbone representations are reasonably aligned with SOLIDER's (CKA 0.40–0.52 at deep stages); the last-mile head is where champion misorders.

**Implication:** A small recipe-side intervention targeted at fine-grained ranking — *not* an architectural change, *not* a backbone swap, *not* more pretrain data — should attack the bulk of the gap. Two candidates worth a controlled experiment:

- **Harder triplet mining with smaller-margin pairs.** The plateau of triplet loss at epoch ~70-140 with 600+ post-plateau epochs to spare means hard-mining could be re-engaged in a second-phase fine-tune, specifically constructing pairs in the [−0.10, 0] margin band where 97% of W queries live. This directly targets the near-miss geometry.
- **Re-ranking on a smaller candidate set with a stronger metric.** Since the true PID is in top-50 in 94% of failures, a second-stage re-ranker (e.g. a small MLP or attention head trained specifically to discriminate the top-50 candidates) might recover most failures without retraining the full backbone.

Both can be tested in <10 GPU-hours each on seetacloud. Either way, **the lever is calibrating the head, not capacity-adding or data-side**.

## Caveats

- No IG saliency was extracted, so cluster 2's visual mechanism is inferred from numbers alone (contact sheets are coarse). A targeted IG pass on the 167 cluster-2 queries (~30 min) would resolve whether SOLIDER attends to a specific body region champion ignores.
- Occlusion segmentation was skipped (yolo11n-seg auto-download failed on westd). The occlusion-bin axis of the failure taxonomy is empty — we can't quantify the occlusion stratification.
- Per-epoch val wasn't logged during champion training, so we can't say at what epoch R1 saturated. A re-run with periodic val would close this.
- t5fix not on this box — distillation post-mortem (Stage 4 Analysis 3) skipped.
