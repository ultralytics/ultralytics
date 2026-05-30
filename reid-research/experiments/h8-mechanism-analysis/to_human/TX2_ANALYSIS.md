# h8 follow-up — Re-analyzing TX2 (imgsz=448) under h8 stages

After h9's two null results, we re-extracted TX2's embeddings (the only confirmed
positive lever in the wider research, +0.44pp R1 with TTA+rerank) using the h8
pipeline and compared its failure structure to champion's.

## Headline metrics (no TTA, no rerank)

| Model | R1 | R5 | R10 | mAP | imgsz |
|---|---|---|---|---|---|
| champion | 0.8996 | 0.9656 | 0.9795 | 0.7404 | 384 |
| **tx2** | **0.9044** | 0.9679 | 0.9813 | 0.7467 | **448** |
| solider | 0.9682 | 0.9893 | 0.9932 | 0.9355 | 384 (Swin) |

TX2 = champion + 0.48pp R1 at no-TTA — consistent with the +0.44pp R1 reported
in `to_human/REPORT.md` for TX2 at TTA+rerank.

## Joint cross-tab (champion R1 × tx2 R1)

| | tx2 wrong | tx2 right |
|---|---|---|
| **champion wrong** | 204 (both fail) | **134 (FIXED by tx2)** |
| **champion right** | **118 (NEWLY BROKEN by tx2)** | 2912 (both succeed) |

Net delta: 134 − 118 = +16 queries = +0.48pp. **TX2 is not a refinement of
champion — it's a sideways shift.** 252 out of 3368 (7.5%) of all queries
change classification between the two models. The two embedding spaces are
genuinely different on a meaningful subset.

## TX2 vs SOLIDER on the new W/H sets

| | tx2 | champion | delta |
|---|---|---|---|
| \|W\| (model wrong, sol right) | 240 | 261 | −21 |
| \|H\| (both wrong) | 82 | 77 | +5 |
| total failures | 322 | 338 | −16 |

TX2 fixed **48.7% (127/261) of champion's W set** — half the recoverable
failures that motivated h9. The other half are still failing in TX2.

## TX2's failure structure

`hard_neg_distractor`: 301/322 (93%) — same fine-grained ranking pattern as
champion. The qualitative failure mode didn't change.

Cross-camera: 200/322 (62%) — slightly more than champion's 58%.

## TX2 margin on its W set (n=232)

| | tx2 | champion |
|---|---|---|
| median margin | −0.0321 | −0.0379 |
| frac in [−0.10, 0] | 86% | 97% |
| frac < −0.10 | 14% | 3% |

The remaining TX2 failures are **harder** (4× more queries below −0.10 margin
compared to champion's W). TX2 "ate" the easy near-miss failures and what's
left are harder cases. This is consistent with the picture: imgsz scaling
fixes the easiest geometry-near-miss failures, but harder cases need a
different lever.

## Bridge to champion's Stage-2 clusters

Of the 126 TX2-W queries that also appeared in champion's failure-residual
clusters: 81 still sit in cluster 2 (champion's dominant 167-query cluster),
42 in noise, 3 in clusters 0/1. So **51% of cluster 2 was fixed by TX2** —
cluster 2 contains both "easy" failures (fixed by resolution) and "hard"
failures (still failing).

## The ensemble check — the surprising lever

Quick zero-cost test: average / concat champion's and TX2's embeddings on
the same Market test set.

| Method | R1 | mAP | Δ R1 |
|---|---|---|---|
| champion alone | 0.8996 | 0.7404 | — |
| tx2 alone | 0.9044 | 0.7467 | +0.0048 |
| **sum + L2 (512d)** | **0.9139** | 0.7658 | **+0.0143** |
| **concat + L2 (1024d)** | **0.9243** | 0.7761 | **+0.0247** |

**This is the most informative result so far.** Two models from the same
architecture, trained at different imgsz, produce embedding spaces that are
**nearly complementary on the failure region**. Concat preserves both spaces
losslessly; the resulting R1=0.9243 is +2.47pp over champion (no TTA, no
rerank) — and with TTA+rerank would likely cross R1=0.95.

Two important constraints:
- This concat trick uses **two separate model forward passes** — it violates
  the project's "single-model only" rule (see `memory/reid-single-model-only.md`).
- But the IMPLICATION is single-model-able: a single network that internalizes
  both 384-scale and 448-scale geometries should be able to capture this gain
  at single-shot inference cost.

## What the analysis says about next experiments

1. **Multi-scale training: train ONE model on randomly-chosen imgsz ∈ {384, 448}
   per batch.** If the model internalizes both scale-specific geometries, R1
   could match the concat ensemble — at single-shot inference. The most direct
   route to the +2.47pp without breaking the single-model constraint.
2. **Two-head architecture.** A single backbone with two parallel heads (one
   per scale) emitting embeddings averaged at inference. Single .pt, but
   architectural change.
3. **Self-distillation.** Use the concat ensemble's embeddings as a teacher
   signal for a new student model. Same-architecture teacher → student
   (champion architecture, distilled from champion+tx2 concat embeddings) —
   avoids the cross-arch failure mode of the T5 series. The student is a
   single model.
4. **Cluster-2 hard core (81 queries).** The half of cluster 2 that TX2
   couldn't fix is the most-resistant failure mode. Needs IG saliency or
   manual inspection to characterize.

## Caveats

- TX2's `ty3_tx2_seed1` was running on seetacloud during this analysis; we
  used the original TX2 (seed 0) only.
- No TTA / no rerank in these numbers. Full headline R1 (TTA+rerank) for
  ensemble untested.
