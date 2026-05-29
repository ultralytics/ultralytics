# h8 — Mechanism Analysis of the R1=0.927 Champion ReID Model

**Question:** What mechanism caps the champion at R1=0.927?

**Outcome (one paragraph, edit after reading s2-s4 findings):**
(Summarise here: the named mechanism, the failure slice it explains, the SOLIDER evidence for it, the training-dynamics evidence for it, and what experiment in Stage 6 tested it.)

## Stage 2 — failure taxonomy (excerpt)

(Paste 2-3 most striking facts from `s2_findings.md`; cite figures by relative path.)

![failure crosstab](../figures/s2/failure_crosstab.png)

## Stage 3 — champion vs SOLIDER (excerpt)

> Caveat: cross-architecture CKA is a localization hint only, not a content explanation.

(Paste 2-3 most striking facts from `s3_findings.md`.)

![margin scatter on W](../figures/s3/margin_scatter_W.png)
![saliency divergence](../figures/s3/saliency_divergence_hist.png)

## Stage 4 — training dynamics (excerpt)

(Paste from `s4_findings.md`.)

![champion saturation fit](../figures/s4/champion_saturation_fit.png)

## Synthesis

See `s5_decision.md` for the scored hypothesis card.
See `EXPERIMENT.md` for the validation result.
