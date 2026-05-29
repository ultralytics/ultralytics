"""Stage 5 of h8 — synthesize findings, draft s5_decision.md for human review.

The script's job is to pull together the quantitative facts from s2/s3/s4 into a
side-by-side scoring table for each candidate hypothesis the data has surfaced.
A human (you) reads the draft, picks the winning hypothesis, fills in the
predicted R1 lift band, falsification condition, and recipe diff.

The script does NOT auto-select. It auto-summarises.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
OUT = ROOT / "to_human"


def _load_findings(stage: str) -> str:
    p = OUT / f"{stage}_findings.md"
    return p.read_text() if p.exists() else f"(no {stage}_findings.md)"


def _failure_total(champ_retr: pd.DataFrame) -> int:
    return int((champ_retr["r1"] == 0).sum())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    champ = pd.read_parquet(ART / "champion" / "retrieval.parquet")
    n_fail = _failure_total(champ)

    s2 = _load_findings("s2")
    s3 = _load_findings("s3")
    s4 = _load_findings("s4")

    md = f"""# s5 — Synthesis (DRAFT, requires human selection)

Champion R1=0.9267, mAP=0.8844; failure count = {n_fail} queries.

## Inputs
- `s2_findings.md` — failure taxonomy (clusters, slice sizes)
- `s3_findings.md` — champion-vs-SOLIDER gap on W set
- `s4_findings.md` — training-dynamics post-mortem

## Hypothesis scoring rubric (fill in per candidate)

For each candidate, fill these cells:

| Hypothesis | Failure-slice attacked | Slice size (% of {n_fail}) | SOLIDER-mechanism aligned? | Recipe-side lever | Cost (seetacloud GPU-h) |
|---|---|---|---|---|---|
| (e.g.) occlusion-CutMix | high-occlusion failure cluster | TBD | yes if SOLIDER routes around occluders in `figures/s3/saliency_divergence_hist.png` | data-side aug; no arch change | ~30 |
| (e.g.) harder triplet mining | hard_neg_distractor slice of confusion_type | TBD | partial — addresses margin geometry not saliency | loss-side; uses post-saturation epochs | ~30 |
| ... | | | | | |

## Selection rule

Highest `slice_size × mechanism_alignment` subject to budget. Ties break to cheapest test.

## Final selection (HUMAN EDITS THIS)

- **Hypothesis name:** (fill in)
- **Predicted R1 lift band:** (e.g. +1.0pp to +2.5pp median)
- **Falsification condition:** what observation would refute it?
- **Recipe diff vs champion (unified diff against the champion yaml):**

```diff
(fill in unified diff here)
```

- **Second-best hypothesis (next-round candidate):** (fill in)

## Findings references

### s2 ↓
{s2}

### s3 ↓
{s3}

### s4 ↓
{s4}
"""

    (OUT / "s5_decision.md").write_text(md)
    print(f"Draft written to {OUT / 's5_decision.md'}. EDIT IT before running Stage 6.")


if __name__ == "__main__":
    main()
