# SC-ELAN New-Experiment Update Checklist

## Input Requirements

- Log filename
- Model summary line (parameters, GFLOPs)
- Final `all` line: P, R, mAP50, mAP50-95
- Full per-class table
- Speed line (preprocess/inference/postprocess/total)

## Required Document Updates

1. Section 6 (if new design)
- Add variant definition and hypothesis.
- Add risk points and expected outcomes.

2. Section 7.1
- Add overall row.
- Refresh `Key Observations`.

3. Section 7.2
- Add a new numbered subsection with full per-class results.

4. Section 7.3
- Add speed row in main table.
- Add speed row in 7.3.1 if val4 batch.

5. Section 7.4
- Re-rank recommendations by latest evidence.

6. Section 7.5
- Update narrative causal analysis to avoid stale claims.
- Update quantitative targets.

7. Section 7.6
- Add run into batch table.
- Update rigorous comparison bullets.

8. Section 7.7
- Update architecture/head conclusions and next-step priorities.

## Final Consistency Pass

- Check old best score references (e.g., `0.208`) are updated if superseded.
- Check model names are identical across all sections.
- Check no duplicated or broken markdown fences.
- Check 7.2 numbering remains continuous.
