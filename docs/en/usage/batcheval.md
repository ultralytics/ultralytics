---
comments: true
description: Use `yolo batcheval` to evaluate multiple YOLO models on the same dataset split and export a unified metrics summary.
keywords: Ultralytics, YOLO, batcheval, batch evaluation, model comparison, metrics summary
---

## yolo batcheval

`yolo batcheval` evaluates multiple YOLO models on the same dataset split and writes a unified metrics summary as CSV.

This is a lightweight helper built on top of the public `YOLO().val(...)` API. It is designed for quick, local model
comparison rather than long-term experiment tracking.

### Python

```python
from ultralytics.analytics import batcheval

results = batcheval(
    models=["runs/detect/train38", "models/exp2.pt"],
    data="coco8.yaml",
    split="val",
)

for r in results:
    print(r.model_name, r.metrics)
```

### CLI

```bash
# Compare two specific models on the same dataset split
yolo batcheval models="runs/detect/train38,models/exp2.pt" data=coco8.yaml split=val

# Compare all runs under runs/detect/ using a glob
yolo batcheval models="runs/detect/train*" data=coco8.yaml split=test

# With an optional confidence-threshold sweep
yolo batcheval \
  models="runs/detect/train38,models/exp2.pt" \
  data=coco8.yaml \
  split=test \
  sweep_conf=True \
  conf_min=0.05 \
  conf_max=0.95 \
  conf_step=0.05
```

### Outputs

By default, outputs are written under `runs/batcheval/<timestamp>/`:

- `summary.csv`: one row per model with flattened scalar metrics from `YOLO().val(...)`.
- `sweep.csv`: optional, created when `sweep_conf=True`, with metrics across confidence thresholds.

For details on the metrics reported by validation, see the validation documentation and `results.results_dict` in
`ultralytics.utils.metrics`.
