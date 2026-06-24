# CLAUDE.md

**Skills:** Always invoke `/yoloa` and `/expman` at session start (before any other work).

Branch `yolo_anomaly`: training-free anomaly detection on top of YOLO/YOLOE checkpoints via a memory bank of normal features. No gradient training.

## Principles

- **Less is more.** Smallest diff that solves the task. No speculative abstractions, no comments restating code, no new files when an edit suffices.
- **Match the Ultralytics style.** Google-style docstrings, 4-space indent, type hints, terse one-line summaries, `LOGGER` over `print` in library code, snake_case, no emojis in code.
- **Push back, don't just execute.** Don't fully trust the stated direction. Surface fresh ideas grounded in recent anomaly-detection / representation-learning research or new foundation models when they're a credible alternative — even unprompted. Flag when the current plan looks dominated by a newer approach.
- **Deployability over flash.** Goal is a practical, easy-to-deploy detector. Prefer simplifying the current pipeline over adding fancy components; remove knobs that don't earn their weight.

## Code map

| File | Symbols |
| --- | --- |
| `ultralytics/models/yolo/model.py` | `YOLOAnomaly`, `AnomalyValidator`, `AnomalyPredictor` |
| `ultralytics/nn/tasks.py` | `YOLOAnomalyModel(DetectionModel)` |
| `ultralytics/nn/modules/head.py` | `AnomalyDetection(Detect)` |
| `ultralytics/anomaly_utils.py` | `MVTEC_CATEGORIES`, `get_mvtec_yolo_data`, `build_ad_model`, `get_arguments`, `collect_images`, `save_heatmap_overlay` |
| `scripts/yolo_anomaly.py` | thin re-export + `iter_predict`, `iter_predict_heatmap` (VS Code dev tools) |
| `scripts/val_mvtec.py` | MVTec val (plain YOLO + YOLOAnomaly), `--category <X>` or `--all`. Uses `ultra_ext.yoloa.val_plain_yolo` / `val_yoloa`. |
| `scripts/val_dagm.py` / `scripts/val_dagm_yoloa.py` | Same pattern on `DAGM_yolo/Class*/data.yaml`. |

## Flow

```python
m = YOLOAnomaly("yolo26l.pt")
m.setup(names=["anomaly"])              # nc=1 cosine head; ["detect"] = original classifier
m.set_anomaly_args(feature_mode="per_level", active_layers=[1, 2])  # or "fused_heatmap"
m.load_support_set(good_imgs)           # builds + freezes bank, no backprop
m.predict(test_imgs); m.val(data=yaml)  # AnomalyValidator adds image/pixel AUROC
m.save(path)                            # round-trips bank via _restore_anomaly_metadata
```

## Direction

- **Baselines.** Plug in external anomaly models (e.g. PatchCore on ResNet50) under the same data config and metrics so YOLOAnomaly can be compared apples-to-apples.
- **Learned bank.** Explore replacing the memory bank with a small MLP trained on normal features — likely better accuracy and far cheaper at inference than nearest-neighbor over a growing bank.

## Conventions

- `YOLOAnomalyModel.train()` raises — training-free by contract.
- New anomaly knobs: thread through `AnomalyDetection.set_anomaly_args` → `YOLOAnomaly.set_anomaly_args` (model.py ~L963).
- Cached per-category models: `runs/temp/{category}_{base_stem}_anomaly_model.pt`.
- `ultralytics/anomaly_utils.py` uses 4-space indent (library); `scripts/yolo_anomaly.py` keeps tabs (dev scripts).
- Paths: MVTec at `/Users/louis/workspace/ultra_louis_work/buffer/MVTEC/MVTec-YOLO`; sibling lib `ultra_ext` at `../ultra_ext_lib/`.
