# Infrared YOLO Baseline Notes

This note documents the infrared small-target baseline scripts added to this repository.

## Added Scripts

### `mask2yolo.py`

Converts binary mask images into YOLO detection labels.

Supported dataset layout:

```text
NUAA-SIRST/
  images/
  masks/
  labels/
NUDT-SIRST/
  images/
  masks/
  labels/
```

For NUAA-SIRST, mask files are named like `Misc_1_pixels0.png`, while image files are named like `Misc_1.png`.
The script removes the `_pixels0` suffix so YOLO can match `images/Misc_1.png` with `labels/Misc_1.txt`.

Run:

```powershell
.venv\Scripts\python.exe mask2yolo.py
```

Expected output from the current dataset:

```text
NUAA-SIRST: converted=427, empty=0, skipped=0
NUDT-SIRST: converted=1327, empty=0, skipped=0
done: converted=1754, empty=0, skipped=0
```

## `train_infrared.py`

Trains a YOLO detection baseline on NUAA-SIRST and NUDT-SIRST.

The script:

- Builds a deterministic train/val split.
- Generates `infrared_dataset/train.txt`, `val.txt`, and `infrared.yaml`.
- Trains with Ultralytics YOLO.
- Logs per-epoch `Ra`, `Fa`, `mAP50`, `mAP50-95`, and loss values.
- Saves the extra metric file `epoch_metrics_ra_fa.csv` inside the Ultralytics run directory.

Quick CPU baseline:

```powershell
.venv\Scripts\python.exe train_infrared.py --epochs 1 --imgsz 320 --batch 4 --ra-fa-conf 0.25
```

Suggested GPU run:

```powershell
.venv\Scripts\python.exe train_infrared.py --epochs 100 --imgsz 640 --batch 16 --device 0 --ra-fa-conf 0.25
```

## Metrics

The script defines:

```text
Ra = TP / GT
Fa = FP / validation image count
```

Ra/Fa are computed at:

- confidence threshold: `--ra-fa-conf`
- IoU threshold: `0.50 + 0.05 * --ra-fa-iou-index`

The default `--ra-fa-iou-index 0` means IoU `0.50`.

## Parameter Comparison

Small CPU comparison runs:

| Experiment | imgsz | Ra/Fa conf | Ra | Fa | TP/GT | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_e1_img320_conf0001 | 320 | 0.001 | 0.644397 | 280.908571 | 299/464 | 0.61307 | 0.51078 | 0.46183 | 0.14656 |
| cmp_e1_img320_conf025 | 320 | 0.25 | 0.517241 | 0.454286 | 240/464 | 0.61307 | 0.51078 | 0.46183 | 0.14656 |
| cmp_e1_img256_conf025 | 256 | 0.25 | 0.323276 | 0.300000 | 150/464 | 0.49742 | 0.40948 | 0.34571 | 0.12226 |

Interpretation:

- Lowering `ra-fa-conf` from `0.25` to `0.001` increases Ra, but it also creates many false alarms.
- Reducing `imgsz` from `320` to `256` hurts small-target detection. mAP50 drops from `0.46183` to `0.34571`.
- For reporting practical infrared detection results, keep the Ra/Fa confidence threshold fixed.

## Code Reading Notes

See `YOLOv8_CODE_READING_NOTES.md` for a guided reading path through the YOLOv8 training loop, model YAML, detection head, loss function, validation logic, and mAP calculation.
