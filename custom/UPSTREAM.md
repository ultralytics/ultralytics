# Upstream synchronization record

本项目基于官方 Ultralytics 仓库维护：

- Upstream: <https://github.com/ultralytics/ultralytics.git>
- Tracking branch: `upstream/main`
- Local base at scaffold creation: `b5f6c7024`
- Scaffold date: `2026-09-08`

## Synchronization checklist

每次同步完成后更新本文件：

```text
Upstream commit: <40-character SHA>
Sync date: YYYY-MM-DD
Validation: yolo checks; CPU COCO8 one-epoch smoke test; custom model load/export
Notes: <conflicts resolved or behavior changes>
```

## Local extension inventory

- `custom/`: project maintenance files and local configuration templates.
- `ultralytics/nn/modules/`: custom PyTorch modules, when added.
- `custom/configs/models/`: project model YAML files.
- `custom/configs/datasets/`: dataset configuration templates.
