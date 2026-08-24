# Official Ultralytics KD for structured-pruned YOLOv8

This branch is based on the official Ultralytics `v8.4.104` release. It keeps
the official `ultralytics.nn.distill_model.DistillationModel` implementation
unchanged and adds only the compatibility required to load and rebuild
JasonSloan-style structured-pruned YOLOv8 checkpoints.

## What was added

- `C2fPruned`, `SPPFPruned`, and `DetectPruned` checkpoint classes.
- `DetectionModelPruned` graph reconstruction from the checkpoint's
  `maskbndict`.
- Dataset `nc` override when a one-class raw-pruned checkpoint is fine-tuned
  on a multi-class dataset.
- Detection trainer selection of `DetectionModelPruned`.

`DetectPruned` subclasses the official `Detect`, so the official
`DistillationModel` discovers the YOLOv8 neck layers `[15, 18, 21, 22]`.
Student/teacher channel differences are handled by the official projector.

## Training

```python
from ultralytics import YOLO

student = YOLO("/content/yolov8n_p50_raw.pt")
student.train(
    data="/content/drive/MyDrive/FPGA-YOLOv8n/KITSAT3_multi/kitsat3.yaml",
    epochs=80,
    imgsz=1024,
    batch=4,
    device=0,
    distill_model="/content/original_multi.pt",
    dis=6.0,
)
```

Do not pass the old fork's `finetune=True` option. The model type and pruning
masks are detected directly from the checkpoint.

Expected startup evidence:

```text
Overriding pruned model nc=1 with nc=2
Transferred 349/355 items from pretrained weights
... box_loss cls_loss dfl_loss dis_loss ...
```

The complete Colab workflow is maintained in the companion FPGA-YOLOv8n
project at `notebooks/experiments/04_train_kd_multi_1024.ipynb`.
