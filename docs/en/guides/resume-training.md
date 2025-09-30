# Resume Training from Checkpoints

Training large models can take hours or even days, and interruptions (system restarts, crashes, or early stops) are common.
Ultralytics provides built-in support to **resume training** from a previously saved checkpoint so you don’t have to start over.
This guide explains how to continue training from the last saved checkpoint using both the **CLI** and **Python API**.

---

## 1. Using the CLI

To resume training from the **last checkpoint** in your `runs/` directory:

```bash
yolo detect train resume model=runs/detect/train/weights/last.pt
```

Or for segmentation/classification tasks:

```bash
yolo segment train resume model=runs/segment/train/weights/last.pt
yolo classify train resume model=runs/classify/train/weights/last.pt
```

> **Note:**
> `last.pt` is automatically saved during training in the `weights/` folder of your run.
> You can also resume from a specific checkpoint by providing its path:

```bash
yolo detect train resume model=path/to/custom_checkpoint.pt
```

---

## 2. Using the Python API

You can also resume training programmatically:

```python
from ultralytics import YOLO

# Load last checkpoint
model = YOLO("runs/detect/train/weights/last.pt")

# Resume training for 50 more epochs
model.train(resume=True, epochs=50)
```

### Additional Parameters

- `epochs`: Specify how many more epochs to run.
- `device`: Control GPU/CPU placement if needed.
- All other training arguments (`batch`, `imgsz`, `data`, etc.) can still be used.

---

## 3. Best Practices

- **Save regularly**: Ultralytics automatically saves `last.pt` and `best.pt`.
- **Use `resume=True`**: This restores optimizer state, learning rate, and scheduler so training continues smoothly.
- **Monitor logs**: Training will append to the same run folder (e.g., `runs/detect/train`).

---

## 4. Troubleshooting

| Problem                                            | Solution                                                                                      |
| -------------------------------------------------- | --------------------------------------------------------------------------------------------- |
| Training restarts from epoch 0 instead of resuming | Ensure you pass `resume=True` and point to `last.pt`, not `best.pt`.                          |
| Checkpoint not found                               | Verify the path: `runs/detect/train/weights/last.pt`.                                         |
| CUDA out of memory                                 | Resume with a smaller batch size: `model.train(resume=True, batch=8)`.                        |
| Wrong dataset used                                 | When resuming, the dataset is taken from the checkpoint. Use a new run if switching datasets. |

---

## 5. Related Documentation

- [Checkpoints and Weights](https://github.com/ultralytics/ultralytics/blob/main/docs/en/reference/engine/results.md)
- [Python API Reference](https://github.com/ultralytics/ultralytics/blob/main/docs/en/reference/engine/model.md)

    ***

## 6. Example Workflow

1. Start training:

```bash
yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100
```

2. Interrupt training after 20 epochs (Ctrl+C).
3. Resume from last checkpoint:

```bash
yolo detect train resume model=runs/detect/train/weights/last.pt
```

4. Training continues from epoch 21 → 100 instead of restarting. 🎉
