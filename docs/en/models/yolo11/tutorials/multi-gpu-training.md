---
comments: true
description: Train Ultralytics YOLO11 across multiple GPUs using PyTorch DDP. Learn device arguments, batch scaling, and best practices for distributed training.
keywords: YOLO11, multi-GPU, DDP, distributed training, PyTorch, Ultralytics, DistributedDataParallel, NCCL
canonical: https://docs.ultralytics.com/models/yolo11/tutorials/multi-gpu-training/
---

<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "TechArticle",
  "headline": "Multi-GPU Training with YOLO11 Using PyTorch DDP",
  "description": "Train Ultralytics YOLO11 across multiple GPUs using PyTorch DDP. Learn device arguments, batch scaling, and best practices for distributed training.",
  "url": "https://docs.ultralytics.com/models/yolo11/tutorials/multi-gpu-training/",
  "image": "https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/ultralytics-active-learning-loop.avif",
  "datePublished": "2026-06-04",
  "dateModified": "2026-06-04",
  "author": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "publisher": {"@type": "Organization", "name": "Ultralytics", "url": "https://www.ultralytics.com"},
  "mainEntityOfPage": "https://docs.ultralytics.com/models/yolo11/tutorials/multi-gpu-training/"
}
</script>

# Multi-GPU Training with YOLO11

<!-- NOTE FOR MURAT: Please verify that DDP training works correctly with YOLO11 and confirm any known issues with specific GPU counts or configurations. Add measured scaling efficiency numbers (e.g. 1x, 1.85x, 3.6x throughput for 1/2/4 GPUs). Confirm `torchrun` and the `device` argument syntax work as documented below. -->

[Ultralytics YOLO11](../../../models/yolo11.md) supports multi-GPU training via PyTorch [DistributedDataParallel (DDP)](https://pytorch.org/docs/stable/notes/ddp.html), which distributes the workload evenly across all available GPUs on a single machine. This is the fastest way to cut wall-clock training time when you have access to multiple NVIDIA GPUs.

## Prerequisites

- Multiple NVIDIA GPUs on the same machine (NVLink or PCIe)
- CUDA and cuDNN installed
- PyTorch built with NCCL support (standard `pip install torch` includes this)
- Ultralytics package: `pip install ultralytics`

## Single-GPU vs Multi-GPU Command Comparison

| Setup | CLI command | Notes |
|---|---|---|
| Single GPU 0 | `yolo train model=yolo11n.pt device=0` | Default single-GPU |
| Single GPU 1 | `yolo train model=yolo11n.pt device=1` | Use second GPU |
| 2 GPUs (0 and 1) | `yolo train model=yolo11n.pt device=0,1` | DDP auto-launched |
| 4 GPUs (0–3) | `yolo train model=yolo11n.pt device=0,1,2,3` | 4-GPU DDP |
| All GPUs | `yolo train model=yolo11n.pt device=0,1,2,3` | List all GPU IDs |

## Training Commands

=== "CLI"

    ```bash
    # 2-GPU training — GPUs 0 and 1
    yolo train model=yolo11n.pt data=coco.yaml epochs=100 batch=32 device=0,1

    # 4-GPU training
    yolo train model=yolo11m.pt data=coco.yaml epochs=100 batch=64 device=0,1,2,3

    # Using torchrun directly (advanced — gives more DDP control)
    torchrun --nproc_per_node 4 -m ultralytics.utils.dist train \
        model=yolo11m.pt data=coco.yaml epochs=100 batch=64
    ```

=== "Python"

    ```python
    from ultralytics import YOLO

    model = YOLO("yolo11n.pt")

    # 2-GPU training
    results = model.train(
        data="coco.yaml",
        epochs=100,
        batch=32,
        device=[0, 1],
    )

    # 4-GPU training
    results = model.train(
        data="coco.yaml",
        epochs=100,
        batch=64,
        device=[0, 1, 2, 3],
    )
    ```

## Batch Size Scaling

In DDP mode, the `batch` argument you specify is the **total batch size across all GPUs**. Ultralytics automatically splits it equally between devices.

| GPUs | Total `batch` | Per-GPU batch |
|---|---|---|
| 1 | 16 | 16 |
| 2 | 32 | 16 |
| 4 | 64 | 16 |
| 8 | 128 | 16 |

!!! tip "Keep per-GPU batch size constant"

    Scale `batch` linearly with the number of GPUs to keep the per-GPU batch size the same. You may also need to scale the learning rate (linear scaling rule: `lr = base_lr × n_gpus`), though Ultralytics applies auto-scaling by default.

!!! warning "Memory constraints"

    If you hit CUDA out-of-memory errors, reduce the per-GPU batch size or use a smaller model variant (`yolo11n` vs `yolo11x`).

## Monitoring Training Progress

With DDP, only the **rank-0 process** writes logs and saves checkpoints. Use standard monitoring tools:

```bash
# TensorBoard
tensorboard --logdir runs/train

# Watch GPU utilisation across all GPUs
watch -n 1 nvidia-smi
```

## Tips for Multi-GPU Efficiency

!!! tip "Best practices"

    - Use **NVLink** interconnect between GPUs for best gradient synchronisation throughput when available.
    - Avoid mixing GPU models (e.g. V100 + A100) — slowest GPU sets the pace for the entire run.
    - Set `workers` per GPU proportionally (e.g. `workers=8` per GPU = `workers=8` total; Ultralytics scales internally).
    - Use `cache=True` to pre-cache the dataset in RAM, reducing data-loading bottlenecks when using fast GPUs.

```bash
# Recommended flags for large multi-GPU runs
yolo train model=yolo11l.pt data=coco.yaml \
    epochs=200 batch=128 device=0,1,2,3 \
    cache=True workers=8 amp=True
```

## See Also

- [YOLO11 Model Overview](../../../models/yolo11.md)
- [Train YOLO11 on a Custom Dataset](train-custom-dataset.md)
- [Hyperparameter Tuning for YOLO11](hyperparameter-tuning.md)
- [Ultralytics Train Mode Docs](../../../modes/train.md)
