# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Functions for estimating the best YOLO batch size to use a fraction of the available GPU memory in PyTorch."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, get_torch_device_backend, profile_ops


def check_train_batch_size(
    model: torch.nn.Module,
    imgsz: int = 640,
    amp: bool = True,
    batch: float = -1,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    """Compute optimal YOLO training batch size using the autobatch() function.

    Args:
        model (torch.nn.Module): YOLO model to check batch size for.
        imgsz (int, optional): Image size used for training.
        amp (bool, optional): Use automatic mixed precision if True.
        batch (int | float, optional): Fraction of GPU memory to use. If -1, use default.
        max_num_obj (int, optional): The maximum number of objects from dataset.
        dataset_size (int, optional): Total number of training images. If > 0, batch size will not exceed this value.

    Returns:
        (int): Optimal batch size computed using the autobatch() function.

    Raises:
        RuntimeError: If no candidate batch size produces a usable profile.

    Notes:
        If 0.0 < batch < 1.0, it's used as the fraction of GPU memory to use.
        Otherwise, a default fraction of 0.6 is used.
    """
    with autocast(enabled=amp, device=next(model.parameters()).device.type):
        return autobatch(
            deepcopy(model).train(),
            imgsz,
            fraction=batch if 0.0 < batch < 1.0 else 0.6,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )


def autobatch(
    model: torch.nn.Module,
    imgsz: int = 640,
    fraction: float = 0.60,
    batch_size: int = DEFAULT_CFG.batch,
    max_num_obj: int = 1,
    dataset_size: int = 0,
) -> int:
    """Automatically estimate the best YOLO batch size to use a fraction of the available GPU memory.

    Args:
        model (torch.nn.Module): YOLO model to compute batch size for.
        imgsz (int, optional): The image size used as input for the YOLO model.
        fraction (float, optional): The fraction of available CUDA memory to use.
        batch_size (int, optional): The default batch size to use if an error is detected.
        max_num_obj (int, optional): The maximum number of objects from dataset.
        dataset_size (int, optional): Total number of training images. If > 0, batch size will not exceed this value.

    Returns:
        (int): The optimal batch size.

    Raises:
        RuntimeError: If no candidate batch size produces a usable profile.
    """
    # Check device
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% GPU memory utilization.")
    device = next(model.parameters()).device  # get model device
    if device.type in {"cpu", "mps"}:
        LOGGER.warning(f"{prefix}intended for GPU devices, using default batch-size {batch_size}")
        return batch_size
    if device.type == "cuda" and torch.backends.cudnn.benchmark:
        LOGGER.warning(f"{prefix}Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # Inspect GPU memory
    accelerator = get_torch_device_backend(device)
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = f"{device.type.upper()}:{device.index}"
    properties = accelerator.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = accelerator.memory_reserved(device) / gb  # GiB reserved
    a = accelerator.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16] if t < 16 else [1, 2, 4, 8, 16, 32, 64]
    if dataset_size > 0:
        batch_sizes = [b for b in batch_sizes if b <= dataset_size]
    ch = model.yaml.get("channels", 3)
    try:
        img = [torch.empty(b, ch, imgsz, imgsz) for b in batch_sizes]
        results = profile_ops(img, model, n=1, device=device, max_num_obj=max_num_obj)

        # Fit a solution
        xy = [
            [x, y[2]]
            for i, (x, y) in enumerate(zip(batch_sizes, results))
            if y  # valid result
            and isinstance(y[2], (int, float))  # is numeric
            and 0 < y[2] < t  # between 0 and GPU limit
            and (i == 0 or not results[i - 1] or y[2] > results[i - 1][2])  # first item or increasing memory
        ]
        if xy:
            fit_x, fit_y = zip(*xy)
            p = np.polyfit(fit_x, fit_y, deg=1)  # first-degree (linear) polynomial fit
            b = int((round(f * fraction) - p[1]) / p[0])  # y intercept (optimal batch size)
            if None in results:  # some sizes failed
                i = results.index(None)  # first fail index
                if b >= batch_sizes[i]:  # y intercept above failure point
                    b = batch_sizes[max(i - 1, 0)]  # select prior safe point
            if b < 1 or b > 1024:  # b outside of safe range
                LOGGER.warning(f"{prefix}batch={b} outside safe range, using default batch-size {batch_size}.")
                b = batch_size
            if dataset_size > 0:
                b = min(b, dataset_size)

            fraction = (np.polyval(p, b) + r + a) / t  # predicted fraction
            LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
            return b
    except Exception as e:
        LOGGER.warning(f"{prefix}error detected: {e},  using default batch-size {batch_size}.")
        return batch_size
    finally:
        accelerator.empty_cache()

    raise RuntimeError(
        f"{prefix}no usable batch size found while profiling batch={batch_sizes}. "
        f"See the errors above, free GPU memory, reduce imgsz, or set batch explicitly."
    )
