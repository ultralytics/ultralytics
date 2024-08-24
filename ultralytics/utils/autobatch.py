# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Functions for estimating the best YOLO batch size to use a fraction of the available CUDA memory in PyTorch."""

from copy import deepcopy

import numpy as np
import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, profile


def check_train_batch_size(model, imgsz=640, amp=True, batch=-1):
    """
    Compute optimal YOLO training batch size using the autobatch() function.

    Args:
        model (torch.nn.Module): YOLO model to check batch size for.
        imgsz (int, optional): Image size used for training.
        amp (bool, optional): Use automatic mixed precision if True.
        batch (float, optional): Fraction of GPU memory to use. If -1, use default.

    Returns:
        (int): Optimal batch size computed using the autobatch() function.

    Note:
        If 0.0 < batch < 1.0, it's used as the fraction of GPU memory to use.
        Otherwise, a default fraction of 0.6 is used.
    """
    with autocast(enabled=amp):
        return autobatch(deepcopy(model).train(), imgsz, fraction=batch if 0.0 < batch < 1.0 else 0.6)


def autobatch(model, imgsz=640, fraction=0.60, batch_size=DEFAULT_CFG.batch):
    """
    Automatically estimate the best YOLO batch size to use a fraction of the available CUDA memory.

    Args:
        model (torch.nn.module): YOLO model to compute batch size for.
        imgsz (int, optional): The image size used as input for the YOLO model. Defaults to 640.
        fraction (float, optional): The fraction of available CUDA memory to use. Defaults to 0.60.
        batch_size (int, optional): The default batch size to use if an error is detected. Defaults to 16.

    Returns:
        (int): The optimal batch size.
    """
    # Check device
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% CUDA memory utilization.")
    device = next(model.parameters()).device  # get model device
    if device.type in {"cpu", "mps"}:
        LOGGER.info(f"{prefix} ⚠️ intended for CUDA devices, using default batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    # Inspect CUDA memory
    gb = 1 << 30  # bytes to GiB (1024 ** 3)
    d = str(device).upper()  # 'CUDA:0'
    properties = torch.cuda.get_device_properties(device)  # device properties
    t = properties.total_memory / gb  # GiB total
    r = torch.cuda.memory_reserved(device) / gb  # GiB reserved
    a = torch.cuda.memory_allocated(device) / gb  # GiB allocated
    f = t - (r + a)  # GiB free
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    # Profile batch sizes
    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)

        # Fit a solution
        y = [x[2] for x in results if x]  # memory [2]
        p = np.polyfit(batch_sizes[: len(y)], y, deg=1)  # first degree polynomial fit
        b = int((f * fraction - p[1]) / p[0])  # y intercept (optimal batch size)
        if None in results:  # some sizes failed
            i = results.index(None)  # first fail index
            if b >= batch_sizes[i]:  # y intercept above failure point
                b = batch_sizes[max(i - 1, 0)]  # select prior safe point
        if b < 1 or b > 1024:  # b outside of safe range
            b = batch_size
            LOGGER.info(f"{prefix}WARNING ⚠️ CUDA anomaly detected, using default batch-size {batch_size}.")

        fraction = (np.polyval(p, b) + r + a) / t  # actual fraction predicted
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.warning(f"{prefix}WARNING ⚠️ error detected: {e},  using default batch-size {batch_size}.")
        return batch_size
