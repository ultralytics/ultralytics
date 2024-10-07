# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Functions for estimating the best YOLO batch size to use a fraction of the available CUDA memory in PyTorch."""

import multiprocessing
from copy import deepcopy

import numpy as np
import torch

from ultralytics.utils import DEFAULT_CFG, LOGGER, colorstr
from ultralytics.utils.torch_utils import autocast, profile


def run_autobatch(model, imgsz, fraction, batch_size, return_dict):
    """Run autobatch to find optimal batch size for training."""
    try:
        result = autobatch(model, imgsz, fraction, batch_size)
        return_dict["result"] = result
    except Exception as e:
        return_dict["error"] = str(e)
    finally:
        # Explicitly clear CUDA cache
        torch.cuda.empty_cache()


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
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        # Create a separate CUDA context for the new process
        torch.cuda.empty_cache()
        torch.cuda.set_device(model.device)

        p = multiprocessing.Process(
            target=run_autobatch,
            args=(deepcopy(model).train(), imgsz, batch if 0.0 < batch < 1.0 else 0.6, DEFAULT_CFG.batch, return_dict),
        )
        p.start()
        p.join()

        # Ensure the process has terminated
        p.terminate()
        p.join()

        if "error" in return_dict:
            LOGGER.warning(f"Error in autobatch: {return_dict['error']}")
            return DEFAULT_CFG.batch

        # Clear CUDA cache in the main process
        torch.cuda.empty_cache()

        return return_dict["result"]


def autobatch(model, imgsz=640, fraction=0.60, batch_size=DEFAULT_CFG.batch):
    """Autobatch inner function."""
    prefix = colorstr("AutoBatch: ")
    LOGGER.info(f"{prefix}Computing optimal batch size for imgsz={imgsz} at {fraction * 100}% CUDA memory utilization.")
    device = next(model.parameters()).device

    if device.type in {"cpu", "mps"}:
        LOGGER.info(f"{prefix} ⚠️ intended for CUDA devices, using default batch-size {batch_size}")
        return batch_size
    if torch.backends.cudnn.benchmark:
        LOGGER.info(f"{prefix} ⚠️ Requires torch.backends.cudnn.benchmark=False, using default batch-size {batch_size}")
        return batch_size

    gb = 1 << 30
    d = str(device).upper()
    properties = torch.cuda.get_device_properties(device)
    t = properties.total_memory / gb
    r = torch.cuda.memory_reserved(device) / gb
    a = torch.cuda.memory_allocated(device) / gb
    f = t - (r + a)
    LOGGER.info(f"{prefix}{d} ({properties.name}) {t:.2f}G total, {r:.2f}G reserved, {a:.2f}G allocated, {f:.2f}G free")

    batch_sizes = [1, 2, 4, 8, 16]
    try:
        img = [torch.empty(b, 3, imgsz, imgsz) for b in batch_sizes]
        results = profile(img, model, n=3, device=device)

        y = [x[2] for x in results if x]
        p = np.polyfit(batch_sizes[: len(y)], y, deg=1)
        b = int((f * fraction - p[1]) / p[0])

        if None in results:
            i = results.index(None)
            if b >= batch_sizes[i]:
                b = batch_sizes[max(i - 1, 0)]
        if b < 1 or b > 1024:
            b = batch_size
            LOGGER.info(f"{prefix}WARNING ⚠️ CUDA anomaly detected, using default batch-size {batch_size}.")

        fraction = (np.polyval(p, b) + r + a) / t
        LOGGER.info(f"{prefix}Using batch-size {b} for {d} {t * fraction:.2f}G/{t:.2f}G ({fraction * 100:.0f}%) ✅")
        return b
    except Exception as e:
        LOGGER.warning(f"{prefix}WARNING ⚠️ error detected: {e},  using default batch-size {batch_size}.")
        return batch_size
    finally:
        # Ensure CUDA cache is cleared
        torch.cuda.empty_cache()
