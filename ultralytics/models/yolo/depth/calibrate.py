# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Scale-only calibration for monocular depth models.

The log-depth head predicts relative scene structure (shape); absolute scale is a separate
two-parameter log-affine ``d' = exp(a·log d + b)`` stored in the head's ``cal_a``/``cal_b``
buffers. This module fits ``(a, b)`` by closed-form least squares against ground-truth depth
over a small set of images — no gradient training, decoder weights untouched. It powers both
the trainer's automatic post-training calibration and the ``Model.calibrate()`` API.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ultralytics.utils import LOGGER


def _depth_head(model):
    """Return the Depth head module that carries ``cal_a``/``cal_b`` buffers, or None."""
    m = model.module if hasattr(model, "module") else model  # unwrap DDP
    seq = getattr(m, "model", None)
    head = seq[-1] if seq is not None else m
    return head if hasattr(head, "cal_a") else None


def _extract(preds):
    """Pull the (B,1,H,W) depth tensor out of any forward-output container."""
    if isinstance(preds, dict):
        return preds.get("depth")
    if isinstance(preds, (tuple, list)):
        return preds[0]
    return preds


def lstsq_affine(log_pred, log_gt, dist_power: float = 0.0):
    """Closed-form least-squares fit of (a, b) minimizing ||a·log_pred + b − log_gt||.

    With ``dist_power > 0`` the fit is weighted by ``gt**dist_power`` per pixel (gt = exp(log_gt)),
    so far pixels count more. This counters the near-pixel domination of an unweighted fit, which
    on near-heavy data over-compresses the scale and harms the far range. 0.0 = unweighted (default).
    """
    log_pred = np.asarray(log_pred, dtype=np.float64)
    log_gt = np.asarray(log_gt, dtype=np.float64)
    A = np.stack([log_pred, np.ones_like(log_pred)], axis=1)
    if dist_power > 0:
        sw = np.exp(0.5 * dist_power * log_gt)  # sqrt of gt**dist_power; row-scaling = weighted LSQ
        (a, b), *_ = np.linalg.lstsq(A * sw[:, None], log_gt * sw, rcond=None)
    else:
        (a, b), *_ = np.linalg.lstsq(A, log_gt, rcond=None)
    return float(a), float(b)


def fit_calibration(model, dataloader, device, max_images: int = 200, set_buffers: bool = True, dist_power: float = 0.0):
    """Fit the global log-affine ``(a, b)`` so ``exp(a·log(pred) + b) ≈ gt`` over valid pixels.

    Args:
        model: A ``DepthModel`` (or DDP-wrapped) with a log/sigmoid Depth head.
        dataloader: Yields batches with ``img`` (uint8, B×3×H×W) and ``depth`` (B×H×W meters).
        device: Torch device to run inference on.
        max_images: Stop after roughly this many images (calibration needs only a small set).
        set_buffers: If True, write the fitted values into the head's ``cal_a``/``cal_b``.

    Returns:
        (a, b) as Python floats, or None if no Depth head / no valid pixels were found.
    """
    head = _depth_head(model)
    if head is None:
        LOGGER.warning("calibrate: no Depth head with cal buffers found; skipping.")
        return None

    # Fit on the raw (un-calibrated) output: temporarily reset to identity.
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)

    model = model.to(device).eval()
    logp_all, logg_all = [], []
    seen = 0
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for batch in dataloader:
            img = batch["img"].to(device).float() / 255
            gt = batch["depth"].to(device).float()
            if gt.ndim == 3:
                gt = gt.unsqueeze(1)
            pred = _extract(model(img)).float()
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred, size=gt.shape[-2:], mode="bilinear", align_corners=True)
            valid = (gt > 1e-3) & (pred > 1e-3) & torch.isfinite(pred)
            if valid.any():
                lp = torch.log(pred[valid]).cpu().numpy()
                lg = torch.log(gt[valid]).cpu().numpy()
                # subsample to keep memory bounded (calibration is a 2-param fit)
                if lp.size > 50_000:
                    idx = rng.choice(lp.size, 50_000, replace=False)
                    lp, lg = lp[idx], lg[idx]
                logp_all.append(lp)
                logg_all.append(lg)
            seen += img.shape[0]
            if seen >= max_images:
                break

    if not logp_all:
        head.cal_a.fill_(a0)
        head.cal_b.fill_(b0)  # restore
        LOGGER.warning("calibrate: no valid depth pixels found; calibration skipped.")
        return None

    a, b = lstsq_affine(np.concatenate(logp_all), np.concatenate(logg_all), dist_power=dist_power)

    if set_buffers:
        head.cal_a.fill_(a)
        head.cal_b.fill_(b)
    else:
        head.cal_a.fill_(a0)
        head.cal_b.fill_(b0)
    LOGGER.info(f"Depth calibration fit on {seen} images: a={a:.4f} b={b:.4f} (scale ×{np.exp(b):.3f})")
    return a, b


def calibrate_checkpoint(ckpt_path, dataloader, device, dist_power: float = 0.0) -> None:
    """Fit calibration for a saved checkpoint in place (used by automatic post-training calibration).

    Loads the checkpoint, fits ``(a, b)`` on ``dataloader`` using a float copy on ``device``, writes
    the buffers into the stored model, and re-saves — preserving the rest of the checkpoint.
    """
    from copy import deepcopy

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("ema") or ckpt.get("model")
    if saved is None or _depth_head(saved) is None:
        return
    work = deepcopy(saved).float()
    res = fit_calibration(work, dataloader, device, set_buffers=True, dist_power=dist_power)
    if res is None:
        return
    a, b = res
    for key in ("ema", "model"):
        m = ckpt.get(key)
        if m is not None and _depth_head(m) is not None:
            _depth_head(m).cal_a.fill_(a)
            _depth_head(m).cal_b.fill_(b)
    torch.save(ckpt, ckpt_path)
    LOGGER.info(f"Auto-calibration written to {getattr(ckpt_path, 'name', ckpt_path)}: a={a:.4f} b={b:.4f}")
