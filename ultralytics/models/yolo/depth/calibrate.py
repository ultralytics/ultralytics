# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Scale-only calibration for monocular depth models.

The log-depth head predicts relative scene structure (shape); absolute scale is a separate
two-parameter log-affine ``d' = exp(a·log d + b)`` stored in the head's ``cal_a``/``cal_b``
buffers. This module fits ``(a, b)`` by closed-form least squares against ground-truth depth
over a small set of images — no gradient training, decoder weights untouched. It powers both
the trainer's automatic post-training calibration and the ``Model.calibrate()`` API.

Auto-calibration uses a "calibrate only if it helps" policy (:func:`select_calibration_cv`):
candidates are scored on held-out images and applied only when they beat the un-calibrated
output, so the absolute scale is fixed for cross-domain models without harming in-domain ones.
"""

from __future__ import annotations

from pathlib import Path

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


def _delta1_none(log_pred, log_gt, a: float, b: float) -> float:
    """δ1 under no per-image alignment after applying ``d' = exp(a·log_pred + b)``.

    δ1 is the fraction of pixels with ``max(d'/gt, gt/d') < 1.25``; in log space that is
    ``|a·log_pred + b − log_gt| < log(1.25)``. This is the deployment metric (raw absolute
    scale, the ``align="none"`` protocol) the policy optimizes — the val scoreboard's default
    ``align="median"`` is scale-invariant and cannot see calibration.
    """
    ld = (a * np.asarray(log_pred, dtype=np.float64) + b) - np.asarray(log_gt, dtype=np.float64)
    return float(np.mean(np.abs(ld) < np.log(1.25)))


def select_calibration(lp_fit, lg_fit, lp_score, lg_score, dist_power: float = 0.0, margin: float = 0.0):
    """Pick the calibration that best improves *held-out* raw-scale δ1 — "calibrate only if it helps".

    Three candidates are fit on the ``*_fit`` log-pixel arrays and scored on the independent
    ``*_score`` arrays under :func:`_delta1_none`:
      - ``identity`` (a=1, b=0): no calibration,
      - ``scale-only`` (a=1, b=mean(log_gt − log_pred)): a global scale,
      - ``affine`` (a, b from :func:`lstsq_affine`): scale + log-slope.
    The winner is the highest held-out δ1. A candidate must beat the current best by ``margin``
    to win, so ties favor the simpler model (identity > scale-only > affine). Identity is the
    baseline, so a calibration that does not generalize is rejected — auto-cal does no harm.

    Returns:
        dict with ``a``, ``b`` (floats of the winner), ``name``, and ``scores`` (per-candidate δ1).
    """
    lp_fit = np.asarray(lp_fit, dtype=np.float64)
    lg_fit = np.asarray(lg_fit, dtype=np.float64)
    candidates = [
        ("identity", 1.0, 0.0),
        ("scale-only", 1.0, float(np.mean(lg_fit - lp_fit))),
        ("affine", *lstsq_affine(lp_fit, lg_fit, dist_power=dist_power)),
    ]
    scored = [(name, a, b, _delta1_none(lp_score, lg_score, a, b)) for name, a, b in candidates]
    best = scored[0]  # identity is the baseline; simpler candidates come first so ties favor them
    for cand in scored[1:]:
        if cand[3] > best[3] + margin:
            best = cand
    return {"a": best[1], "b": best[2], "name": best[0], "scores": {s[0]: s[3] for s in scored}}


def select_calibration_cv(pairs, dist_power: float = 0.0, margin: float = 0.0, folds: int = 2, allow_affine: bool = False):
    """Cross-validated "calibrate only if it helps": choose a candidate by K-fold held-out δ1.

    ``pairs`` is a list of per-image ``(log_pred, log_gt)`` arrays. Each candidate type is scored
    on every fold while held out (fit on the rest) via :func:`select_calibration`, and the per-type
    held-out δ1 is averaged across folds — so every image contributes to scoring exactly once and a
    candidate that only wins on one noisy split cannot be selected. The winning *type* must beat
    identity's mean held-out δ1 by ``margin`` (ties favor the simpler type); the final ``(a, b)`` is
    then refit on all pairs.

    Only ``identity`` and ``scale-only`` are auto-selected by default. The affine *slope* (``a``)
    overfits within-dataset cross-validation — both folds share a dataset's idiosyncrasies, so the
    extra parameter looks free and gets picked even when it harms cross-distribution generalization
    (confirmed across NYU/SUNRGBD/KITTI). Set ``allow_affine=True`` to include it.

    Returns:
        dict with ``a``, ``b`` (floats), ``name``, and ``cv_scores`` (mean held-out δ1 per type).
    """
    names = ["identity", "scale-only"] + (["affine"] if allow_affine else [])
    k = max(2, min(folds, len(pairs)))
    per_fold = {n: [] for n in names}
    for f in range(k):
        fit = [pairs[i] for i in range(len(pairs)) if i % k != f]
        score = [pairs[i] for i in range(len(pairs)) if i % k == f]
        if not fit or not score:
            continue
        s = select_calibration(
            np.concatenate([p[0] for p in fit]), np.concatenate([p[1] for p in fit]),
            np.concatenate([p[0] for p in score]), np.concatenate([p[1] for p in score]),
            dist_power=dist_power,
        )["scores"]
        for n in names:
            per_fold[n].append(s[n])
    cv = {n: float(np.mean(per_fold[n])) for n in names}
    best = "identity"
    for n in names[1:]:
        if cv[n] > cv[best] + margin:
            best = n
    # refit the chosen type on all pairs (CV selects the type; all data sets the parameters)
    lp = np.concatenate([p[0] for p in pairs])
    lg = np.concatenate([p[1] for p in pairs])
    if best == "identity":
        a, b = 1.0, 0.0
    elif best == "scale-only":
        a, b = 1.0, float(np.mean(lg - lp))
    else:
        a, b = lstsq_affine(lp, lg, dist_power=dist_power)
    return {"a": a, "b": b, "name": best, "cv_scores": cv}


def fit_calibration(model, dataloader, device, max_images: int = 200, set_buffers: bool = True, dist_power: float = 0.0):
    """Fit the global log-affine ``(a, b)`` so ``exp(a·log(pred) + b) ≈ gt`` over valid pixels.

    Args:
        model: A ``DepthModel`` (or DDP-wrapped) with a log/sigmoid Depth head.
        dataloader: Yields batches with ``img`` (uint8, B×3×H×W) and ``depth`` (B×H×W meters).
        device: Torch device to run inference on.
        max_images: Stop after roughly this many images (calibration needs only a small set).
        set_buffers: If True, write the fitted values into the head's ``cal_a``/``cal_b``.
        dist_power (float): Weight each pixel by gt**dist_power in the calibration fit (0.0 = uniform).

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


def _collect_logpairs(model, dataloader, device, max_images: int):
    """Run the model over the loader and return a list of per-image ``(log_pred, log_gt)`` arrays.

    One entry per image (each subsampled to ≤20k valid pixels) so callers can split images into
    independent fit/score sets. Calibration buffers are reset to identity for the duration so the
    fit sees the raw output, then restored.
    """
    head = _depth_head(model)
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    model = model.to(device).eval()
    rng = np.random.default_rng(0)
    pairs = []
    seen = 0
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
            for pi, gi in zip(pred, gt):
                valid = (gi > 1e-3) & (pi > 1e-3) & torch.isfinite(pi)
                if not valid.any():
                    continue
                lp = torch.log(pi[valid]).cpu().numpy()
                lg = torch.log(gi[valid]).cpu().numpy()
                if lp.size > 20_000:
                    idx = rng.choice(lp.size, 20_000, replace=False)
                    lp, lg = lp[idx], lg[idx]
                pairs.append((lp, lg))
            seen += img.shape[0]
            if seen >= max_images:
                break
    head.cal_a.fill_(a0)
    head.cal_b.fill_(b0)  # restore; callers set the chosen value explicitly
    return pairs


def fit_calibration_selective(
    model, dataloader, device, max_images: int = 200, margin: float = 0.002, dist_power: float = 0.0
):
    """Select and apply calibration via "calibrate only if it helps" (see :func:`select_calibration_cv`).

    Collects per-image ``(log_pred, log_gt)`` over the loader, splits images into independent
    fit/score folds (no leakage), chooses identity / scale-only / affine by cross-validated raw-scale
    δ1, and writes the winner into the head's ``cal_a``/``cal_b``. ``dist_power > 0`` weights the
    affine fit toward far pixels (see :func:`lstsq_affine`).

    Returns:
        The :func:`select_calibration_cv` result dict, or None if no Depth head / too few images.
    """
    head = _depth_head(model)
    if head is None:
        LOGGER.warning("calibrate: no Depth head with cal buffers found; skipping.")
        return None
    pairs = _collect_logpairs(model, dataloader, device, max_images)
    if len(pairs) < 2:
        LOGGER.warning("calibrate: fewer than 2 valid images for fit/score split; calibration skipped.")
        return None
    res = select_calibration_cv(pairs, dist_power=dist_power, margin=margin)
    head.cal_a.fill_(res["a"])
    head.cal_b.fill_(res["b"])
    scores = " ".join(f"{n}={v:.4f}" for n, v in res["cv_scores"].items())
    LOGGER.info(
        f"Depth calibration selected '{res['name']}' (a={res['a']:.4f} b={res['b']:.4f}); CV held-out δ1 {scores}"
    )
    return res


def _plot_calibrated_batches(model, dataloader, device, a, b, name, plot_dir, max_batches: int = 3, max_images: int = 4):
    """Write ``val_batch{ni}_calibrated.jpg`` panels (RGB | GT | raw | calibrated) to ``plot_dir``.

    Runs the model with calibration buffers at identity to get the raw prediction; the calibrated
    column is its deterministic affine ``exp(a·log(raw) + b)`` — no second forward. The first
    ``max_batches`` batches are the same ones BaseValidator plots as ``val_batch{ni}.jpg`` (val
    loaders are not shuffled), so the files are directly comparable. With the "only if it helps"
    policy the selected ``name`` may be ``identity``; the panels are still written (raw ==
    calibrated), which documents that calibration was a no-op. Buffers are restored afterwards.
    """
    from .val import plot_depth_panels

    head = _depth_head(model)
    a0, b0 = float(head.cal_a), float(head.cal_b)
    head.cal_a.fill_(1.0)
    head.cal_b.fill_(0.0)
    model = model.to(device).eval()
    titles = ["RGB", "GT", "raw", f"calibrated ({name} x{np.exp(b):.2f})"]
    plot_dir = Path(plot_dir)
    with torch.no_grad():
        for ni, batch in enumerate(dataloader):
            if ni >= max_batches:
                break
            img = batch["img"].to(device).float() / 255
            gt = batch["depth"].to(device).float()
            raw = _extract(model(img)).float()
            if raw.ndim == 3:
                raw = raw.unsqueeze(1)
            cal = torch.exp(a * torch.log(raw.clamp(min=1e-3)) + b)
            plot_depth_panels(
                img, gt, [raw, cal], plot_dir / f"val_batch{ni}_calibrated.jpg",
                titles=titles, max_images=max_images,
            )
    head.cal_a.fill_(a0)
    head.cal_b.fill_(b0)


def calibrate_checkpoint(ckpt_path, dataloader, device, dist_power: float = 0.0, plot_dir=None) -> None:
    """Fit calibration for a saved checkpoint in place (used by automatic post-training calibration).

    Loads the checkpoint, selects calibration with the "calibrate only if it helps" policy
    (:func:`fit_calibration_selective`) on ``dataloader`` using a float copy on ``device``, writes
    the chosen buffers into the stored model, and re-saves — preserving the rest of the checkpoint.

    Args:
        ckpt_path: Path to the ``.pt`` checkpoint file to calibrate in place.
        dataloader: Yields batches with ``img`` (uint8, B×3×H×W) and ``depth`` (B×H×W meters).
        device: Torch device to run inference on.
        dist_power (float): Weight each pixel by gt**dist_power in the calibration fit (0.0 = uniform).
        plot_dir: If set, also write ``val_batch{ni}_calibrated.jpg`` comparison panels
            (RGB | GT | raw | calibrated) for the first val batches into this directory.
    """
    from copy import deepcopy

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    saved = ckpt.get("ema") or ckpt.get("model")
    if saved is None or _depth_head(saved) is None:
        return
    work = deepcopy(saved).float()
    res = fit_calibration_selective(work, dataloader, device, dist_power=dist_power)
    if res is None:
        return
    a, b = res["a"], res["b"]
    for key in ("ema", "model"):
        m = ckpt.get(key)
        if m is not None and _depth_head(m) is not None:
            _depth_head(m).cal_a.fill_(a)
            _depth_head(m).cal_b.fill_(b)
    torch.save(ckpt, ckpt_path)
    LOGGER.info(
        f"Auto-calibration written to {getattr(ckpt_path, 'name', ckpt_path)}: '{res['name']}' a={a:.4f} b={b:.4f}"
    )
    if plot_dir is not None:
        try:
            _plot_calibrated_batches(work, dataloader, device, a, b, res["name"], plot_dir)
            LOGGER.info(f"Calibrated val_batch plots written to {plot_dir}")
        except Exception as e:
            LOGGER.warning(f"Calibrated val plots skipped ({type(e).__name__}: {e})")
