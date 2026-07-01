# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Shared helpers for the anomaly_v2 task package."""

from __future__ import annotations

from ultralytics.utils.torch_utils import unwrap_model


def resolve_v2_model(m):  # TODO
    """Peel off all wrappers to reach the underlying ``YOLOAnomalyV2Model``.

    ``unwrap_model`` from ``ultralytics.utils.torch_utils`` only handles the
    ``torch.compile`` (``_orig_mod``) and ``DataParallel`` / ``DistributedDataParallel``
    (``.module``) wrappers. At predict/val time the v2 model is additionally
    wrapped by :class:`~ultralytics.nn.autobackend.AutoBackend`, whose
    real ``nn.Module`` lives at ``backend.model``. Without this extra hop our
    v2-specific methods (``set_mask_input`` / ``set_external_mask_once`` /
    ``disable_mask_once``) are silently never called and prompts have no
    effect on inference.

    Returns ``None`` if ``m`` is ``None``.
    """
    if m is None:
        return None
    m = unwrap_model(m)
    if hasattr(m, "backend") and hasattr(m.backend, "model"):
        m = m.backend.model
    return unwrap_model(m)
