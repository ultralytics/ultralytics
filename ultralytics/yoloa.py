# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Backward-compatible import shim for YOLOA.

YOLOA is now defined in ``ultralytics.models.yolo.model`` alongside YOLOE and YOLO-World.
The ``ultralytics.yoloa`` submodule is kept so existing ``from ultralytics.yoloa import YOLOA``
imports continue to work.
"""

from ultralytics.models.yolo import YOLOA

__all__ = ("YOLOA",)
