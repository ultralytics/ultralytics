# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license

import warnings

from ultralytics.utils.tlc.engine.model import TLCYOLO

warnings.warn(
    "Importing `TLCYOLO` from `ultralytics.utils.tlc.detect.model` is deprecated. "
    "Please import `TLCYOLO` from `ultralytics.utils.tlc` instead, as the former will be removed in a future commit.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["TLCYOLO"]