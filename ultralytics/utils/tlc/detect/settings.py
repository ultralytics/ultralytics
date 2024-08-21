# Ultralytics YOLO 🚀 3LC Integration, AGPL-3.0 license

import warnings

from ultralytics.utils.tlc.settings import Settings

warnings.warn(
    "Importing `Settings` from `ultralytics.utils.tlc.detect.settings` is deprecated. "
    "Please import `Settings` from `ultralytics.utils.tlc` instead, as the former will be removed in a future commit.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["Settings"]