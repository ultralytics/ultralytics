# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

# Lazy import to avoid circular dependency (depth.train imports yolo.detect)
import importlib as _importlib

from .predict import DepthPredictor
from .val import DepthValidator


def __getattr__(name: str):
    """Lazy-load DepthTrainer on first access to avoid a circular import with yolo.detect."""
    if name == "DepthTrainer":
        return _importlib.import_module(".train", __name__).DepthTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = "DepthPredictor", "DepthTrainer", "DepthValidator"
