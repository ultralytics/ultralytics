"""
Analytics helpers for Ultralytics.

This module currently exposes the `batcheval` helper for batch evaluation of
multiple YOLO models on a single dataset split.
"""

from .batcheval import BatchEvalResult, ModelSpec, batcheval, evaluate_model, resolve_models, run_cli

__all__ = ["BatchEvalResult", "ModelSpec", "batcheval", "evaluate_model", "resolve_models", "run_cli"]
