# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

"""Stereo 3D Object Detection module for YOLO.

This module implements stereo-based 3D object detection with CenterNet-style outputs,
including geometric construction, dense alignment, and occlusion handling based on
the Stereo CenterNet paper.

Key Components:
    - Stereo3DDetModel: Main model class for stereo 3D detection
    - Stereo3DDetTrainer: Training logic with stereo-specific augmentation
    - Stereo3DDetValidator: Validation with KITTI AP3D metrics
    - Stereo3DDetPredictor: Inference pipeline for stereo images

Implementation Modules:
    - nms: Heatmap-based NMS for CenterNet detection
    - keypoints: Perspective-aware keypoint selection
    - geometric: Geometric construction for 3D box estimation
    - dense_align: Sub-pixel stereo refinement via dense alignment
    - occlusion: Occlusion classification and handling

See Also:
    - scripts/benchmark_stereo3ddet.py for benchmarking utilities
"""

# =============================================================================
# Core Module Exports
# =============================================================================
from .model import Stereo3DDetModel
from .train import Stereo3DDetTrainer
from .val import Stereo3DDetValidator
from .predict import Stereo3DDetPredictor
from .visualize import plot_stereo_sample, plot_stereo_predictions
from .metrics import Stereo3DDetMetrics


from .keypoints import (
    select_perspective_keypoints,
    select_perspective_keypoints_batch,
    get_visible_face_indices,
    get_quadrant_name,
)

# =============================================================================
# GAP-001: Geometric Construction (User Story 2)
# =============================================================================
from .geometric import (
    GeometricObservations,
    CalibParams,
    GeometricConstruction,
    solve_geometric_batch,
    solve_geometric_single,
    fallback_simple_triangulation,
)

# =============================================================================
# GAP-002: Dense Alignment (User Story 5)
# =============================================================================
from .dense_align import (
    DenseAlignment,
    create_dense_alignment_from_config,
)

# =============================================================================
# GAP-006: Occlusion Classification (User Story 6)
# =============================================================================
from .occlusion import (
    classify_occlusion,
    should_skip_dense_alignment,
    get_occlusion_stats,
)

# =============================================================================
# Preprocessing and Postprocessing Utilities
# =============================================================================
from .preprocess import (
    preprocess_stereo_batch,
    preprocess_stereo_images,
    compute_letterbox_params,
    decode_and_refine_predictions,
    get_geometric_config,
    get_dense_alignment_config,
    clear_config_cache,
)

# =============================================================================
# Utility Functions
# =============================================================================
from .utils import (
    get_paper_class_mapping,
    filter_and_remap_class_id,
    is_paper_class,
    get_paper_class_names,
)

# =============================================================================
# Data Augmentation
# =============================================================================
from .augment import (
    StereoCalibration,
    PhotometricAugmentor,
    HorizontalFlipAugmentor,
    RandomScaleAugmentor,
    RandomCropAugmentor,
    StereoAugmentationPipeline,
)

__all__ = [
    # -------------------------------------------------------------------------
    # Core Classes
    # -------------------------------------------------------------------------
    "Stereo3DDetModel",
    "Stereo3DDetTrainer",
    "Stereo3DDetValidator",
    "Stereo3DDetPredictor",
    "Stereo3DDetMetrics",
    # -------------------------------------------------------------------------
    # Visualization
    # -------------------------------------------------------------------------
    "plot_stereo_sample",
    "plot_stereo_predictions",
    # -------------------------------------------------------------------------
    # GAP-003: Heatmap NMS
    # -------------------------------------------------------------------------
    "heatmap_nms",
    # -------------------------------------------------------------------------
    # GAP-004: Perspective Keypoint Selection
    # -------------------------------------------------------------------------
    "select_perspective_keypoints",
    "select_perspective_keypoints_batch",
    "get_visible_face_indices",
    "get_quadrant_name",
    # -------------------------------------------------------------------------
    # GAP-001: Geometric Construction
    # -------------------------------------------------------------------------
    "GeometricConstruction",
    "GeometricObservations",
    "CalibParams",
    "solve_geometric_batch",
    "solve_geometric_single",
    "fallback_simple_triangulation",
    # -------------------------------------------------------------------------
    # GAP-002: Dense Alignment
    # -------------------------------------------------------------------------
    "DenseAlignment",
    "create_dense_alignment_from_config",
    # -------------------------------------------------------------------------
    # GAP-006: Occlusion Classification
    # -------------------------------------------------------------------------
    "classify_occlusion",
    "should_skip_dense_alignment",
    "get_occlusion_stats",
    # -------------------------------------------------------------------------
    # Preprocessing and Postprocessing Utilities
    # -------------------------------------------------------------------------
    "preprocess_stereo_batch",
    "preprocess_stereo_images",
    "compute_letterbox_params",
    "decode_and_refine_predictions",
    "get_geometric_config",
    "get_dense_alignment_config",
    "clear_config_cache",
    # -------------------------------------------------------------------------
    # Utility Functions
    # -------------------------------------------------------------------------
    "get_paper_class_mapping",
    "filter_and_remap_class_id",
    "is_paper_class",
    "get_paper_class_names",
    # -------------------------------------------------------------------------
    # Data Augmentation
    # -------------------------------------------------------------------------
    "StereoCalibration",
    "PhotometricAugmentor",
    "HorizontalFlipAugmentor",
    "RandomScaleAugmentor",
    "RandomCropAugmentor",
    "StereoAugmentationPipeline",
]
