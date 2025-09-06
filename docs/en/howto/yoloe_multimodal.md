# YOLOE: Combine Text and Visual Prompts

YOLOE can accept both text prompt embeddings (TPE) and visual prompt embeddings (VPE) simultaneously, enabling more robust object detection through multimodal fusion.

## Overview

Single modality prompts sometimes fail to capture all necessary features for accurate detection:
- **Text prompts only**: May miss visual nuances and specific object variations
- **Visual prompts only**: May lack semantic understanding and generalization

By combining both modalities, YOLOE can leverage the strengths of each approach for improved detection performance.

## Fusion Modes

### 1. Concatenation (`concat`)
Treats text and visual prompts as separate class banks with zero overhead.

### 2. Weighted Sum (`sum`)
Combines embeddings per-class via weighted averaging. Requires a two-pass inference to extract VPE first.

## Quick Start

```python
import numpy as np
from ultralytics import YOLOE
from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

# 1) Load model and set text classes once
model = YOLOE("yoloe-11l-seg.pt")
names = ["person", "bus"]
model.set_classes(names, model.get_text_pe(names))

# 2) Build visual prompts (same as existing API)
visual_prompts = dict(
    bboxes=[np.array([[221.5, 405.8, 344.9, 857.5]])],
    cls=[np.array([0])],  # map prompt to class 0 ("person")
)

# 3) Run with both prompts - Concatenation Mode
#    Treats text and visual as separate class banks (zero overhead)
results = model.predict(
    "ultralytics/assets/zidane.jpg",
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    fuse_prompts="concat",
)

# 4) Run with fusion - Sum Mode 
#    Combines per-class via weighted sum; provide a fuse_map if counts differ
fuse_map = {0: [0]}  # text idx 0 ("person") fused with vpe indices [0]
results = model.predict(
    "ultralytics/assets/zidane.jpg",
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    fuse_prompts="sum",
    fuse_alpha=0.6,   # weight for text vs visual (0.6 text, 0.4 visual)
    fuse_map=fuse_map
)
```

## Parameters

### `fuse_prompts`
- **Type**: `str`
- **Options**: `"concat"`, `"sum"`
- **Default**: `None` (visual prompts only)
- **Description**: Fusion mode for combining text and visual prompts

### `fuse_alpha`
- **Type**: `float`
- **Range**: `0.0` to `1.0`
- **Default**: `0.5`
- **Description**: Weight for text embeddings in sum fusion (1-alpha for visual)

### `fuse_map`
- **Type**: `Dict[int, List[int]]`
- **Default**: `None`
- **Description**: Maps text class indices to visual prompt indices for aggregation

## Fusion Details

### Concatenation Mode
```python
# Zero overhead - simply enables both modalities concurrently
results = model.predict(
    image,
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    fuse_prompts="concat"
)
```

### Sum Mode
```python
# Two-pass inference with fusion
fuse_map = {0: [0, 1], 1: [2]}  # text class 0 maps to visual indices [0,1]
results = model.predict(
    image,
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    fuse_prompts="sum",
    fuse_alpha=0.7,  # 70% text, 30% visual
    fuse_map=fuse_map
)
```

## Class Name Handling

In concatenation mode, text and visual prompts are treated as separate class banks. To distinguish between them in results:

- **Text classes**: Use original names (e.g., "person", "bus") 
- **Visual classes**: Use auto-generated names (e.g., "object0", "object1")

For more control over class naming, you can modify the model's class names after prediction:

```python
# After concatenation prediction
results = model.predict(
    image,
    visual_prompts=visual_prompts,
    predictor=YOLOEVPSegPredictor,
    fuse_prompts="concat"
)

# Update class names to distinguish text vs visual
text_names = ["person", "bus"]  # your original text classes
visual_names = ["vis:person", "vis:glasses"]  # visual prompt classes
combined_names = text_names + visual_names
model.names = combined_names
```

## Implementation Notes

- **Concatenation Mode**: Zero overhead, simply enables both modalities concurrently
- **Sum Mode**: Performs two-pass inference (VPE extraction + fusion), adding small overhead
- **Fusion Maps**: If omitted, assumes equal text/visual class counts and fuses index-wise
- **Normalization**: Text and visual embeddings are normalized for stability in sum fusion
- **Backward Compatibility**: All existing YOLOE functionality remains unchanged

## Use Cases

### Concatenation Mode
Best for scenarios where you want both text and visual prompts to work independently:
- Detecting both known classes (text) and novel objects (visual examples)
- Combining general detection with specific visual examples
- Zero computational overhead

### Sum Mode  
Best for scenarios where you want to combine complementary information:
- Text prompt fails to capture visual nuances → visual prompt provides specificity
- Visual prompt lacks semantic understanding → text prompt provides context
- Small performance cost for improved robustness

## Error Handling

The system will raise appropriate errors for:
- Mismatched tensor dimensions
- Missing fusion maps when class counts differ
- Invalid fusion modes
- Both TPE and VPE being None
