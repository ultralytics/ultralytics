# Data Model: stereo3ddet_full.yaml Refactoring

**Date**: 2025-01-27  
**Phase**: Phase 1 - Design

## Overview

This refactoring involves updating a single YAML configuration file to align the model architecture specification with the actual implementation. No new data entities are created; we're modifying the structure of an existing configuration file.

## Configuration Structure

### Current Structure (Before)

```yaml
stereo3ddet_full.yaml:
  - Parameters (nc, input_channels, stereo flag)
  - Backbone: TorchVision ResNet18 (single layer)
  - Head: Simple Conv layers + Detect (incorrect)
  - Inference settings
  - Geometric construction config
  - Dense alignment config
  - Occlusion config
  - Training settings
```

### Target Structure (After)

```yaml
stereo3ddet_full.yaml:
  - Parameters (nc, input_channels, stereo flag)
  - Backbone: YOLO11-style (exactly like yolo11-obb.yaml:18-30) with StereoConv first layer
  - Head: PAN neck (merged, like yolo11-obb.yaml:34-48) + StereoCenterNetHead (correct)
  - Inference settings (unchanged)
  - Geometric construction config (unchanged)
  - Dense alignment config (unchanged)
  - Occlusion config (unchanged)
  - Training settings (unchanged)
```

## Key Changes

### 1. Backbone Section

**Before**:

```yaml
backbone:
  - [-1, 1, TorchVision, [512, resnet18, DEFAULT, True, 2]]
```

**After** (exactly like `yolo11-obb.yaml:18-30`, only first layer changed):

```yaml
backbone:
  # Exactly like yolo11-obb.yaml:18-30, but first layer uses StereoConv
  - [-1, 1, StereoConv, [64, 3, 2]] # 0-P1/2: 6→64 channels (was Conv in yolo11-obb.yaml)
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4 (same as yolo11-obb.yaml:21)
  - [-1, 2, C3k2, [256, False, 0.25]] # 2 (same as yolo11-obb.yaml:22)
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8 (same as yolo11-obb.yaml:23)
  - [-1, 2, C3k2, [512, False, 0.25]] # 4 (same as yolo11-obb.yaml:24)
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16 (same as yolo11-obb.yaml:25)
  - [-1, 2, C3k2, [512, True]] # 6 (same as yolo11-obb.yaml:26)
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32 (same as yolo11-obb.yaml:27)
  - [-1, 2, C3k2, [1024, True]] # 8 (same as yolo11-obb.yaml:28)
  - [-1, 1, SPPF, [1024, 5]] # 9 (same as yolo11-obb.yaml:29)
  - [-1, 2, C2PSA, [1024]] # 10 (same as yolo11-obb.yaml:30)
```

### 2. Head Section (with merged neck)

**Before**:

```yaml
head:
  - [-1, 1, Conv, [256, 3, 1]]
  - [-1, 1, Conv, [256, 3, 1]]
  - [[-1], 1, Detect, [nc]]
```

**After** (PAN neck like `yolo11-obb.yaml:34-51`, replace OBB with StereoCenterNetHead):

```yaml
head:
  # PAN neck - Top-down path (like yolo11-obb.yaml:34-40)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11 (same as yolo11-obb.yaml:34)
  - [[-1, 6], 1, Concat, [1]] # 12: cat backbone P4 (same as yolo11-obb.yaml:35)
  - [-1, 2, C3k2, [512, False]] # 13 (same as yolo11-obb.yaml:36)

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14 (same as yolo11-obb.yaml:38)
  - [[-1, 4], 1, Concat, [1]] # 15: cat backbone P3 (same as yolo11-obb.yaml:39)
  - [-1, 2, C3k2, [256, False]] # 16: P3/8-small (same as yolo11-obb.yaml:40)

  # PAN neck - Bottom-up path (like yolo11-obb.yaml:42-48)
  - [-1, 1, Conv, [256, 3, 2]] # 17 (same as yolo11-obb.yaml:42)
  - [[-1, 13], 1, Concat, [1]] # 18: cat head P4 (same as yolo11-obb.yaml:43)
  - [-1, 2, C3k2, [512, False]] # 19: P4/16-medium (same as yolo11-obb.yaml:44)

  - [-1, 1, Conv, [512, 3, 2]] # 20 (same as yolo11-obb.yaml:46)
  - [[-1, 10], 1, Concat, [1]] # 21: cat head P5 (same as yolo11-obb.yaml:47)
  - [-1, 2, C3k2, [1024, True]] # 22: P5/32-large (same as yolo11-obb.yaml:48)

  # Detection head (replaces OBB in yolo11-obb.yaml:50)
  - [[16], 1, StereoCenterNetHead, [nc, 256]] # Single-scale: P3 only
  # OR for multi-scale: - [[16, 19, 22], 1, StereoCenterNetHead, [nc, 256]]  # P3, P4, P5
```

## Validation Rules

1. **Input Channels**: Must be 6 (left RGB + right RGB)
2. **First Layer**: Must use `StereoConv` to handle 6-channel input
3. **Head Input**: `StereoCenterNetHead` expects 256 input channels
4. **Layer Connections**: All `from` indices must reference valid previous layers
5. **Channel Consistency**: Output channels of one layer must match input of next

## State Transitions

N/A - Configuration file has no runtime state.

## Relationships

- **Backbone → PAN Neck → Head**: Sequential feature processing pipeline
- **Backbone layers 4, 6, 10**: Referenced by PAN neck for skip connections (P3, P4, P5)
- **PAN top-down path**: P5 (layer 10) → P4 (layer 6) → P3 (layer 4)
- **PAN bottom-up path**: P3 (layer 16) → P4 (layer 19) → P5 (layer 22)
- **Head layer 16, 19, or 22**: Output feeds into `StereoCenterNetHead` (single-scale uses P3/layer 16, multi-scale uses all three)

## Notes

- All other sections (inference, geometric_construction, dense_alignment, occlusion, training) remain unchanged
- **Trainer Modification**: `Stereo3DDetTrainer.get_model()` will be modified to build from YAML config (following PoseTrainer/OBBTrainer pattern)
- **Model Builder**: `Stereo3DDetModel` will use `parse_model()` to build architecture from YAML
- **Module Imports**: Need to add `StereoCenterNetHead` and `StereoConv` imports to `tasks.py` for `parse_model()` to resolve them
