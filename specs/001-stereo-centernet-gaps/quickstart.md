# Quickstart: Refactored stereo3ddet_full.yaml

**Date**: 2025-01-27  
**Purpose**: Guide for using the refactored `stereo3ddet_full.yaml` configuration

## Overview

The `stereo3ddet_full.yaml` config has been refactored to:

- Use YOLO11-style backbone (exactly like `yolo11-obb.yaml`) with `StereoConv` for 6-channel input
- Include PAN-style neck (Path Aggregation Network, like `yolo11-obb.yaml:34-48`) for feature refinement
- Use correct `StereoCenterNetHead` instead of placeholder `Detect` head
- **Minimal code changes**: Follow `yolo11-obb.yaml` structure closely, only modify input (StereoConv) and head (StereoCenterNetHead)

## Key Changes

### Before (Placeholder)

```yaml
backbone:
  - [-1, 1, TorchVision, [512, resnet18, DEFAULT, True, 2]]
head:
  - [-1, 1, Conv, [256, 3, 1]]
  - [-1, 1, Conv, [256, 3, 1]]
  - [[-1], 1, Detect, [nc]] # ❌ Wrong head
```

### After (Correct - follows `yolo11-obb.yaml` structure)

```yaml
backbone:
  # Exactly like yolo11-obb.yaml:18-30, but first layer uses StereoConv
  - [-1, 1, StereoConv, [64, 3, 2]] # ✅ 6-channel input (was Conv in yolo11-obb.yaml)
  # ... rest same as yolo11-obb.yaml:21-30 ...
head:
  # PAN neck - top-down then bottom-up (like yolo11-obb.yaml:34-48)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # Top-down path
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # P3 output
  - [-1, 1, Conv, [256, 3, 2]] # Bottom-up path
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]]
  # Detection head (replaces OBB in yolo11-obb.yaml)
  - [[16], 1, StereoCenterNetHead, [nc, 256]] # ✅ Correct head (single-scale P3)
```

## Usage

### Training

```bash
# The trainer currently overrides the YAML, but config now matches actual architecture
yolo task=stereo3ddet mode=train model=stereo3ddet_full.yaml data=dataset.yaml epochs=100
```

### Configuration Structure

The refactored config maintains all existing sections:

1. **Parameters**: `nc`, `input_channels`, `stereo` flag
2. **Backbone**: YOLO11-style (exactly like `yolo11-obb.yaml:18-30`) with `StereoConv` first layer
3. **Head**: PAN neck (like `yolo11-obb.yaml:34-48`) + `StereoCenterNetHead`
4. **Inference Settings**: Unchanged
5. **Geometric Construction**: Unchanged
6. **Dense Alignment**: Unchanged
7. **Occlusion**: Unchanged
8. **Training Settings**: Unchanged (includes `optimizer` and `val`)

## Architecture Flow

```
Input (6 channels: left+right RGB)
  ↓
Backbone (StereoConv → YOLO11 layers, like yolo11-obb.yaml:18-30)
  ↓ P3, P4, P5 features (layers 4, 6, 10)
PAN Neck - Top-down path (like yolo11-obb.yaml:34-40)
  ↓ P5 → P4 → P3 (upsampling + concatenation)
PAN Neck - Bottom-up path (like yolo11-obb.yaml:42-48)
  ↓ P3 → P4 → P5 (downsampling + concatenation)
  ↓ P3 output (256 channels, layer 16) OR multi-scale [P3, P4, P5]
StereoCenterNetHead (10 branches)
  ↓
Outputs: heatmap, offset, bbox_size, lr_distance, right_width,
         dimensions, orientation, vertices, vertex_offset, vertex_dist
```

## Important Notes

1. **Trainer Modification**: `Stereo3DDetTrainer.get_model()` will be modified to build `Stereo3DDetModel` from YAML config (production-ready approach, following PoseTrainer/OBBTrainer pattern)

2. **Channel Flow** (follows `yolo11-obb.yaml` exactly):
   - Input: 6 channels
   - After StereoConv: 64 channels
   - Backbone: 64 → 128 → 256 → 512 → 1024 (P1 → P2 → P3 → P4 → P5)
   - PAN top-down: 1024 → 512 → 256 (P5 → P4 → P3)
   - PAN bottom-up: 256 → 512 → 1024 (P3 → P4 → P5)
   - Head input: 256 channels (P3, single-scale) or [256, 512, 1024] (multi-scale)

3. **Detection Scale**: Configurable - can use single-scale (P3 only, layer 16) or multi-scale (P3, P4, P5, layers 16, 19, 22). Default: single-scale P3 (matches stereo paper).

## Verification

To verify the config structure:

```python
from ultralytics.nn.tasks import yaml_model_load

config = yaml_model_load("ultralytics/cfg/models/stereo3ddet_full.yaml")
assert config["stereo"] is True
assert config["input_channels"] == 6
assert config["backbone"][0][2] == "StereoConv"
assert config["head"][-1][2] == "StereoCenterNetHead"
```

## Next Steps

1. ✅ Config refactored to match actual architecture
2. 🔄 Modify `Stereo3DDetTrainer.get_model()` to build from YAML config
3. 🔄 Add `StereoCenterNetHead` and `StereoConv` imports to `tasks.py`
4. 🔄 Add tests to verify config can be parsed correctly and model builds successfully
