# Implementation Plan: Refactor stereo3ddet_full.yaml Architecture

**Branch**: `001-stereo-centernet-gaps` | **Date**: 2025-01-27 | **Spec**: [spec.md](./spec.md)
**Input**: User request to refactor `stereo3ddet_full.yaml` to use correct head (StereoCenterNetHead) and neck (PAN-style, similar to yolo11-obb.yaml), using YOLO11's backbone and neck structure with minimal code changes.

## Summary

Refactor the `stereo3ddet_full.yaml` configuration and `Stereo3DDetTrainer.get_model()` to:
1. Replace the placeholder `Detect` head with the proper `StereoCenterNetHead` (10-branch output)
2. Add a PAN-style neck section (similar to `yolo11-obb.yaml:34-51`) - NOT exactly FPN, but Path Aggregation Network structure
3. Use YOLO11's backbone structure (similar to `yolo11-obb.yaml`) with `StereoConv` for 6-channel input
4. **Modify `Stereo3DDetTrainer.get_model()` to build model from YAML config** (following pattern from PoseTrainer, OBBTrainer) instead of using the prototype override behavior
5. **Minimal code changes**: Use YOLO11 config structure directly, only modify input (StereoConv) and head (StereoCenterNetHead)

This refactoring will align the YAML config with YOLO11 structure (like yolo11-obb.yaml) and enable proper YAML-based model building for stereo 3D detection, making it production-ready instead of prototype code. The PAN neck structure matches the stereo paper's approach but uses YOLO11 config format for consistency.

## Technical Context

**Language/Version**: Python 3.11  
**Primary Dependencies**: PyTorch, Ultralytics YOLO framework, torchvision (for backbones)  
**Storage**: N/A (model configuration only)  
**Testing**: pytest, integration tests with KITTI dataset  
**Target Platform**: Linux (CUDA-capable GPUs)  
**Project Type**: Model architecture configuration refactoring within existing codebase  
**Performance Goals**: Maintain ≥30 FPS inference with ResNet-18, ≥20 FPS with DLA-34  
**Constraints**: 
- Must maintain backward compatibility with existing trained models
- Must support 6-channel input (left + right RGB)
- Must output 10 branches: heatmap, offset, bbox_size, lr_distance, right_width, dimensions, orientation, vertices, vertex_offset, vertex_dist
**Scale/Scope**: 
- YAML config file refactoring (stereo3ddet_full.yaml)
- Trainer method modification (Stereo3DDetTrainer.get_model())
- Potential model class updates (Stereo3DDetModel) to ensure compatibility with StereoCenterNetHead

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Status**: ✅ PASS

- No new libraries required (using existing Ultralytics infrastructure)
- Configuration-only change (no new code modules)
- Maintains existing API contracts
- Backward compatible (old configs can still work via trainer override)

## Project Structure

### Documentation (this feature)

```text
specs/001-stereo-centernet-gaps/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (/speckit.plan command)
├── data-model.md        # Phase 1 output (/speckit.plan command)
├── quickstart.md        # Phase 1 output (/speckit.plan command)
├── contracts/           # Phase 1 output (/speckit.plan command)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
ultralytics/
├── cfg/models/
│   └── stereo3ddet_full.yaml  # Target file for refactoring
├── nn/
│   ├── modules/
│   │   └── __init__.py  # Already has StereoCenterNetHead registered
│   └── tasks.py         # parse_model() - may need neck support
└── models/yolo/stereo3ddet/
    ├── train.py         # Stereo3DDetTrainer.get_model() - MODIFY to build from YAML (like PoseTrainer/OBBTrainer)
    └── model.py         # Stereo3DDetModel - may need updates to handle StereoCenterNetHead properly
```

**Structure Decision**: Single-file refactoring with potential minor modifications to model builder if neck section support is needed. The existing codebase structure supports this change.

## Complexity Tracking

> **No violations identified** - This is a configuration refactoring that aligns existing code with proper architecture.

## Phase 0: Research Questions ✅ COMPLETE

**Status**: All research questions resolved. See `research.md` for details.

### Resolved Questions

1. ✅ **Neck Section Support**: Merge `neck` into `head` section (parse_model doesn't support separate neck)
2. ✅ **YOLO11 Backbone Adaptation**: Use `StereoConv` for first layer, then standard YOLO11 layers (like yolo11-obb.yaml)
3. ✅ **Trainer Override Behavior**: **CHANGED** - Remove override, modify `get_model()` to build from YAML config (production-ready approach)
4. ✅ **Neck Type**: **UPDATED** - Use PAN-style neck (Path Aggregation Network) similar to `yolo11-obb.yaml:34-51`, NOT exactly FPN. This matches stereo paper's PAN approach but uses YOLO11 config format.
5. ✅ **Multi-scale vs Single-scale**: Can use multi-scale (P3, P4, P5) like yolo11-obb.yaml, or single-scale (P3 only) - configurable
6. ✅ **Backbone Options**: Use YOLO11-style backbone (exactly like yolo11-obb.yaml backbone) with `StereoConv` first layer
7. ✅ **Module Imports**: Need to add `StereoCenterNetHead` and `StereoConv` imports to `tasks.py` for `parse_model()` to resolve them
8. ✅ **Config Structure**: Follow `yolo11-obb.yaml` structure closely - only change input (StereoConv) and head (StereoCenterNetHead)

## Phase 1: Design Decisions ✅ COMPLETE

**Status**: Design artifacts generated. See `data-model.md`, `contracts/`, and `quickstart.md`.

### Key Design Decisions

1. **Backbone Structure**: YOLO11-style (exactly like `yolo11-obb.yaml:18-30`) with `StereoConv` first layer (6→64 channels) instead of `Conv`
2. **Neck Integration**: PAN-style neck merged into `head` section (similar to `yolo11-obb.yaml:34-48`) - top-down then bottom-up path aggregation
3. **Head**: `StereoCenterNetHead` with 256 input channels, outputs 10 branches (replaces OBB head in yolo11-obb.yaml)
4. **Backward Compatibility**: All other config sections (inference, geometric_construction, etc.) unchanged
5. **Trainer Modification**: `Stereo3DDetTrainer.get_model()` will build `Stereo3DDetModel` from YAML config (following PoseTrainer/OBBTrainer pattern)
6. **Model Compatibility**: Ensure `Stereo3DDetModel` and `parse_model()` can properly handle `StereoCenterNetHead` in YAML configs
7. **Module Imports**: Add `StereoCenterNetHead` and `StereoConv` to imports in `tasks.py` so `parse_model()` can resolve them via `globals()[m]`

## Phase 2: Implementation Tasks

*To be generated by /speckit.tasks command*
