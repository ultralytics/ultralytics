# Research: Refactor stereo3ddet_full.yaml Architecture

**Date**: 2025-01-27  
**Phase**: Phase 0 - Research & Clarification  
**Status**: ✅ All clarifications resolved

## Research Questions & Answers

### 1. Neck Section Support in parse_model()

**Question**: Does `parse_model()` in `tasks.py` support a separate `neck` section, or must it be merged into `head`?

**Finding**:

- Current `parse_model()` implementation (line 1577) only processes: `d["backbone"] + d["head"]`
- No native support for separate `neck` section
- The stereo-centernet configs (`stereo-centernet-n.yaml`, etc.) have `neck` sections, but these may not be actively used or may require code modification

**Decision**: Merge `neck` into `head` section for compatibility with existing parser

- **Rationale**: Avoids modifying core `parse_model()` function, maintains backward compatibility
- **Alternative Considered**: Modify `parse_model()` to support `d.get("neck", [])` - rejected due to complexity and risk of breaking existing configs

**Implementation**: Structure the YAML as (similar to `yolo11-obb.yaml:34-51`):

```yaml
backbone:
  # ... backbone layers (like yolo11-obb.yaml:18-30) ...
head:
  # PAN neck (top-down path) - similar to yolo11-obb.yaml:34-40
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 6], 1, Concat, [1]] # cat backbone P4
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 4], 1, Concat, [1]] # cat backbone P3
  - [-1, 2, C3k2, [256, False]] # P3 output
  # PAN neck (bottom-up path) - similar to yolo11-obb.yaml:42-48
  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 13], 1, Concat, [1]] # cat head P4
  - [-1, 2, C3k2, [512, False]]
  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 10], 1, Concat, [1]] # cat head P5
  - [-1, 2, C3k2, [1024, True]]
  # Detection head (single-scale or multi-scale)
  - [-1, 1, StereoCenterNetHead, [nc, 256]] # or use P3 only: [[16], 1, StereoCenterNetHead, [nc, 256]]
```

---

### 2. YOLO11 Backbone Adaptation for 6-Channel Input

**Question**: How should YOLO11's backbone be adapted for 6-channel input?

**Finding**:

- YOLO11 backbone starts with `Conv` layer expecting 3 channels
- Stereo models need 6 channels (left RGB + right RGB)
- `StereoConv` module exists and is registered in `nn/modules/__init__.py`
- Example usage in `stereo-centernet-n.yaml`: `- [-1, 1, StereoConv, [64, 3, 2]]`

**Decision**: Use `StereoConv` for the first layer, then standard YOLO11 backbone structure (exactly like `yolo11-obb.yaml:18-30`)

- **Rationale**: `StereoConv` handles 6→N channel conversion, subsequent layers work with standard channel counts
- **Pattern**: First layer uses `StereoConv` instead of `Conv`, all other layers match `yolo11-obb.yaml` exactly
- **Minimal Changes**: Only change line 20 of yolo11-obb.yaml from `Conv` to `StereoConv`

**Implementation**:

```yaml
backbone:
  # Exactly like yolo11-obb.yaml:18-30, but first layer uses StereoConv
  - [-1, 1, StereoConv, [64, 3, 2]] # 0-P1/2: 6→64 channels (was Conv in yolo11-obb.yaml)
  - [-1, 1, Conv, [128, 3, 2]] # 1-P2/4 (same as yolo11-obb.yaml)
  - [-1, 2, C3k2, [256, False, 0.25]] # 2 (same)
  - [-1, 1, Conv, [256, 3, 2]] # 3-P3/8 (same)
  - [-1, 2, C3k2, [512, False, 0.25]] # 4 (same)
  - [-1, 1, Conv, [512, 3, 2]] # 5-P4/16 (same)
  - [-1, 2, C3k2, [512, True]] # 6 (same)
  - [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32 (same)
  - [-1, 2, C3k2, [1024, True]] # 8 (same)
  - [-1, 1, SPPF, [1024, 5]] # 9 (same)
  - [-1, 2, C2PSA, [1024]] # 10 (same)
```

---

### 3. Trainer Override Behavior ⚠️ UPDATED

**Question**: Should we modify `Stereo3DDetTrainer.get_model()` to respect the YAML config, or keep the override but make the YAML match the actual architecture for documentation?

**Finding**:

- Current `Stereo3DDetTrainer.get_model()` (line 225-239) completely bypasses YAML and directly instantiates `StereoYOLOv11Wrapper`
- This was intentional for prototype, but user wants production-ready code
- Other trainers (PoseTrainer, OBBTrainer) build models from YAML config:
  - `PoseTrainer.get_model()`: `PoseModel(cfg, nc=..., ch=..., data_kpt_shape=..., verbose=...)`
  - `OBBTrainer.get_model()`: `OBBModel(cfg, nc=..., ch=..., verbose=...)`
- `Stereo3DDetModel` already extends `DetectionModel` and handles YAML config

**Decision**: **MODIFY `get_model()` to build from YAML config** (production-ready approach)

- **Pattern**: Follow `PoseTrainer`/`OBBTrainer` pattern
- **Implementation**: `Stereo3DDetModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)`
- **Rationale**:
  - Aligns with other task trainers (consistent API)
  - Production-ready instead of prototype code
  - YAML config becomes the source of truth
  - Enables proper model architecture customization via YAML

**Requirements**:

- Ensure `Stereo3DDetModel` can properly parse YAML with `StereoCenterNetHead`
- Verify `parse_model()` handles `StereoCenterNetHead` correctly
- **Import `StereoCenterNetHead` in `tasks.py`** - Currently exported in `nn/modules/__init__.py` but not imported in `tasks.py`, so `parse_model()` won't find it via `globals()[m]`
- Test that model built from YAML matches expected architecture

**Implementation Details**:

- `StereoCenterNetHead` is defined in `stereo_yolo_v11.py` and exported in `nn/modules/__init__.py`
- `parse_model()` uses `globals()[m]` to resolve module names (line 1583)
- Need to add `from ultralytics.nn.modules import StereoCenterNetHead, StereoConv` to `tasks.py`
- `StereoCenterNetHead` constructor: `__init__(self, in_channels: int = 256, num_classes: int = 3)`
- YAML usage: `- [-1, 1, StereoCenterNetHead, [nc, 256]]` where `nc` is number of classes

---

### 4. Neck Type: PAN vs FPN ⚠️ UPDATED

**Question**: Should we use FPN (Feature Pyramid Network) or PAN (Path Aggregation Network) for the neck?

**Finding**:

- Stereo paper uses PAN (Path Aggregation Network) - see `stereo_yolo_v11.py:78-128` (`StereoPAN` class)
- `yolo11-obb.yaml:34-51` shows PAN structure: top-down path (upsample) then bottom-up path (downsample)
- PAN provides better feature aggregation than FPN alone
- YOLO11 configs use PAN structure (top-down + bottom-up) for better multi-scale feature fusion

**Decision**: Use PAN-style neck (similar to `yolo11-obb.yaml:34-51`), NOT exactly FPN

- **Rationale**:
  - Matches stereo paper's PAN approach (`StereoPAN` in stereo_yolo_v11.py)
  - Uses YOLO11 config format directly (minimal code changes)
  - PAN provides better feature aggregation than FPN alone
  - Can use multi-scale (P3, P4, P5) or single-scale (P3 only) - configurable
- **Structure**: Top-down path (upsample P5→P4→P3) then bottom-up path (downsample P3→P4→P5)

**Implementation** (following `yolo11-obb.yaml:34-51`):

```yaml
head:
  # PAN neck - Top-down path (like yolo11-obb.yaml:34-40)
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11
  - [[-1, 6], 1, Concat, [1]] # 12: cat backbone P4
  - [-1, 2, C3k2, [512, False]] # 13

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14
  - [[-1, 4], 1, Concat, [1]] # 15: cat backbone P3
  - [-1, 2, C3k2, [256, False]] # 16 (P3/8-small)

  # PAN neck - Bottom-up path (like yolo11-obb.yaml:42-48)
  - [-1, 1, Conv, [256, 3, 2]] # 17
  - [[-1, 13], 1, Concat, [1]] # 18: cat head P4
  - [-1, 2, C3k2, [512, False]] # 19 (P4/16-medium)

  - [-1, 1, Conv, [512, 3, 2]] # 20
  - [[-1, 10], 1, Concat, [1]] # 21: cat head P5
  - [-1, 2, C3k2, [1024, True]] # 22 (P5/32-large)

  # Detection head - can use single-scale (P3) or multi-scale
  - [[16], 1, StereoCenterNetHead, [nc, 256]] # Single-scale: P3 only
  # OR: - [[16, 19, 22], 1, StereoCenterNetHead, [nc, 256]]  # Multi-scale: P3, P4, P5
```

---

### 5. Backbone Options ⚠️ UPDATED

**Question**: Should we support both TorchVision backbones (ResNet18/50) and YOLO11-style backbones, or standardize on one?

**Finding**:

- Current `stereo3ddet_full.yaml` uses TorchVision ResNet18
- `StereoYOLOv11` supports ResNet18/50/DLA-34 via programmatic construction
- `yolo11-obb.yaml` provides a clean YOLO11 backbone structure
- Using YOLO11 structure directly allows minimal code changes (only input and head)

**Decision**: Use YOLO11-style backbone exactly like `yolo11-obb.yaml:18-30` (only change first layer to `StereoConv`)

- **Rationale**:
  - Minimal code changes: copy yolo11-obb.yaml structure, only modify input (StereoConv) and head (StereoCenterNetHead)
  - Consistent with YOLO11 architecture patterns
  - Better integration with YOLO11 PAN neck structure
  - Production-ready approach using standard YOLO11 config format
- **Implementation**: Copy `yolo11-obb.yaml` backbone exactly, replace first `Conv` with `StereoConv`

**Config Structure**: Follow `yolo11-obb.yaml` closely:

- Backbone: Same as yolo11-obb.yaml:18-30 (first layer: StereoConv instead of Conv)
- Neck: Same as yolo11-obb.yaml:34-48 (PAN structure)
- Head: Replace OBB with StereoCenterNetHead

---

## Technical Decisions Summary

| Decision         | Choice                                      | Rationale                                                     |
| ---------------- | ------------------------------------------- | ------------------------------------------------------------- |
| Neck handling    | Merge into `head` section                   | Avoids parser modification, maintains compatibility           |
| First layer      | `StereoConv` for 6-channel input            | Required for stereo input, matches existing pattern           |
| Trainer behavior | Build from YAML config                      | Production-ready, consistent with other trainers              |
| Neck type        | PAN-style (like yolo11-obb.yaml)            | Matches stereo paper's PAN, uses YOLO11 config format         |
| Detection scale  | Configurable (P3 only or multi-scale)       | Can use single-scale (P3) or multi-scale (P3, P4, P5)         |
| Backbone type    | YOLO11-style (exactly like yolo11-obb.yaml) | Minimal code changes, use YOLO11 structure directly           |
| Config structure | Follow yolo11-obb.yaml closely              | Only modify input (StereoConv) and head (StereoCenterNetHead) |

## Dependencies & Prerequisites

- ✅ `StereoCenterNetHead` already registered in `nn/modules/__init__.py`
- ✅ `StereoConv` already registered in `nn/modules/__init__.py`
- ✅ YOLO11 backbone/neck structure available in `yolo11.yaml`
- ✅ Example stereo configs exist (`stereo-centernet-*.yaml`) showing pattern

## Implementation Notes

1. **Channel Flow**: Follow `yolo11-obb.yaml` channel flow exactly
   - Backbone: 64 → 128 → 256 → 512 → 1024 (same as yolo11-obb.yaml)
   - PAN neck top-down: 1024 → 512 → 256 (P5 → P4 → P3)
   - PAN neck bottom-up: 256 → 512 → 1024 (P3 → P4 → P5)
   - Head input: 256 channels (if single-scale P3) or [256, 512, 1024] (if multi-scale)

2. **Layer Indexing**: When merging neck into head, adjust layer indices in comments for clarity

3. **Backward Compatibility**: Keep all existing config sections (inference, geometric_construction, etc.) unchanged

4. **Testing**: Verify config can be loaded and model can be instantiated (even if trainer overrides it)
