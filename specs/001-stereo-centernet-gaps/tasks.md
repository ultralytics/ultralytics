# Tasks: Refactor stereo3ddet_full.yaml Architecture

**Input**: Design documents from `/specs/001-stereo-centernet-gaps/`  
**Prerequisites**: plan.md (required), research.md, data-model.md, quickstart.md

**Organization**: Tasks are organized by implementation phases to enable proper YAML-based model building for stereo 3D detection.

## Format: `[ID] [P?] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

- **Config path**: `ultralytics/cfg/models/`
- **Model path**: `ultralytics/models/yolo/stereo3ddet/`
- **NN path**: `ultralytics/nn/`
- **Test path**: `tests/`

---

## Phase 1: Setup (Module Imports)

**Purpose**: Add required module imports so `parse_model()` can resolve `StereoCenterNetHead` and `StereoConv`

- [X] T001 Add `StereoCenterNetHead` import to `ultralytics/nn/tasks.py` from `ultralytics.nn.modules`
- [X] T002 Add `StereoConv` import to `ultralytics/nn/tasks.py` from `ultralytics.nn.modules`
- [X] T003 Verify imports are available in `globals()` for `parse_model()` resolution in `ultralytics/nn/tasks.py`

**Checkpoint**: Module imports ready - `parse_model()` can now resolve stereo modules

---

## Phase 2: Config Refactoring (Backbone & Neck)

**Purpose**: Refactor `stereo3ddet_full.yaml` to follow `yolo11-obb.yaml` structure with PAN neck

**Dependencies**: Requires Phase 1 (module imports)

### Backbone Section

- [X] T004 [P] Replace TorchVision ResNet18 backbone with YOLO11-style backbone in `ultralytics/cfg/models/stereo3ddet_full.yaml` (copy from `yolo11-obb.yaml:18-30`)
- [X] T005 Replace first `Conv` layer with `StereoConv` for 6-channel input in `ultralytics/cfg/models/stereo3ddet_full.yaml` backbone section
- [X] T006 Verify backbone layer indices match `yolo11-obb.yaml` structure (layers 0-10) in `ultralytics/cfg/models/stereo3ddet_full.yaml`

### Head Section (PAN Neck)

- [X] T007 [P] Add PAN neck top-down path (like `yolo11-obb.yaml:34-40`) to head section in `ultralytics/cfg/models/stereo3ddet_full.yaml`
- [X] T008 Add PAN neck bottom-up path (like `yolo11-obb.yaml:42-48`) to head section in `ultralytics/cfg/models/stereo3ddet_full.yaml`
- [X] T009 Replace `Detect` head with `StereoCenterNetHead` in `ultralytics/cfg/models/stereo3ddet_full.yaml` (single-scale P3: `[[16], 1, StereoCenterNetHead, [nc, 256]]`)
- [X] T010 Verify layer indices and skip connections match `yolo11-obb.yaml` structure in `ultralytics/cfg/models/stereo3ddet_full.yaml`
- [X] T011 Preserve all existing config sections (inference, geometric_construction, dense_alignment, occlusion, training) in `ultralytics/cfg/models/stereo3ddet_full.yaml`

**Checkpoint**: Config structure matches `yolo11-obb.yaml` with stereo-specific modifications

---

## Phase 3: Trainer Modification

**Purpose**: Modify `Stereo3DDetTrainer.get_model()` to build from YAML config instead of override

**Dependencies**: Requires Phase 2 (config refactoring)

- [X] T012 Remove prototype override behavior in `Stereo3DDetTrainer.get_model()` in `ultralytics/models/yolo/stereo3ddet/train.py`
- [X] T013 Implement YAML-based model building following `PoseTrainer.get_model()` pattern in `ultralytics/models/yolo/stereo3ddet/train.py`
- [X] T014 Call `Stereo3DDetModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose)` in `ultralytics/models/yolo/stereo3ddet/train.py`
- [X] T015 Handle weights loading if provided in `ultralytics/models/yolo/stereo3ddet/train.py`
- [X] T016 Add verbose logging for model initialization in `ultralytics/models/yolo/stereo3ddet/train.py`

**Checkpoint**: Trainer builds model from YAML config (production-ready approach)

---

## Phase 4: Model Compatibility

**Purpose**: Ensure `Stereo3DDetModel` can properly handle `StereoCenterNetHead` from YAML

**Dependencies**: Requires Phase 1 (imports) and Phase 2 (config)

- [X] T017 [P] Verify `Stereo3DDetModel.__init__()` properly handles YAML config with `StereoCenterNetHead` in `ultralytics/models/yolo/stereo3ddet/model.py`
- [X] T018 Verify `parse_model()` correctly instantiates `StereoCenterNetHead` from YAML in `ultralytics/nn/tasks.py`
- [X] T019 Test model instantiation from refactored config in `tests/test_stereo3ddet_config.py`
- [X] T020 Verify model forward pass works with `StereoCenterNetHead` from YAML in `tests/test_stereo3ddet_config.py`

**Checkpoint**: Model can be built and run from YAML config

---

## Phase 5: Validation & Testing

**Purpose**: Validate config structure and ensure backward compatibility

**Dependencies**: Requires all previous phases

- [X] T021 [P] Create test to verify config can be loaded via `yaml_model_load()` in `tests/test_stereo3ddet_config.py`
- [X] T022 [P] Create test to verify backbone structure matches `yolo11-obb.yaml` (with `StereoConv` first layer) in `tests/test_stereo3ddet_config.py`
- [X] T023 [P] Create test to verify PAN neck structure matches `yolo11-obb.yaml:34-48` in `tests/test_stereo3ddet_config.py`
- [X] T024 [P] Create test to verify head uses `StereoCenterNetHead` instead of `Detect` in `tests/test_stereo3ddet_config.py`
- [X] T025 [P] Create test to verify model can be instantiated from config in `tests/test_stereo3ddet_config.py`
- [X] T026 [P] Create test to verify model forward pass produces 10-branch output in `tests/test_stereo3ddet_config.py`
- [X] T027 Verify all existing config sections are preserved in `ultralytics/cfg/models/stereo3ddet_full.yaml`
- [X] T028 Test training can start with refactored config in `tests/test_stereo3ddet_training.py`

**Checkpoint**: Config validated and model works end-to-end

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Config Refactoring (Phase 2)**: Depends on Phase 1 (module imports)
- **Trainer Modification (Phase 3)**: Depends on Phase 2 (config structure)
- **Model Compatibility (Phase 4)**: Depends on Phase 1 and Phase 2
- **Validation (Phase 5)**: Depends on all previous phases

### Execution Flow

```
Phase 1: Setup (Module Imports)
  ↓
Phase 2: Config Refactoring
  ├─→ Backbone (T004-T006)
  └─→ Head/PAN Neck (T007-T011)
  ↓
Phase 3: Trainer Modification (T012-T016)
  ↓
Phase 4: Model Compatibility (T017-T020)
  ↓
Phase 5: Validation & Testing (T021-T028)
```

### Parallel Opportunities

- **Phase 1**: T001 and T002 can run in parallel (different imports)
- **Phase 2**: T004 (backbone) and T007 (neck top-down) can start in parallel
- **Phase 4**: T017 (model verification) and T018 (parse_model verification) can run in parallel
- **Phase 5**: T021-T026 (all test tasks) can run in parallel

---

## Implementation Strategy

### MVP First

1. Complete Phase 1: Setup (module imports)
2. Complete Phase 2: Config Refactoring (backbone + PAN neck + head)
3. Complete Phase 3: Trainer Modification
4. **STOP and VALIDATE**: Test model can be instantiated and forward pass works
5. Complete Phase 4: Model Compatibility
6. Complete Phase 5: Validation & Testing

### Incremental Delivery

1. **Step 1**: Add imports → Config can reference stereo modules
2. **Step 2**: Refactor backbone → YOLO11 structure with StereoConv
3. **Step 3**: Refactor head → PAN neck + StereoCenterNetHead
4. **Step 4**: Modify trainer → Build from YAML
5. **Step 5**: Test & validate → End-to-end verification

### Recommended Order (Single Developer)

1. **Day 1**: Phase 1 (Setup) + Phase 2 Backbone section
2. **Day 2**: Phase 2 Head section + Phase 3 (Trainer)
3. **Day 3**: Phase 4 (Compatibility) + Phase 5 (Testing)

---

## Notes

- All tasks follow the checklist format: `- [ ] T### Description with file path`
- [P] tasks = different files, no dependencies
- Each phase should be independently testable
- Commit after each phase completion
- Config structure should match `yolo11-obb.yaml` exactly (only input and head differ)
- Preserve all existing config sections (inference, geometric_construction, etc.)

---

## Summary

| Metric | Value |
|--------|-------|
| **Total Tasks** | 28 |
| **Setup Phase** | 3 tasks |
| **Config Refactoring** | 8 tasks |
| **Trainer Modification** | 5 tasks |
| **Model Compatibility** | 4 tasks |
| **Validation & Testing** | 8 tasks |
| **Parallel Opportunities** | 10 tasks marked [P] |

**MVP Scope**: Phase 1 + Phase 2 + Phase 3 = 16 tasks for core refactoring
