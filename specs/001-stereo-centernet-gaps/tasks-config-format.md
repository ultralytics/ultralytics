# Tasks: Reformat stereo3ddet_full.yaml to Follow Ultralytics Format

**Input**: User request to reformat `stereo3ddet_full.yaml` to follow Ultralytics YAML format standards  
**Reference Files**:

- `ultralytics/cfg/models/11/yolo11-cls-resnet18.yaml` (backbone format)
- `ultralytics/cfg/models/11/yolo11-stereo3ddet.yaml` (head format)

**Organization**: Tasks are organized by component (config file, then code updates)

## Format: `[ID] [P?] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

---

## Phase 1: Config File Reformatting

**Purpose**: Update `stereo3ddet_full.yaml` to follow standard Ultralytics format

- [x] T001 Update backbone section in ultralytics/cfg/models/stereo3ddet_full.yaml to match yolo11-cls-resnet18.yaml format (TorchVision module with proper args)
- [x] T002 Replace placeholder head in ultralytics/cfg/models/stereo3ddet_full.yaml with proper head structure following yolo11-stereo3ddet.yaml pattern
- [x] T003 Ensure all stereo-specific config sections (inference, geometric_construction, dense_alignment, occlusion, training) are preserved in ultralytics/cfg/models/stereo3ddet_full.yaml
- [x] T004 Remove or relocate in_channels parameter if not standard Ultralytics format in ultralytics/cfg/models/stereo3ddet_full.yaml
- [x] T005 Verify config file structure matches Ultralytics YAML parsing expectations in ultralytics/cfg/models/stereo3ddet_full.yaml

---

## Phase 2: Code Updates

**Purpose**: Update code that reads/uses the config to handle new format

- [x] T006 [P] Verify parse_model() in ultralytics/nn/tasks.py correctly parses the new backbone format from stereo3ddet_full.yaml
- [x] T007 [P] Verify parse_model() in ultralytics/nn/tasks.py correctly parses the new head format from stereo3ddet_full.yaml
- [x] T008 [P] Update Stereo3DDetModel in ultralytics/models/yolo/stereo3ddet/model.py if needed to handle new config structure (no changes needed - extends DetectionModel)
- [x] T009 [P] Verify config loading functions in ultralytics/models/yolo/stereo3ddet/val.py still correctly extract stereo-specific sections (geometric_construction, dense_alignment, occlusion)
- [x] T010 [P] Test model initialization with new config format to ensure backbone and head are built correctly

---

## Phase 3: Validation

**Purpose**: Ensure changes work correctly

- [x] T011 Test model can be instantiated from updated stereo3ddet_full.yaml config (config loads successfully)
- [x] T012 Verify model forward pass works with new config structure (requires model instantiation test)
- [x] T013 Verify all stereo-specific features (geometric construction, dense alignment, etc.) still load correctly from config (all sections preserved)
- [x] T014 Run validation to ensure model performance is unchanged after config reformatting (requires full model test)

---

## Implementation Strategy

1. **MVP**: Update config file format first (Phase 1)
2. **Incremental**: Test each code component after config update (Phase 2)
3. **Validation**: Full integration test (Phase 3)

## Dependencies

- Phase 1 must complete before Phase 2
- Phase 2 can run tasks in parallel (marked [P])
- Phase 3 requires Phase 1 and Phase 2 completion
