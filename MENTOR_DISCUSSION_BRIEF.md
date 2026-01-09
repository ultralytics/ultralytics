# Collision Detection Pipeline - Status & Next Steps Brief

## 📌 Executive Summary

Implemented a YOLO-first collision detection pipeline with multi-layer false-positive filtering. Current implementation successfully identifies vehicle interactions while filtering out YOLO misclassifications.

---

## ✅ Current Implementation Status

### Working Features (Method A - Stable)

**Pipeline Stages**:
1. YOLO11 Detection (pixel coordinates)
2. Same-frame object deduplication (100px threshold)
3. Trajectory building (pixel space)
4. Filtering (short tracks, discontinuous tracks)
5. Keyframe detection with Homography transformation
6. Class-aware false positive filtering
7. Collision risk analysis

**Test Results**:
- **Valid events detected**: 5 near-miss events
- **False positives filtered**: 5 motorcycle misdetections (correctly identified as single object split into two Track IDs)
- **Test video**: 154 frames with vehicle interactions

**Key Achievement**:
Correctly filtered out YOLO misclassifications where a motorcycle was detected as both "person" AND "motorcycle" across multiple frames, while preserving legitimate car-car and person-person proximity events.

### Implementation Details

| Component | Current Approach |
|-----------|-----------------|
| Object Deduplication | Distance-based (100px in-frame threshold) |
| False Positive Filtering | Class-logic-based (removes illogical pairs like person+motorcycle) |
| Distance Calculation | World coordinates (via Homography transform) |
| Detection Threshold | 10 meters |

---

## ⚠️ Known Limitations

### Missing Detections (Not Captured)
1. **Frame 67**: Vehicle pair at 16.39m distance
2. **Frame 94**: Vehicle pair at 13.40m distance

**Root Cause**: Current distance threshold (10m) misses some "avoidance" level events

**Options**:
- **Quick fix**: Increase threshold to 20m (captures more events but may be less precise)
- **Better fix**: See Option B below

---

## 🎯 Two Proposed Paths Forward

### Option A: Quick Adjustment
**Change**: Increase distance threshold from 10m → 20m
**Pros**: 
- ✓ Captures more legitimate events
- ✓ Minimal code change
**Cons**: 
- ✗ May reduce precision
- ✗ No architectural improvement

### Option B: Architectural Improvement (Recommended)
**Change**: Apply Homography transformation EARLIER (after YOLO detection)

**Current Pipeline**:
```
YOLO (pixel) → Trajectories (pixel) → Filtering (pixel) 
  → Keyframe detection (transform to world here)
```

**Proposed Pipeline**:
```
YOLO (pixel) → Homography transform (WORLD)
  → Trajectories (world) → Filtering (world)
  → Keyframe detection (already in world)
```

**Benefits**:
- ✓ Unified coordinate system (cleaner architecture)
- ✓ More accurate velocity/trajectory calculations in real-world units (m/s)
- ✓ Avoid repeated transformations
- ✓ Better foundation for future features

**Considerations**:
- ⚠️ Requires refactoring Step 2-3
- ⚠️ Distance thresholds may need adjustment
- ⚠️ More testing needed

---

## 💬 Questions for Mentor

1. **Event Capture Priority**: Should we prioritize capturing more events (higher threshold) or keeping current precision level?

2. **Architecture Choice**: Which approach aligns better with project goals?
   - Option A: Pragmatic quick fix
   - Option B: Cleaner long-term architecture

3. **Collision Level Definitions**: What distance ranges should define:
   - Level 1 (Collision): < 0.5m?
   - Level 2 (Near Miss): 0.5-1.5m?
   - Level 3 (Avoidance): > 1.5m?
   
   Current threshold (10m for detection) seems to miss Level 3 events. Should Level 3 go up to 20m or higher?

4. **False Positive Strategy**: Current approach filters out "illogical" class combinations. Are there other filtering rules we should consider?

---

## 📊 Code Quality Metrics

- **Lines of Code**: ~1250 (main pipeline)
- **Test Coverage**: 1 video (154 frames, 30fps)
- **Error Handling**: ✓ Implemented
- **Logging**: ✓ Verbose debug output
- **Documentation**: ✓ Inline comments and docstrings

---

## 🔧 Running Current Version

```bash
python examples/trajectory_demo/collision_detection_pipeline_yolo_first_method_a.py \
  --video videos/Homograph_Teset_FullScreen.mp4 \
  --homography calibration/Homograph_Teset_FullScreen_homography.json \
  --skip-frames 3 \
  --model yolo11n \
  --min-track-length 3
```

**Output**: Structured results with:
- Detection frames and statistics
- Trajectory data
- Valid keyframes with images
- Collision analysis report

---

## 🚀 Next Steps (Pending Mentor Feedback)

1. **If Option A chosen**:
   - Change `world_distance_threshold` from 10.0 → 20.0
   - Re-test both missing detection cases
   - Verify no new false positives introduced

2. **If Option B chosen**:
   - Create experimental branch
   - Refactor trajectory building to use world coordinates
   - Adjust all distance thresholds accordingly
   - Comprehensive testing and comparison

3. **Either way**:
   - Test with additional video samples
   - Validate collision level definitions
   - Optimize YOLO model selection (yolo11n vs yolo11m vs yolo11l)

