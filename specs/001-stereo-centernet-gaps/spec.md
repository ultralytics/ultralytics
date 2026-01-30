# Feature Specification: Stereo CenterNet Implementation Gap Analysis

**Feature Branch**: `001-stereo-centernet-gaps`  
**Created**: December 18, 2024  
**Status**: Draft  
**Input**: User description: "Stereo CenterNet Implementation Gap Analysis"

**Reference Paper**: "Stereo CenterNet based 3D Object Detection for Autonomous Driving" (Shi et al., 2021)

## Clarifications

### Session 2024-12-18

- Q: Which photometric error method should be the primary implementation for dense alignment? → A: Support both NCC and SAD with zero-mean normalization, with NCC as the default. This provides flexibility for ablation studies.
- Q: At what confidence score threshold should detections proceed to 3D refinement? → A: Configurable threshold with 0.3 as default. This allows tuning for different use cases while matching standard KITTI evaluation practices.

## User Scenarios & Testing _(mandatory)_

### User Story 1 - Accurate 3D Box Estimation via Geometric Construction (Priority: P1)

As an ML researcher training stereo 3D object detection models, I need the system to solve for 3D bounding box parameters using the full set of geometric constraints from stereo imagery, so that my model can achieve the paper's reported accuracy on KITTI benchmark.

**Why this priority**: The geometric construction module is the core differentiator of Stereo CenterNet. Without it, the model uses simple stereo triangulation which is fundamentally less accurate. This is the highest-impact gap affecting detection quality.

**Independent Test**: Can be tested by running inference on KITTI validation set and comparing AP3D@0.7 metrics before and after enabling geometric construction. Expected improvement: 5-10% AP3D on moderate difficulty.

**Acceptance Scenarios**:

1. **Given** a trained stereo 3D detection model and stereo image pair, **When** the system decodes detections using geometric construction, **Then** the 3D box center (x, y, z) and orientation (θ) are solved using 7 geometric constraint equations via iterative optimization.

2. **Given** predicted 2D box corners, stereo disparity, 3D dimensions, and initial orientation estimate, **When** the geometric solver runs, **Then** it converges within 10 iterations or terminates with a valid fallback result.

3. **Given** geometric construction is enabled during validation, **When** evaluating on KITTI validation set with ResNet-18 backbone, **Then** AP3D@0.7 (Moderate) reaches at least 30% (baseline target before dense alignment).

---

### User Story 2 - Depth Refinement via Dense Photometric Alignment (Priority: P2)

As an ML researcher, I need the system to refine depth estimates using photometric matching between left and right images, so that depth accuracy improves for unoccluded objects.

**Why this priority**: Dense alignment provides incremental accuracy improvement (+0.1-0.5% AP3D) on top of geometric construction. It depends on geometric construction being implemented first.

**Independent Test**: Can be tested by comparing depth error (in meters) on KITTI validation before and after dense alignment. Measurable via absolute depth error reduction.

**Acceptance Scenarios**:

1. **Given** an initial 3D box estimate from geometric construction and the original stereo image pair, **When** dense alignment is applied, **Then** the system searches depth hypotheses within ±2 meters and selects the depth with minimum photometric error.

2. **Given** an object classified as heavily occluded, **When** the system processes it, **Then** dense alignment is skipped and the geometric construction result is used directly.

3. **Given** dense alignment is enabled during inference, **When** processing a stereo image pair, **Then** inference speed remains above 20 FPS on a modern GPU.

---

### User Story 3 - Correct Peak Detection with Max Pooling NMS (Priority: P3)

As an ML practitioner running inference, I need the detection pipeline to use 3×3 max pooling for non-maximum suppression on heatmaps, so that only true local maxima are selected as detection centers.

**Why this priority**: This is a fundamental correctness issue. Without proper NMS, the current topk selection may pick non-peak locations leading to duplicate or shifted detections. Low implementation effort, high correctness impact.

**Independent Test**: Can be tested by visualizing detection centers on heatmap. Before: multiple detections per object. After: single detection at true peak.

**Acceptance Scenarios**:

1. **Given** a detection heatmap after sigmoid activation, **When** NMS is applied, **Then** only local maxima (pixels equal to their 3×3 neighborhood maximum) remain non-zero.

2. **Given** two adjacent high-scoring peaks in a heatmap, **When** NMS is applied, **Then** both peaks are preserved if they are both local maxima in their respective neighborhoods.

3. **Given** a smooth heatmap region without clear peak, **When** NMS is applied and topk selects from the result, **Then** no detections are produced from that region (all values suppressed).

---

### User Story 4 - Higher Accuracy with DLA-34 Backbone (Priority: P4)

As an ML researcher seeking best possible accuracy, I need the option to use DLA-34 backbone with deformable convolutions, so that I can achieve the paper's top-line results (+8.73% AP3D over ResNet-18).

**Why this priority**: DLA-34 provides significant accuracy improvement but is optional since ResNet-18 already provides a working baseline. This is an enhancement for users who prioritize accuracy over speed.

**Independent Test**: Can be tested by training with DLA-34 backbone and comparing AP3D metrics against ResNet-18. Expected: AP3D@0.7 (Moderate) increases from ~32% to ~41%.

**Acceptance Scenarios**:

1. **Given** a configuration specifying DLA-34 backbone, **When** the model is initialized, **Then** the shared backbone uses DLA-34 architecture with appropriate output channels.

2. **Given** a model with DLA-34 backbone, **When** training on KITTI training set, **Then** the model converges and validation metrics improve over epochs.

3. **Given** a trained DLA-34 model, **When** running inference, **Then** speed is at least 20 FPS on a modern GPU (acceptable trade-off for accuracy).

---

### User Story 5 - Correct Handling of Occluded Objects (Priority: P5)

As an ML researcher, I need the system to classify objects as occluded or unoccluded before applying dense alignment, so that heavily occluded objects don't get incorrect depth refinement.

**Why this priority**: Occlusion screening prevents incorrect optimization of heavily occluded objects. Depends on dense alignment being implemented first.

**Independent Test**: Can be tested by visualizing occlusion classifications on KITTI images and comparing against ground truth occlusion labels.

**Acceptance Scenarios**:

1. **Given** multiple detected objects with known depths and 2D bounding boxes, **When** occlusion classification runs, **Then** objects whose left and right boundaries are both occluded by closer objects are classified as "occluded".

2. **Given** an object classified as occluded, **When** the inference pipeline processes it, **Then** dense alignment is skipped for that object.

3. **Given** objects at varying depths in a scene, **When** occlusion classification runs, **Then** the depth ordering is correctly determined using the depth-line algorithm from the paper.

---

### User Story 6 - Automatic Loss Weight Tuning (Priority: P6)

As an ML researcher training the model, I need the multi-task loss to automatically balance weights using uncertainty estimation, so that I don't need to manually tune loss weights.

**Why this priority**: The uncertainty weighting mechanism already exists in the codebase but is disabled. Enabling it is low effort and may improve training stability.

**Independent Test**: Can be tested by training with uncertainty weighting enabled and observing loss convergence behavior and final metrics.

**Acceptance Scenarios**:

1. **Given** uncertainty weighting is enabled in configuration, **When** training runs, **Then** each loss component is weighted by learned uncertainty parameters that adapt during training.

2. **Given** a training run with uncertainty weighting, **When** observing loss curves, **Then** individual loss components are balanced (no single loss dominates).

---

### Edge Cases

- What happens when geometric construction fails to converge within max iterations?
  - Fallback to simple stereo triangulation result
- How does the system handle objects at extreme depths (very close < 2m or very far > 70m)?
  - Apply depth clipping and mark as low confidence
- What happens when stereo baseline information is missing from calibration?
  - Raise clear error during model initialization, not at inference time
- How does dense alignment behave when left/right images have significant exposure differences?
  - Use normalized cross-correlation (NCC) which is robust to brightness changes
- What happens when all detected objects are classified as occluded?
  - All objects skip dense alignment; geometric construction results are used directly

## Requirements _(mandatory)_

### Functional Requirements

#### Core Accuracy (HIGH Priority)

- **FR-001**: System MUST implement geometric construction that solves for 3D box center (x, y, z) and orientation (θ) using 7 constraint equations from stereo observations via Gauss-Newton optimization.

- **FR-002**: System MUST implement 3×3 max pooling NMS on detection heatmaps before selecting top-k detections, ensuring only local maxima are considered as detection centers.

- **FR-003**: System MUST select perspective keypoints from the 4 predicted bottom vertices based on orientation angle, following the paper's quadrant-based selection rules.

#### Depth Refinement (HIGH Priority)

- **FR-004**: System MUST implement dense photometric alignment that refines depth by searching within a configurable range (default ±2 meters) and selecting the depth with minimum photometric error.

- **FR-005**: System MUST support both NCC (Normalized Cross-Correlation) and SAD with zero-mean normalization for photometric error computation, with NCC as the default. Configuration option MUST allow switching between methods for ablation studies.

#### Occlusion Handling (MEDIUM Priority)

- **FR-006**: System MUST classify objects as occluded or unoccluded using depth-line analysis based on 2D bounding box boundaries and estimated depths.

- **FR-007**: System MUST skip dense alignment for objects classified as heavily occluded.

#### Backbone Options (MEDIUM Priority)

- **FR-008**: System MUST support DLA-34 backbone as an alternative to ResNet-18, with appropriate output channel handling for the detection head.

- **FR-009**: System MUST maintain weight sharing between left and right image feature extraction regardless of backbone choice.

#### Training Enhancements (MEDIUM Priority)

- **FR-010**: System MUST support enabling/disabling uncertainty-weighted multi-task loss via configuration.

#### Integration Requirements

- **FR-011**: System MUST maintain backward compatibility with existing trained models when new components are added as optional features.

- **FR-012**: System MUST provide clear configuration options to enable/disable each new component independently for ablation studies.

- **FR-013**: System MUST log which optional components are enabled during training and inference for reproducibility.

- **FR-014**: System MUST apply a configurable confidence threshold (default: 0.3) to filter detections before geometric construction and dense alignment processing.

### Key Entities

- **GeometricConstruction**: Solver that takes 2D observations (box corners, disparity, perspective keypoints), 3D dimensions, initial orientation, and camera calibration to produce refined 3D box parameters (x, y, z, θ).

- **DenseAlignment**: Refinement module that takes stereo images, initial 3D box, and calibration to produce refined depth via photometric matching.

- **OcclusionClassifier**: Module that takes list of detected objects with 2D boxes and depths to produce occlusion classifications.

- **PerspectiveKeypoints**: Selection logic that takes 4 bottom vertices and orientation to produce 2 visible perspective keypoints for geometric construction.

- **Box3D**: 3D bounding box representation containing center (x, y, z), dimensions (l, w, h), and orientation (θ).

## Success Criteria _(mandatory)_

### Measurable Outcomes

- **SC-001**: Model with geometric construction achieves AP3D@0.7 (Moderate) of at least 30% on KITTI validation set with ResNet-18 backbone.

- **SC-002**: Model with geometric construction + dense alignment achieves AP3D@0.7 (Moderate) of at least 32% on KITTI validation set with ResNet-18 backbone (paper target: 32.71%).

- **SC-003**: Model with DLA-34 backbone achieves AP3D@0.7 (Moderate) of at least 40% on KITTI validation set (paper target: 41.44%).

- **SC-004**: Inference speed with ResNet-18 backbone remains above 30 FPS on a modern GPU (RTX 3090 or equivalent).

- **SC-005**: Inference speed with DLA-34 backbone remains above 20 FPS on a modern GPU.

- **SC-006**: 3×3 max pooling NMS reduces duplicate detections per object to zero (single detection per ground truth object).

- **SC-007**: Geometric construction solver converges (residual below tolerance) on at least 95% of valid detections.

- **SC-008**: Occlusion classification accuracy matches manual labels on at least 85% of test cases.

## Assumptions

- KITTI dataset is available and properly configured with stereo calibration files.
- Training and inference are performed on CUDA-capable GPUs.
- The existing detection head architecture (10 branches) is correct per the paper and does not need modification.
- Camera intrinsics (fx, fy, cx, cy) and stereo baseline are available in calibration files.
- Per-class mean 3D dimensions (for KITTI: Car, Pedestrian, Cyclist) are correctly configured.
- The existing Gaussian heatmap generation with aspect ratio adaptation is correct per the paper.

## Dependencies

- Existing implementation of: heatmap generation, detection head, loss functions, data augmentation, KITTI data loader.
- Camera calibration parsing utilities.
- 3D IoU computation for metrics.

## Out of Scope

- KFPN (Keypoint Feature Pyramid Network) - marked LOW priority in gap analysis.
- Support for datasets other than KITTI (e.g., nuScenes, Waymo).
- Real-time deployment optimizations (TensorRT, ONNX export).
- Multi-GPU distributed training enhancements.
