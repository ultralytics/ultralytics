# Tennis Ball Motion Analysis

This directory contains motion analysis tools and visualizations for the tennis ball tracking dataset.

## Overview

We've created a comprehensive motion analysis pipeline to understand tennis ball movement patterns and prepare for YOLO11Pose adaptation. The analysis focuses on center point trajectories and motion characteristics.

## Key Findings

### Motion Characteristics
- **Average Speed**: 15.67 ± 3.61 pixels/frame
- **Maximum Speed**: 117.64 pixels/frame (observed in game4/Clip1)
- **Speed Range**: 10.17 - 23.15 pixels/frame across clips
- **Visibility Rate**: 93.9% (4,643 visible frames out of 4,943 total)

### Trajectory Analysis
- **Average Trajectory Length**: 3,756 pixels
- **Longest Trajectory**: 9,495 pixels (game6/Clip2)
- **Motion Range**: Average 561 pixels (X) × 482 pixels (Y)

### Dataset Statistics
- **Total Clips Analyzed**: 20 clips from 10 games
- **Total Frames**: 4,943 frames
- **Games with Highest Activity**: game4, game6 (longest clips and trajectories)

## Files Generated

### Analysis Scripts
- `motion_visualization.py` - Main motion analysis and visualization tool
- `quick_motion_test.py` - Simple test script for single clip analysis
- `motion_mask_demo.py` - Motion mask generation demonstration
- `motion_analysis_summary.py` - Summary statistics and insights

### Motion Utilities (NEW)
- `ultralytics/nn/modules/motion_utils.py` - **Core motion mask generation utilities**
- `test_motion_standalone.py` - Standalone test for motion utilities
- `test_flexible_motion.py` - Test for enhanced flexible motion features

### Output Files
- `motion_analysis_output/` - Directory containing all generated visualizations
  - Individual clip trajectory plots (40 files)
  - Motion analysis plots (40 files)
  - Clips comparison plot
  - Motion summary CSV with statistics
- `motion_analysis_summary.png` - Summary visualization
- `motion_mask_*.png` - Motion mask demonstrations

## Motion Mask Generation

The motion mask generation follows the pipeline specifications:
- **Pixel Threshold**: 15 (configurable, with adaptive thresholding)
- **Frame Delta**: 1 (consecutive frames)
- **Window Size**: 5 frames
- **Output**: Binary motion masks for 4-channel input

### Enhanced Motion Utilities

The new `motion_utils.py` provides flexible motion detection with:

#### Core Functions
- `generate_motion_mask()` - Basic motion mask generation
- `generate_motion_mask_batch()` - Batch processing for training
- `combine_rgb_motion()` - Create 4-channel input (RGB + motion)
- `MotionMaskGenerator` - Configurable motion mask generator

#### Advanced Features
- `MotionConfig` - Flexible configuration class
- `FlexibleMotionMaskGenerator` - Enhanced generator with adaptive features
- `adaptive_motion_threshold()` - Dynamic threshold adjustment
- `create_motion_visualization()` - Comprehensive visualization
- `precompute_motion_masks()` - Pre-computation for faster training

#### Configuration Options
- **Adaptive Thresholding**: Adjusts threshold based on frame brightness/contrast
- **Motion Validation**: Filters out noise (min/max motion pixel limits)
- **Morphological Operations**: Clean motion masks with OpenCV operations
- **Blur Filtering**: Reduce noise with Gaussian blur
- **Caching Support**: Pre-compute and cache motion masks

## Key Insights for Model Development

### 1. YOLO11Pose Adaptation
- Use 4-channel input (RGB + motion mask) for enhanced tracking
- Implement visibility loss component (similar to pose visibility)
- Consider temporal smoothing for high-speed motion (>100 pixels/frame)

### 2. Data Pipeline Recommendations
- Pre-compute motion masks for faster training
- Use window-based motion detection (5 frames)
- Implement motion-aware data augmentation
- Consider frame sampling for very long clips (>500 frames)

### 3. Model Architecture
- Modify first layer for 4-channel input (4 → 64 channels)
- Use keypoint shape `[1, 3]` for ball center (x, y, visibility)
- Focus on pose head only (no detection head for prototype)

## Usage

### Quick Analysis
```bash
python quick_motion_test.py
```

### Full Motion Analysis
```bash
python motion_visualization.py
```

### Motion Mask Demo
```bash
python motion_mask_demo.py
```

### Summary Analysis
```bash
python motion_analysis_summary.py
```

### Motion Utilities Testing
```bash
# Test basic motion utilities
python test_motion_standalone.py

# Test enhanced flexible features
python test_flexible_motion.py
```

### Using Motion Utilities in Code
```python
from ultralytics.nn.modules.motion_utils import (
    MotionConfig, 
    FlexibleMotionMaskGenerator,
    generate_motion_mask,
    combine_rgb_motion
)

# Basic usage
frames = [frame1, frame2, frame3, frame4, frame5]
motion_mask = generate_motion_mask(frames, pixel_threshold=15)

# Advanced usage with flexible configuration
config = MotionConfig(
    pixel_threshold=15,
    adaptive_threshold=True,
    min_motion_pixels=100,
    max_motion_pixels=50000,
    morphological_ops=True
)

generator = FlexibleMotionMaskGenerator(config)
enhanced_mask = generator.generate_enhanced(frames)

# Create 4-channel input for YOLO11Pose
rgb_frame = frames[0]  # Single RGB frame
combined_input = combine_rgb_motion(rgb_frame, enhanced_mask)
```

## Next Steps

Based on this analysis, the recommended next steps are:

1. **Phase 1**: ✅ **COMPLETED** - Implement motion mask generation utilities
2. **Phase 2**: Create custom dataloader for 4-channel inputs
3. **Phase 3**: Modify YOLO11Pose for 4-channel input
4. **Phase 4**: Implement keypoint + visibility loss functions
5. **Phase 5**: Set up training pipeline with motion-aware augmentation

### Current Status
- ✅ **Motion Analysis**: Complete analysis of 20 clips (4,943 frames)
- ✅ **Motion Utilities**: Flexible motion mask generation with adaptive features
- ✅ **Visualization**: Comprehensive motion visualization and testing
- ✅ **Documentation**: Complete documentation and usage examples
- 🔄 **Next**: Custom dataloader implementation for 4-channel inputs

## Data Format

The analysis works with CSV files containing:
- `file name`: Frame filename (e.g., "0000.jpg")
- `visibility`: 0=can't see, 1=visible, 2=hard to see, 3=occluded
- `x-coordinate`: Ball center X position
- `y-coordinate`: Ball center Y position  
- `status`: 0=flying, 1=hit, 2=bouncing

## Performance Notes

- Motion analysis processes 20 clips (4,943 frames) in ~30 seconds
- Motion mask generation is optimized for real-time processing
- All visualizations are saved as high-resolution PNG files (300 DPI)

This analysis provides the foundation for implementing the tennis ball motion detection pipeline as outlined in `Dataset/PIPELINE.md`.
