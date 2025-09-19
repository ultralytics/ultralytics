# MDE (Monocular Depth Estimation) for YOLO

This module provides Monocular Depth Estimation capabilities for YOLO models, allowing simultaneous object detection and depth estimation.

## Overview

The MDE implementation extends YOLO to predict:
- **Bounding boxes** for object detection
- **Class probabilities** for classification  
- **Depth values** for each detected object

## Architecture

### Detect_MDE Head

The `Detect_MDE` class extends the standard YOLO detection head with an additional depth estimation branch:

```python
class Detect_MDE(nn.Module):
    def __init__(self, nc=80, ch=(), reg_max=16, beta=-14.4):
        # Box regression branch (cv2)
        # Classification branch (cv3) 
        # Depth estimation branch (cv_depth)
```

**Key Features:**
- **3 prediction branches**: box regression, classification, depth estimation
- **Log-sigmoid activation**: `fd = β * log(sigmoid(Od))` where β = -14.4
- **Depth normalization**: Outputs normalized depth values [0, 1]

### MDE Model

The `MDE` class provides a complete model that integrates the MDE head with YOLO backbone:

```python
model = MDE('yolov8n-mde.yaml', nc=5)  # 5 classes for KITTI
```

## Training

### MDELoss

Combines detection loss with depth estimation loss:

```python
class MDELoss(nn.Module):
    def __init__(self, model, depth_loss_weight=1.0, depth_loss_type='l1'):
        self.det_loss = v8DetectionLoss(model)  # Detection loss
        self.depth_criterion = nn.L1Loss()      # Depth loss
```

**Loss Components:**
- **Detection Loss**: Standard YOLO detection loss (box + classification)
- **Depth Loss**: L1/L2/SmoothL1 loss for depth estimation
- **Combined Loss**: `total_loss = det_loss + weight * depth_loss`

### MDETrainer

Specialized trainer for MDE models:

```python
trainer = MDETrainer(cfg, overrides)
trainer.train()
```

## Dataset Format

### Input Format

The MDE model expects YOLO format labels with an additional depth channel:

```
class_id x_center y_center width height depth
```

Where:
- `class_id`: Object class (0=Car, 1=Pedestrian, etc.)
- `x_center, y_center, width, height`: Normalized coordinates [0, 1]
- `depth`: Normalized depth value [0, 1] (actual_depth / max_depth)

### KITTI Dataset

The implementation includes support for KITTI dataset:

```yaml
# kitti-mde.yaml
path: /path/to/kitti_yolo_depth
train: images
val: images
nc: 5
names: ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck']
depth_max: 100.0
```

## Usage

### 1. Prepare Dataset

```python
from ultralytics.models.yolo.depth.prepare_kitti_depth import prepare_kitti_depth_labels

# Convert KITTI to YOLO format with depth
prepare_kitti_depth_labels(
    kitti_root="/path/to/kitti",
    output_dir="/path/to/kitti_yolo_depth",
    max_depth=100.0
)
```

### 2. Train Model

```python
from ultralytics.models.yolo.depth import MDE

# Create model
model = MDE('yolov8n-mde.yaml', nc=5)

# Train
results = model.train(
    data='kitti-mde.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

### 3. Inference

```python
# Load trained model
model = MDE('kitti_mde_model.pt')

# Run inference
results = model('image.jpg')

# Access predictions
for result in results:
    boxes = result.boxes
    depths = boxes.data[:, -1]  # Last channel is depth
    print(f"Detected {len(boxes)} objects with depths: {depths}")
```

## Configuration Files

### Model Configuration

```yaml
# yolov8n-mde.yaml
nc: 5
depth_multiple: 0.33
width_multiple: 0.25

backbone:
  # YOLOv8 backbone layers...

head:
  # YOLOv8 head layers...
  - [[15, 18, 21], 1, Detect_MDE, [nc]]  # MDE head
```

### Dataset Configuration

```yaml
# kitti-mde.yaml
path: /path/to/dataset
train: images
val: images
nc: 5
names: ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck']
depth_max: 100.0
depth_loss_weight: 1.0
depth_loss_type: 'l1'
```

## Performance

### KITTI Dataset Results

- **Training samples**: 7,481 images
- **Total objects**: 38,675 annotations
- **Depth range**: 0.00m - 99.58m
- **Mean depth**: 26.08m
- **Classes**: 5 (Car, Pedestrian, Cyclist, Van, Truck)

### Model Specifications

- **Input size**: 640x640
- **Output channels**: nc + 64 + 1 (classes + box_regression + depth)
- **Activation**: Log-sigmoid for depth
- **Loss**: Combined detection + depth loss

## Files Structure

```
ultralytics/models/yolo/depth/
├── __init__.py          # Module exports
├── mde_head.py          # Detect_MDE head implementation
├── mde_model.py         # Complete MDE model
├── train.py             # Training utilities
├── nyu.py               # NYU dataset support
└── README.md            # This file
```

## Examples

### Complete Training Pipeline

```python
#!/usr/bin/env python3
from ultralytics.models.yolo.depth import MDE

# 1. Create model
model = MDE('yolov8n-mde.yaml', nc=5)

# 2. Train
results = model.train(
    data='kitti-mde.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device='cuda'
)

# 3. Validate
metrics = model.val()

# 4. Save
model.save('kitti_mde_model.pt')
```

### Custom Depth Loss

```python
from ultralytics.models.yolo.depth.train import MDELoss

# Custom loss with different weight
loss_fn = MDELoss(
    model=model,
    depth_loss_weight=2.0,  # Higher weight for depth
    depth_loss_type='smooth_l1'  # Different loss type
)
```

## Integration with Ultralytics

The MDE module integrates seamlessly with the Ultralytics ecosystem:

- **YOLO CLI**: Use with `yolo train` command
- **Hub**: Upload/download MDE models
- **Export**: Export to ONNX, TensorRT, etc.
- **Validation**: Standard YOLO validation metrics + depth metrics

## References

- **Paper**: Monocular Depth Estimation for Object Detection
- **Dataset**: KITTI Object Detection Dataset
- **Implementation**: Based on YOLOv8 architecture
