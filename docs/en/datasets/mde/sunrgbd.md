# SUN RGB-D for Object-Level Depth Estimation

## Overview

The SUN RGB-D dataset is a comprehensive RGB-D scene understanding benchmark suite containing 10,335 indoor scene images with corresponding depth maps, 2D and 3D bounding boxes, and semantic labels for 37 object categories. It is widely used for depth estimation, object detection, and scene understanding tasks.

<p align="center">
  <img width="640" src="https://rgbd.cs.princeton.edu/teaser.jpg" alt="SUN RGB-D Dataset Examples">
</p>

## Dataset Structure

The SUN RGB-D dataset is organized by sensor type (Kinect v1, Kinect v2, RealSense, Xtion) and includes:

- **RGB images**: High-quality color images from various indoor scenes
- **Depth maps**: Aligned depth information captured by RGB-D sensors
- **2D bounding boxes**: Object locations and class labels in image coordinates
- **3D bounding boxes**: 3D object locations in camera coordinates
- **Scene labels**: Room type and scene category information
- **Camera intrinsics**: Calibration parameters for each image

## Key Features

- **Size**: 10,335 RGB-D images
- **Scenes**: 47 different scene categories
- **Objects**: 37 common indoor object classes
- **Depth range**: Typically 0-10 meters for indoor scenes
- **Resolution**: Variable (commonly 640×480)
- **Sensors**: Kinect v1/v2, Intel RealSense, ASUS Xtion PRO LIVE

## Object Classes (37 Categories)

The dataset includes annotations for 37 common indoor object classes:

| ID | Class | ID | Class | ID | Class | ID | Class |
|----|-------|----|----|----|----|----|----|
| 0 | bed | 10 | person | 20 | bin | 30 | window |
| 1 | table | 11 | cabinet | 21 | sink | 31 | blinds |
| 2 | sofa | 12 | box | 22 | books | 32 | shelves |
| 3 | chair | 13 | pillow | 23 | curtain | 33 | picture |
| 4 | toilet | 14 | door | 24 | mirror | 34 | counter |
| 5 | desk | 15 | tv | 25 | floor_mat | 35 | floor |
| 6 | dresser | 16 | lamp | 26 | clothes | 36 | wall |
| 7 | night_stand | 17 | bag | 27 | ceiling | | |
| 8 | bookshelf | 18 | computer | 28 | book | | |
| 9 | bathtub | 19 | monitor | 29 | fridge | | |

## Dataset Preparation

### Automatic Preparation

Use the built-in preparation script to automatically download and convert the dataset:

```python
from ultralytics.data.prepare_sunrgbd_mde import prepare_sunrgbd_mde_dataset

# Prepare dataset with default settings
summary = prepare_sunrgbd_mde_dataset(
    dataset_dir="datasets/sunrgbd_mde",
    depth_max=10.0,
    download_assets=True,
)
```

Or use the command-line example script:

```bash
python examples/prepare_sunrgbd_mde.py --download --dataset-dir datasets/sunrgbd_mde
```

### Manual Preparation

If you prefer to prepare the dataset manually:

1. **Download the dataset**:
   ```bash
   wget http://rgbd.cs.princeton.edu/data/SUNRGBD.zip
   wget http://rgbd.cs.princeton.edu/data/SUNRGBDMeta2DBB_v2.mat
   wget http://rgbd.cs.princeton.edu/data/SUNRGBDMeta3DBB_v2.mat
   wget http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip
   ```

2. **Extract the files**:
   ```bash
   unzip SUNRGBD.zip
   unzip SUNRGBDtoolbox.zip
   ```

3. **Run the preparation script**:
   ```python
   from ultralytics.data.prepare_sunrgbd_mde import prepare_sunrgbd_mde_dataset
   
   summary = prepare_sunrgbd_mde_dataset(
       dataset_dir="datasets/sunrgbd_mde",
       depth_max=10.0,
       download_assets=False,  # Already downloaded
       use_existing_splits=True,
   )
   ```

### Dataset Structure After Preparation

After running the preparation script, your dataset will be organized as follows:

```
datasets/sunrgbd_mde/
├── train/
│   ├── images/         # RGB images
│   │   ├── kv1_scene_0001.jpg
│   │   ├── kv1_scene_0002.jpg
│   │   └── ...
│   └── labels/         # YOLO MDE format labels
│       ├── kv1_scene_0001.txt
│       ├── kv1_scene_0002.txt
│       └── ...
└── val/
    ├── images/         # RGB images
    │   └── ...
    └── labels/         # YOLO MDE format labels
        └── ...
```

### Label Format

Each label file contains object-level depth annotations in YOLO MDE format:

```
class_id x_center y_center width height depth_normalized
```

Where:
- `class_id`: Object class (0-36)
- `x_center, y_center`: Normalized bounding box center (0-1)
- `width, height`: Normalized bounding box dimensions (0-1)
- `depth_normalized`: Normalized depth value (0-1), where 1.0 corresponds to `depth_max` meters

## Training

### Train a YOLO11 Model for Object-Level Depth Estimation

```python
from ultralytics import YOLO

# Load a pretrained YOLO11 model
model = YOLO("yolo11n.pt")

# Train on SUN RGB-D
results = model.train(
    data="sunrgbd-mde.yaml",
    task="mde",
    epochs=100,
    imgsz=640,
    batch=16,
)
```

### Command Line Training

```bash
yolo train model=yolo11n.pt data=sunrgbd-mde.yaml task=mde epochs=100 imgsz=640
```

## Inference

### Predict Object Depths

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/mde/train/weights/best.pt")

# Predict on new images
results = model.predict(
    source="path/to/indoor_scene.jpg",
    save=True,
    conf=0.25,
)

# Access predictions
for result in results:
    boxes = result.boxes  # Bounding boxes
    depths = result.depths  # Predicted depths for each object
```

## Configuration Options

The preparation script supports various configuration options:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dataset_dir` | Required | Output directory for prepared dataset |
| `depth_max` | 10.0 | Maximum depth in meters for normalization |
| `train_ratio` | 0.8 | Train/val split ratio (if not using official splits) |
| `download_assets` | True | Download dataset if not present |
| `overwrite` | False | Overwrite existing prepared dataset |
| `use_existing_splits` | True | Use official train/test splits |
| `min_box_size` | 10 | Minimum bounding box size in pixels |
| `seed` | 0 | Random seed for splitting |

## Example Usage

### Basic Preparation

```python
from ultralytics.data.prepare_sunrgbd_mde import prepare_sunrgbd_mde_dataset

summary = prepare_sunrgbd_mde_dataset("datasets/sunrgbd_mde")
print(f"Training images: {summary['train']['images']}")
print(f"Validation images: {summary['val']['images']}")
```

### Custom Configuration

```python
summary = prepare_sunrgbd_mde_dataset(
    dataset_dir="datasets/sunrgbd_custom",
    depth_max=8.0,  # Use 8m as max depth
    min_box_size=20,  # Filter out very small boxes
    use_existing_splits=False,  # Create custom splits
    train_ratio=0.85,
    seed=42,
)
```

## Citations and References

If you use the SUN RGB-D dataset in your research, please cite:

```bibtex
@inproceedings{song2015sun,
  title={SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite},
  author={Song, Shuran and Lichtenberg, Samuel P and Xiao, Jianxiong},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={567--576},
  year={2015}
}
```

## Additional Resources

- **Official Website**: [https://rgbd.cs.princeton.edu/](https://rgbd.cs.princeton.edu/)
- **Paper**: [SUN RGB-D: A RGB-D Scene Understanding Benchmark Suite](https://rgbd.cs.princeton.edu/paper.pdf)
- **Toolbox**: [SUNRGBDtoolbox (MATLAB)](http://rgbd.cs.princeton.edu/data/SUNRGBDtoolbox.zip)

## FAQ

**Q: How large is the SUN RGB-D dataset?**  
A: The complete dataset is approximately 18 GB compressed (SUNRGBD.zip is ~18 GB).

**Q: What depth range should I use?**  
A: For indoor scenes, 10 meters is typically sufficient. You can adjust based on your specific use case.

**Q: Can I use this for dense depth estimation?**  
A: This preparation script focuses on object-level depth estimation. For dense depth estimation, you would need to modify the approach to work with full depth maps rather than object bounding boxes.

**Q: What's the difference between KITTI and SUN RGB-D?**  
A: KITTI focuses on outdoor autonomous driving scenes with depths up to 80-100m, while SUN RGB-D focuses on indoor scenes with depths up to ~10m. SUN RGB-D has more object classes suitable for indoor environments.

**Q: How do I handle missing annotations?**  
A: The preparation script automatically handles missing annotations by creating empty label files for images without annotations.

