---
comments: true
description: Learn about the YOLO 3D Stereo dataset format for training stereo-based 3D object detection models. Understand the label format, coordinate systems, and annotation structure.
keywords: KITTI stereo, 3D object detection, stereo vision, YOLO 3D format, dataset labeling, camera coordinates, depth estimation
---

# KITTI Stereo 3D Detection Dataset Format

<img width="1024" src="https://cdn.jsdelivr.net/gh/ultralytics/assets@main/docs/kitti-stereo-sample.avif" alt="KITTI stereo dataset sample showing annotated left image and right image pair">

The YOLO 3D Stereo format is a specialized dataset format for training stereo-based 3D object detection models. This format extends the standard YOLO format to include stereo image pairs, 3D bounding box annotations, and camera calibration data.

## Sample Images

The image above shows a KITTI stereo pair: the **left image** (top) with 2D bounding box annotations for Cars, and the **right image** (bottom) from the same timestamp. The horizontal offset between objects in left and right views encodes depth — closer objects have larger disparity.

## Dataset Structure

A YOLO 3D Stereo dataset should be organized as follows:

```
dataset_root/
├── images/
│   ├── train/
│   │   ├── left/          # Left camera images
│   │   └── right/         # Right camera images
│   └── val/
│       ├── left/
│       └── right/
├── labels/
│   ├── train/             # Training labels (one .txt file per image)
│   └── val/               # Validation labels
└── calib/
    ├── train/             # Camera calibration files
    └── val/
```

## Label Format

Each label file (`.txt`) corresponds to one stereo image pair and contains one line per object. The format uses 18 values per line:

```
class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y truncated occluded
```

## Field Descriptions

### Class ID

- **Type**: Integer
- **Description**: Object class identifier (zero-indexed)
- **Example**: `0` for Car, `1` for Pedestrian, `2` for Cyclist

### Left Image 2D Bounding Box (normalized)

- **x_l**: Left image center x-coordinate (normalized 0-1)
- **y_l**: Left image center y-coordinate (normalized 0-1)
- **w_l**: Left image bounding box width (normalized 0-1)
- **h_l**: Left image bounding box height (normalized 0-1)

### Right Image 2D Bounding Box (normalized)

- **x_r**: Right image center x-coordinate (normalized 0-1)
- **y_r**: Right image center y-coordinate (normalized 0-1)
- **w_r**: Right image bounding box width (normalized 0-1)
- **h_r**: Right image bounding box height (normalized 0-1)
- **Note**: Due to the epipolar constraint, y_r and h_r are typically the same as y_l and h_l.

### 3D Dimensions (meters)

- **dim_l**: Object length in meters (forward direction)
- **dim_w**: Object width in meters (lateral direction)
- **dim_h**: Object height in meters (vertical direction)
- **Note**: Stored as `[length, width, height]` in label files. Internally converted to `(H, W, L)` order for training.

### 3D Location (camera coordinates, meters)

- **loc_x**: X-coordinate in camera frame (right = positive)
- **loc_y**: Y-coordinate in camera frame (down = positive, bottom center of box)
- **loc_z**: Z-coordinate in camera frame (forward = positive, depth)
- **Note**: Y is at bottom center of box (KITTI convention). The geometric center is computed during parsing as `y_center = y_bottom - height/2`.

### Orientation

- **rot_y**: Global rotation around Y-axis in radians (range: [-π, π])
- **Description**: Direct global yaw angle (rotation around vertical Y-axis, which points down)
- **Rotation_y = 0**: Object faces along the camera Z-axis (forward direction)

### Truncation and Occlusion

- **truncated**: Float value [0.0, 1.0] indicating truncation level where 0 = fully visible (object fully within image boundaries) and 1 = fully truncated (object leaving image boundaries)
- **occluded**: Integer value indicating occlusion level:
    - `0` = fully visible
    - `1` = partly occluded
    - `2` = largely occluded
    - `3` = unknown/fully occluded

## Coordinate Systems

### Camera Coordinate System

- **X-axis**: Points right (positive to the right)
- **Y-axis**: Points down (positive downward)
- **Z-axis**: Points forward (positive into the scene)
- **Origin**: Camera optical center

### 3D Bounding Box Convention

- **Location**: Bottom center of the box (Y coordinate)
- **Dimensions**: (length, width, height) in meters in label files
- **Orientation**: Rotation around Y-axis (vertical axis) in radians
- **Rotation_y = 0**: Object faces along Z-axis (forward into the scene)

## Example Label Line

```
0 0.739219 0.739093 0.256667 0.505120 0.681871 0.880345 0.318305 0.489783 3.580000 1.710000 1.490000 2.810000 1.600000 7.590000 -1.610000 0.000000 0
```

Breaking down:

- **Class**: `0` (Car)
- **Left box**: center=(0.739, 0.739), size=(0.257, 0.505)
- **Right box**: center=(0.682, 0.880), size=(0.318, 0.490)
- **Dimensions**: length=3.58m, width=1.71m, height=1.49m
- **3D location**: X=2.81m, Y=1.6m (bottom center), Z=7.59m
- **Rotation_y**: -1.61 radians (global yaw)
- **Truncated**: 0.0 (fully visible)
- **Occluded**: 0 (fully visible)

## Calibration File Format

Each calibration file (`.txt`) contains camera intrinsic and extrinsic parameters:

```
fx: 721.5377
fy: 721.5377
cx: 609.5593
cy: 172.8540
baseline: 0.54
image_width: 1242
image_height: 375
```

Or in original KITTI format:

```
P2: 7.215377e+02 0.000000e+00 6.095593e+02 0.000000e+00 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
P3: 7.215377e+02 0.000000e+00 6.095593e+02 -3.875744e+02 0.000000e+00 7.215377e+02 1.728540e+02 0.000000e+00 0.000000e+00 0.000000e+00 1.000000e+00 0.000000e+00
```

Where:

- **P2**: Left camera projection matrix (3x4)
- **P3**: Right camera projection matrix (3x4)
- **fx, fy**: Focal lengths in pixels
- **cx, cy**: Principal point coordinates
- **baseline**: Stereo baseline in meters (computed from P2/P3)

## Dataset YAML Configuration

The dataset YAML file should specify:

```yaml
path: /path/to/dataset
train: images/train/left
val: images/val/left
train_right: images/train/right
val_right: images/val/right

stereo: true
channels: 6 # left RGB + right RGB

names:
    0: Car
    1: Pedestrian
    2: Cyclist

baseline: 0.54 # stereo baseline in meters (fallback when calib files missing)

# Mean dimensions per class [length, width, height] in meters
# Internally converted to (H, W, L) order for training
mean_dims:
    Car: [3.9, 1.6, 1.5]
    Pedestrian: [0.8, 0.6, 1.7]
    Cyclist: [1.8, 0.6, 1.7]

# Standard deviation of dimensions per class [length, width, height] in meters
std_dims:
    Car: [0.42, 0.10, 0.15]
    Pedestrian: [0.20, 0.08, 0.12]
    Cyclist: [0.25, 0.10, 0.15]
```

## Usage

To train a stereo 3D detection model:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26-s3d.yaml")
        results = model.train(data="kitti-stereo.yaml", epochs=1000, imgsz=[384, 1248])
        ```

    === "CLI"

        ```bash
        yolo task=s3d train data=kitti-stereo.yaml model=yolo26-s3d.yaml epochs=1000 imgsz=384,1248
        ```

## Important Notes

1. **Normalized Coordinates**: All 2D coordinates (x_l, y_l, w_l, h_l, x_r, y_r, w_r, h_r) are normalized to [0, 1] relative to image dimensions.

2. **Coordinate System**: 3D coordinates use camera coordinate system with Y pointing down (KITTI convention).

3. **Box Center**: The Y coordinate in labels represents the bottom center of the 3D box, not the geometric center. The geometric center is computed during parsing.

4. **Dimensions Order**: Labels store `[length, width, height]`. The code internally reorders to `(H, W, L)` for training. The YAML `mean_dims` and `std_dims` also use `[L, W, H]` order and are automatically converted.

5. **Orientation**: `rotation_y` is the global yaw angle in radians. The observation angle `alpha` is computed internally as `alpha = rotation_y - atan2(x, z)` for encoding.

6. **Truncation and Occlusion**: Used for KITTI R40 difficulty classification (Easy/Moderate/Hard) during evaluation.

7. **Format Compatibility**: The parser accepts both 18-value (current) and legacy 26-value label formats. Use the conversion script below to generate labels from raw KITTI data.

## Conversion from KITTI Format

To convert from original KITTI format to YOLO 3D format, use the conversion script:

```bash
# Convert all classes
python ultralytics/data/scripts/convert_kitti_3d.py --kitti-root /path/to/kitti_raw

# Convert only specific classes (e.g., Car, Pedestrian, Cyclist)
python ultralytics/data/scripts/convert_kitti_3d.py --kitti-root /path/to/kitti_raw --filter-classes Car Pedestrian Cyclist
```

The script will:

- Process the KITTI training split
- Use 3DOP strategy: indices 0-3711 -> train, 3712+ -> val
- Output converted dataset to the same directory as `--kitti-root`
- Include all classes by default, or only specified classes if `--filter-classes` is used
- Available classes: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc

This will create the proper directory structure and convert all annotations to the YOLO 3D format (18 values per object, including truncated and occluded attributes). When using `--filter-classes`, class IDs will be remapped to be consecutive starting from 0.
