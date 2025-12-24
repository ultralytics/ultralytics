---
comments: true
description: Learn about the YOLO 3D Stereo dataset format for training stereo-based 3D object detection models. Understand the label format, coordinate systems, and annotation structure.
keywords: KITTI stereo, 3D object detection, stereo vision, YOLO 3D format, dataset labeling, camera coordinates, depth estimation
---

# KITTI Stereo 3D Detection Dataset Format

The YOLO 3D Stereo format is a specialized dataset format for training stereo-based 3D object detection models. This format extends the standard YOLO format to include stereo image pairs, 3D bounding box annotations, and camera calibration data.

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

Each label file (`.txt`) corresponds to one stereo image pair and contains one line per object. The format supports two versions:

### Standard Format (22 values)

```
class x_l y_l w_l h_l x_r w_r dim_h dim_w dim_l alpha v1_x v1_y v2_x v2_y v3_x v3_y v4_x v4_y X Y Z
```

### Legacy Format (19 values)

```
class x_l y_l w_l h_l x_r w_r dim_h dim_w dim_l alpha v1_x v1_y v2_x v2_y v3_x v3_y v4_x v4_y
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
- **w_r**: Right image bounding box width (normalized 0-1)
- **Note**: Right image uses same y-coordinate and height as left (epipolar constraint)

### 3D Dimensions (meters)
- **dim_h**: Object height in meters
- **dim_w**: Object width in meters
- **dim_l**: Object length in meters
- **Note**: Dimensions are in camera coordinate system (height=Y, width=Z, length=X)

### Observation Angle
- **alpha**: Observation angle in radians (range: [-π, π])
- **Description**: Angle between object orientation and camera ray in image plane
- **Note**: This is converted to global yaw (rotation_y) during training/inference

### Bottom 4 Vertices (normalized)
- **v1_x, v1_y**: First bottom vertex (normalized 0-1)
- **v2_x, v2_y**: Second bottom vertex (normalized 0-1)
- **v3_x, v3_y**: Third bottom vertex (normalized 0-1)
- **v4_x, v4_y**: Fourth bottom vertex (normalized 0-1)
- **Description**: Projected 2D coordinates of the 4 bottom corners of the 3D bounding box

### 3D Location (camera coordinates, meters) - Standard Format Only
- **X**: X-coordinate in camera frame (right = positive)
- **Y**: Y-coordinate in camera frame (down = positive, bottom center of box)
- **Z**: Z-coordinate in camera frame (forward = positive, depth)
- **Note**: Y is at bottom center of box (KITTI convention). The geometric center is computed during parsing.

## Coordinate Systems

### Camera Coordinate System
- **X-axis**: Points right (positive to the right)
- **Y-axis**: Points down (positive downward)
- **Z-axis**: Points forward (positive into the scene)
- **Origin**: Camera optical center

### 3D Bounding Box Convention
- **Location**: Bottom center of the box (Y coordinate)
- **Dimensions**: (height, width, length) in meters
- **Orientation**: Rotation around Y-axis (vertical axis) in radians
- **Rotation_y = 0**: Object faces camera X direction (forward along X-axis)

## Example Label Line

```
0 0.491935 0.461333 0.193548 0.293333 0.478226 0.193548 1.52 1.73 3.89 0.1234 0.395 0.461 0.589 0.461 0.589 0.754 0.395 0.754 2.8 1.6 7.6
```

Breaking down this example:
- **Class**: `0` (Car)
- **Left box**: center=(0.491935, 0.461333), size=(0.193548, 0.293333)
- **Right box**: center_x=0.478226, width=0.193548
- **Dimensions**: height=1.52m, width=1.73m, length=3.89m
- **Alpha**: 0.1234 radians
- **Vertices**: 4 bottom corners in normalized coordinates
- **3D location**: X=2.8m, Y=1.6m, Z=7.6m

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
- **P2**: Left camera projection matrix (3×4)
- **P3**: Right camera projection matrix (3×4)
- **fx, fy**: Focal lengths in pixels
- **cx, cy**: Principal point coordinates
- **baseline**: Stereo baseline in meters

## Dataset YAML Configuration

The dataset YAML file should specify:

```yaml
path: /path/to/dataset
train: images/train/left
val: images/val/left
train_right: images/train/right
val_right: images/val/right

names:
  0: Car
  1: Pedestrian
  2: Cyclist
```

## Usage

To train a stereo 3D detection model:

!!! example

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo11-stereo3ddet.yaml")
        results = model.train(data="kitti-stereo.yaml", epochs=100, imgsz=384)
        ```

    === "CLI"

        ```bash
        yolo task=stereo3ddet train data=kitti-stereo.yaml model=yolo11-stereo3ddet.yaml epochs=100 imgsz=384
        ```

## Important Notes

1. **Normalized Coordinates**: All 2D coordinates (x_l, y_l, w_l, h_l, x_r, w_r, vertices) are normalized to [0, 1] relative to image dimensions.

2. **Coordinate System**: 3D coordinates use camera coordinate system with Y pointing down (KITTI convention).

3. **Box Center**: The Y coordinate represents the bottom center of the 3D box, not the geometric center. The geometric center is computed during parsing.

4. **Epipolar Constraint**: Right image y-coordinate and height are the same as left image due to stereo rectification.

5. **Observation Angle vs. Global Yaw**: The `alpha` (observation angle) is converted to `rotation_y` (global yaw) using: `rotation_y = alpha + arctan(x/z)`.

6. **Dimensions Order**: Dimensions are stored as (height, width, length) but used as (length, width, height) in Box3D objects.

## Conversion from KITTI Format

To convert from original KITTI format to YOLO 3D format, use the conversion script:

```bash
python ultralytics/data/scripts/convert_kitti_3d.py \
    --kitti-root /path/to/kitti_raw \
    --output-root /path/to/kitti-stereo \
    --split-strategy 3dop
```

This will create the proper directory structure and convert all annotations to the YOLO 3D format.

