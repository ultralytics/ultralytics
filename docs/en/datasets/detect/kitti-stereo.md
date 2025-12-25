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

Each label file (`.txt`) corresponds to one stereo image pair and contains one line per object. The format supports multiple versions:

### Current Format (26 values)

```
class x_l y_l w_l h_l x_r y_r w_r h_r dim_l dim_w dim_h loc_x loc_y loc_z rot_y kp1_x kp1_y kp2_x kp2_y kp3_x kp3_y kp4_x kp4_y truncated occluded
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
- **y_r**: Right image center y-coordinate (normalized 0-1) - *24-value format only*
- **w_r**: Right image bounding box width (normalized 0-1)
- **h_r**: Right image bounding box height (normalized 0-1) - *24-value format only*
- **Note**: In 22-value format, only x_r and w_r are stored (y and h are same as left due to epipolar constraint)

### 3D Dimensions (meters)
- **24-value format**: `dim_l dim_w dim_h` - length, width, height in meters
- **Note**: Dimensions are in camera coordinate system (height=Y, width=Z, length=X). The order differs between formats.

### Orientation
- **24-value format**: `rot_y` - Global rotation around Y-axis in radians (range: [-π, π])
- **Description**: 
  - `rot_y` (24-value): Direct global yaw angle (rotation around vertical Y-axis)
  using: `rotation_y = alpha + arctan(x/z)`

### Bottom 4 Vertices (normalized)
- **kp1_x, kp1_y** (24-value) or **v1_x, v1_y** (22-value): First bottom vertex (normalized 0-1)
- **kp2_x, kp2_y** (24-value) or **v2_x, v2_y** (22-value): Second bottom vertex (normalized 0-1)
- **kp3_x, kp3_y** (24-value) or **v3_x, v3_y** (22-value): Third bottom vertex (normalized 0-1)
- **kp4_x, kp4_y** (24-value) or **v4_x, v4_y** (22-value): Fourth bottom vertex (normalized 0-1)
- **Description**: Projected 2D coordinates of the 4 bottom corners of the 3D bounding box in normalized image coordinates [0, 1]

### 3D Location (camera coordinates, meters)
- **loc_x** : X-coordinate in camera frame (right = positive)
- **loc_y** : Y-coordinate in camera frame (down = positive, bottom center of box)
- **loc_z** : Z-coordinate in camera frame (forward = positive, depth)
- **Note**: Y is at bottom center of box (KITTI convention). The geometric center is computed during parsing. Only present in 22+ value formats.

### Truncation and Occlusion (26-value format only)
- **truncated**: Float value [0.0, 1.0] indicating truncation level where 0 = fully visible (object fully within image boundaries) and 1 = fully truncated (object leaving image boundaries)
- **occluded**: Integer value indicating occlusion level:
  - `0` = fully visible
  - `1` = partly occluded
  - `2` = largely occluded
  - `3` = unknown/fully occluded
- **Note**: These attributes are only present in the 26-value format. They are extracted from the original KITTI labels and preserved in the YOLO format for training and evaluation purposes.

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

## Example Label Lines

### 26-Value Format Example

```
0 0.739219 0.739093 0.256667 0.505120 0.681871 0.880345 0.318305 0.489783 3.580000 1.710000 1.490000 2.810000 1.600000 7.590000 -1.610000 0.610886 0.486533 0.867552 0.486533 0.867552 0.991653 0.610886 0.991653 0.000000 0
```

Breaking down this 26-value example:
- **Class**: `0` (Car)
- **Left box**: center=(0.739219, 0.739093), size=(0.256667, 0.505120)
- **Right box**: center=(0.681871, 0.880345), size=(0.318305, 0.489783)
- **Dimensions**: length=3.58m, width=1.71m, height=1.49m
- **3D location**: X=2.81m, Y=1.6m, Z=7.59m
- **Rotation_y**: -1.61 radians (global yaw)
- **Vertices**: 4 bottom corners (kp1-kp4) in normalized coordinates
- **Truncated**: 0.000000 (fully visible, not truncated)
- **Occluded**: 0 (fully visible, not occluded)

### 22-Value Format Example

```
0 0.491935 0.461333 0.193548 0.293333 0.478226 0.193548 1.52 1.73 3.89 0.1234 0.395 0.461 0.589 0.461 0.589 0.754 0.395 0.754 2.8 1.6 7.6
```

Breaking down this 22-value example:
- **Class**: `0` (Car)
- **Left box**: center=(0.491935, 0.461333), size=(0.193548, 0.293333)
- **Right box**: center_x=0.478226, width=0.193548 (y and h same as left)
- **Dimensions**: height=1.52m, width=1.73m, length=3.89m
- **Alpha**: 0.1234 radians (observation angle)
- **Vertices**: 4 bottom corners (v1-v4) in normalized coordinates
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

1. **Normalized Coordinates**: All 2D coordinates (x_l, y_l, w_l, h_l, x_r, y_r, w_r, h_r, vertices) are normalized to [0, 1] relative to image dimensions.

2. **Coordinate System**: 3D coordinates use camera coordinate system with Y pointing down (KITTI convention).

3. **Box Center**: The Y coordinate represents the bottom center of the 3D box, not the geometric center. The geometric center is computed during parsing.

4. **Right Image Box**: 
   - **26-value format**: Stores full right box (x_r, y_r, w_r, h_r)
   - **22-value format**: Only stores x_r and w_r (y and h are same as left due to epipolar constraint)

5. **Dimensions Order**: 
   - **26-value format**: `dim_l dim_w dim_h` (length, width, height)
   - **22-value format**: `dim_h dim_w dim_l` (height, width, length)
   - Both are converted to (length, width, height) in Box3D objects

6. **Orientation Representation**:
   - **26-value format**: Direct `rotation_y` (global yaw) in radians
   - **22-value format**: `alpha` (observation angle) converted to `rotation_y` using: `rotation_y = alpha + arctan(x/z)`

7. **Truncation and Occlusion**:
   - **26-value format**: Includes `truncated` (float [0.0, 1.0]) and `occluded` (int [0-3]) at the end
   - These values are extracted from original KITTI labels and preserved for training/evaluation
   - Truncated indicates if object leaves image boundaries, occluded indicates visibility level

8. **Format Compatibility**: The parser in `kitti_stereo.py` expects the 26-value format. If you have older 24-value labels, you will need to reconvert them using the updated conversion script.

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
- Use 3DOP strategy: indices 0-3711 → train, 3712+ → val
- Output converted dataset to the same directory as `--kitti-root`
- Include all classes by default, or only specified classes if `--filter-classes` is used
- Available classes: Car, Van, Truck, Pedestrian, Person_sitting, Cyclist, Tram, Misc

This will create the proper directory structure and convert all annotations to the YOLO 3D format (26 values per object, including truncated and occluded attributes). When using `--filter-classes`, class IDs will be remapped to be consecutive starting from 0.

