---
comments: true
description: Discover how to extend the utility of the Ultralytics package to support your development process.
keywords: Ultralytics, YOLO, custom, function, workflow, utility, support, 
---

# NOTE: add image and/or video

<p align="center">
  <img src="" alt="THIS IS A PLACEHOLDER ONLY">
</p>

The `ultralytics` package comes with a myriad of utilities that can support, enhance, and speed up your workflows. There are many more available, but here are some that will be useful for most developers. They're also a great reference point to use when learning to program.

## Data

### YOLO Data Explorer

[YOLO Explorer](../datasets/explorer/index.md) was added in the `8.1.0` anniversary update and is a powerful tool you can use to better understand your dataset.

### Auto Labeling / Annotations

Dataset annotation is an _extremely_ resource heavy and time consuming process. If you have a YOLO object detection model trained on a reasonable amount of data, you can use it and [SAM](../models/sam.md) to auto-annotate additional data (segmentation format).

```python
from ultralytics.data.annotator import auto_annotate

auto_annotate(
    data='path/to/new/data',
    det_model='yolov8n.pt',
    sam_model='mobile_sam.pt',
    device="cuda",
    output_dir="path/to/save_labels",
)
# NOTE nothing returned
```

- [See the reference section for `annotator.auto_annotate`](../reference/data/annotator.md#ultralytics.data.annotator.auto_annotate) for more insight on how the function operates.

- Use in combination with the [function `segments2boxes`](#convert-segments-to-bounding-boxes) to generate object detection bounding boxes as well

### Convert COCO into YOLO Format

Use to convert COCO JSON annotations into proper YOLO format. For object detection (bounding box) datasets, `use_segments` and `use_keypoints` should both be `False`

```python
from ultralytics.data.converter import convert_coco

convert_coco(
    '../datasets/coco/annotations/',
    use_segments=False, 
    use_keypoints=False,
    cls91to80=True,
)
```

For additional information about the `convert_coco` function, [visit the reference page](../reference/data/converter.md#ultralytics.data.converter.convert_coco)

### Convert Bounding Boxes to Segments

With existing `x y w h` bounding box data, convert to segments using the `yolo_bbox2segment` function. The files for images and annotations need to be organized like this:

```
data
|__ images
    ├─ 001.jpg
    ├─ 002.jpg
    ├─ ..
    └─ NNN.jpg
|__ labels
    ├─ 001.txt
    ├─ 002.txt
    ├─ ..
    └─ NNN.txt
```

```python
from ultralytics.data.converter import yolo_bbox2segment

yolo_bbox2segment(
    im_dir="path/to/images",
    save_dir=None, # saved to "labels-segment" in images directory
    sam_model="sam_b.pt"
)
# NOTE nothing returned
```

[Visit the `yolo_bbox2segment` reference page](../reference/data/converter.md#ultralytics.data.converter.yolo_bbox2segment) for more information regarding the function.

### Convert Segments to Bounding Boxes

If you have a dataset that uses the [segmentation dataset format](../datasets/segment/index.md) you can easily convert these into up-right (or horizontal) bounding boxes with this function.

```python
from ultralytics.utils.ops import segments2boxes

segments = [...] # segment labels from files as list, no class index

boxes = segments2boxes(segments) # xywh bounding boxes
```

To understand how this function works, visit the [reference page](../reference/utils/ops.md#ultralytics.utils.ops.segments2boxes)

## Utilities

### Image Compression

Compresses a single image file to reduced size while preserving its aspect ratio and quality. If the input image is smaller than the maximum dimension, it will not be resized.

```python
from pathlib import Path
from ultralytics.data.utils import compress_one_image

for f in Path('path/to/dataset').rglob('*.jpg'):
    compress_one_image(f)
```

### Auto-split Dataset

Automatically split a dataset into train/val/test splits and save the resulting splits into autosplit_*.txt files.

```python
from ultralytics.data.utils import autosplit

autosplit(
    path="path/to/images",
    weights=(0.9, 0.1, 0.0), # (train, validation, test) fractional splits
    annotated_only=False # split only images with annotation file when True
)
```

### Segment-polygons to Binary Mask

Convert a list of polygons to a binary mask of the specified image size. Polygons in the form of `[N, M]` with `N` as the number of polygons, and `M` as the number of coordinate pairs `m` that define the contour of the polygon `m * (x,y)`.

!!! warning

    `M` <b><u>must always</b></u> be even.

```python
from ultralytics.data.utils import polygon2mask

polygon2mask(
    imgsz,
    polygons,
    color=1,
    downsample_ratio=1
)
```

## Bounding Boxes

### Scaling Boxes

When scaling and image up or down, corresponding bounding box coordinates can be appropriately scaled to match using `ultralytics.utils.ops.scale_boxes`.

```python
import cv2 as cv
import numpy as np
from ultralytics.utils.ops import scale_boxes

image = cv.imread("ultralytics/assets/bus.jpg")
*(h, w), c = image.shape
resized = cv.resize(image, None, (), fx=1.2, fy=1.2)
*(new_h, new_w), _ = resized.shape

xyxy_boxes = np.array(
    [[  22.878,  231.27,  804.98,  756.83,],
    [   48.552,  398.56,  245.35,  902.71,],
    [   669.47,  392.19,  809.72,  877.04,],
    [   221.52,   405.8,  344.98,  857.54,],
    [        0,  550.53,   63.01,  873.44,],
    [ 0.058402,  254.46,  32.561,  324.87,]]
)

new_boxes = scale_boxes(
    img1_shape=(h, w),          # original image dimensions
    boxes=xyxy_boxes,           # boxes from original image
    img0_shape=(new_h, new_w),  # resized image dimensions (scale to)
    ratio_pad=None,
    padding=False,
    xywh=False,
)

new_boxes
>>> array(
    [[  27.454,  277.52,  965.98,   908.2],
    [   58.262,  478.27,  294.42,  1083.3],
    [   803.36,  470.63,  971.66,  1052.4],
    [   265.82,  486.96,  413.98,    1029],
    [        0,  660.64,  75.612,  1048.1],
    [ 0.070082,  305.35,  39.073,  389.84]]
)
```

### Bounding Box Format Conversion XYXY → XYWH

Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

```python
import numpy as np
from ultralytics.utils.ops import xyxy2xywh

xyxy_boxes = np.array(
    [[  22.878,  231.27,  804.98,  756.83,],
    [   48.552,  398.56,  245.35,  902.71,],
    [   669.47,  392.19,  809.72,  877.04,],
    [   221.52,   405.8,  344.98,  857.54,],
    [        0,  550.53,   63.01,  873.44,],
    [ 0.058402,  254.46,  32.561,  324.87,]]
)
xywh = xyxy2xywh(xyxy_boxes)

xywh
>>> array(
    [[ 413.93,  494.05,   782.1, 525.56],
    [  146.95,  650.63,   196.8, 504.15],
    [   739.6,  634.62,  140.25, 484.85],
    [  283.25,  631.67,  123.46, 451.74],
    [  31.505,  711.99,   63.01, 322.91],
    [   16.31,  289.67,  32.503,  70.41]]
)
```

### All Bounding Box Conversions

```python
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.ops import xywhn2xyxy # normalized → pixel
from ultralytics.utils.ops import xyxy2xywhn # pixel → normalized
from ultralytics.utils.ops import xywh2ltwh  # xywh → top-left corner, w, h
from ultralytics.utils.ops import xyxy2ltwh  # xyxy → top-left corner, w, h
from ultralytics.utils.ops import ltwh2xywh
from ultralytics.utils.ops import ltwh2xyxy
```

See docstring for each function or visit the `ultralytics.utils.ops` [reference page](../reference/utils/ops.md) to read more about each function.

## Miscellaneous 

### Code Profiling

Check duration for code to run/process either using `with` or as a decorator.

```python
from ultralytics.utils.ops import Profile

with Profile(device=device) as dt:
    pass  # slow operation here

print(dt)
>>> "Elapsed time is 9.5367431640625e-07 s"
```

### Ultralytics Supported Formats

Want or need to pull in the formats of images or videos types supported by Ultralytics? Use these constants if you need

```python
from ultralytics.data.utils import IMG_FORMATS
from ultralytics.data.utils import VID_FORMATS

print(IMG_FORMATS)
>>> ('bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm')
```

### Make Divisible

Calculates the nearest whole number to `x` to make evenly divisible when divided by `y`.

```python
from ultralytics.utils.ops import make_divisible

make_divisible(7, 3)
>>> 9
make_divisible(7, 2)
>>> 8
```
