---
comments: true
description: Learn how to upload, manage, and organize datasets in Ultralytics Platform for YOLO model training with automatic processing and statistics.
keywords: Ultralytics Platform, datasets, dataset management, YOLO, data upload, training data, computer vision, machine learning
---

# Datasets

[Ultralytics Platform](https://platform.ultralytics.com) datasets provide a streamlined solution for managing your training data. Once uploaded, datasets can be immediately used for model training, with automatic processing and statistics generation.

## Upload Dataset

Ultralytics Platform accepts multiple upload formats for flexibility.

### Supported Image Formats

| Format | Extensions      | Notes                    |
| ------ | --------------- | ------------------------ |
| JPEG   | `.jpg`, `.jpeg` | Most common, recommended |
| PNG    | `.png`          | Supports transparency    |
| WebP   | `.webp`         | Modern, good compression |
| BMP    | `.bmp`          | Uncompressed             |
| GIF    | `.gif`          | First frame extracted    |
| TIFF   | `.tiff`, `.tif` | High quality             |
| HEIC   | `.heic`         | iPhone photos            |
| AVIF   | `.avif`         | Next-gen format          |
| JP2    | `.jp2`          | JPEG 2000                |
| DNG    | `.dng`          | Raw camera               |

### Supported Video Formats

Videos are automatically extracted to frames:

| Format | Extensions | Extraction            |
| ------ | ---------- | --------------------- |
| MP4    | `.mp4`     | 1 FPS, max 100 frames |
| WebM   | `.webm`    | 1 FPS, max 100 frames |
| MOV    | `.mov`     | 1 FPS, max 100 frames |
| AVI    | `.avi`     | 1 FPS, max 100 frames |
| MKV    | `.mkv`     | 1 FPS, max 100 frames |
| M4V    | `.m4v`     | 1 FPS, max 100 frames |

### File Size Limits

| Type      | Maximum Size |
| --------- | ------------ |
| Images    | 50 MB each   |
| Videos    | 1 GB each    |
| ZIP files | 50 GB        |

### Archives

ZIP files up to 50GB are supported with folder structure preserved and automatic extraction and processing.

### Preparing Your Dataset

For labeled datasets, use the standard YOLO format:

```
my-dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   └── img002.jpg
│   └── val/
│       ├── img003.jpg
│       └── img004.jpg
├── labels/
│   ├── train/
│   │   ├── img001.txt
│   │   └── img002.txt
│   └── val/
│       ├── img003.txt
│       └── img004.txt
└── data.yaml
```

The YAML file defines your dataset configuration:

```yaml
# data.yaml
path: .
train: images/train
val: images/val

names:
    0: person
    1: car
    2: dog
```

### Upload Process

1. Navigate to **Datasets** in the sidebar
2. Click **Upload Dataset** or drag files into the upload zone
3. Select the task type (detect, segment, pose, OBB, classify)
4. Add a name and optional description
5. Click **Upload**

<!-- Screenshot: platform-datasets-upload.avif -->

After upload, the Platform processes your data:

1. **Normalization**: Large images resized (max 4096px)
2. **Thumbnails**: 256px previews generated
3. **Label Parsing**: YOLO format labels extracted
4. **Statistics**: Class distributions computed

<!-- Screenshot: platform-datasets-upload-progress.avif -->

??? tip "Validate Before Upload"

    You can validate your dataset locally before uploading:

    ```python
    from ultralytics.hub import check_dataset

    check_dataset("path/to/dataset.zip", task="detect")
    ```

## Browse Images

View your dataset images in multiple layouts:

| View        | Description                                      |
| ----------- | ------------------------------------------------ |
| **Grid**    | Thumbnail grid with annotation overlays          |
| **Compact** | Smaller thumbnails for quick scanning            |
| **Table**   | List with filename, dimensions, and label counts |

<!-- Screenshot: platform-datasets-gallery.avif -->

### Fullscreen Viewer

Click any image to open the fullscreen viewer with:

- **Navigation**: Arrow keys or click to browse
- **Metadata**: Filename, dimensions, split, label count
- **Annotations**: Toggle annotation visibility
- **Class Breakdown**: Per-class label counts

<!-- Screenshot: platform-datasets-fullscreen.avif -->

### Filter by Split

Filter images by their dataset split:

| Split       | Purpose                             |
| ----------- | ----------------------------------- |
| **Train**   | Used for model training             |
| **Val**     | Used for validation during training |
| **Test**    | Used for final evaluation           |
| **Unknown** | No split assigned                   |

## Dataset Statistics

The **Statistics** tab provides automatic analysis of your dataset:

### Class Distribution

Bar chart showing the number of annotations per class:

<!-- Screenshot: platform-datasets-stats-class.avif -->

### Location Heatmap

Visualization of where annotations appear in images:

<!-- Screenshot: platform-datasets-stats-heatmap.avif -->

### Dimension Analysis

Scatter plot of image dimensions (width vs height):

<!-- Screenshot: platform-datasets-stats-dimensions.avif -->

!!! tip "Statistics Caching"

    Statistics are cached for 5 minutes. Changes to annotations will be reflected after the cache expires.

## Export Dataset

Export your dataset in NDJSON format for offline use:

1. Open the dataset actions menu
2. Click **Export**
3. Download the NDJSON file

<!-- Screenshot: platform-datasets-export.avif -->

The NDJSON format stores one JSON object per line:

```json
{"filename": "img001.jpg", "split": "train", "labels": [...]}
{"filename": "img002.jpg", "split": "train", "labels": [...]}
```

See the [Ultralytics NDJSON format documentation](https://docs.ultralytics.com/datasets/detect/#ultralytics-ndjson-format) for full specification.

## Dataset URI

Reference Platform datasets using the `ul://` URI format:

```
ul://username/datasets/dataset-slug
```

Use this URI to train models from anywhere:

```bash
export ULTRALYTICS_API_KEY="your_api_key"
yolo train model=yolo26n.pt data=ul://username/datasets/my-dataset epochs=100
```

!!! example "Train Anywhere with Platform Data"

    The `ul://` URI works from any environment:

    - **Local machine**: Train on your hardware, data downloaded automatically
    - **Google Colab**: Access your Platform datasets in notebooks
    - **Remote servers**: Train on cloud VMs with full dataset access

## Visibility Settings

Control who can see your dataset:

| Setting     | Description                     |
| ----------- | ------------------------------- |
| **Private** | Only you can access             |
| **Public**  | Anyone can view on Explore page |

<!-- Screenshot: platform-datasets-visibility.avif -->

To change visibility:

1. Open dataset actions menu
2. Click **Edit**
3. Toggle visibility setting
4. Click **Save**

## Edit Dataset

Update dataset name, description, or visibility:

1. Open dataset actions menu
2. Click **Edit**
3. Make changes
4. Click **Save**

## Delete Dataset

Delete a dataset you no longer need:

1. Open dataset actions menu
2. Click **Delete**
3. Confirm deletion

!!! note "Trash and Restore"

    Deleted datasets are moved to Trash for 30 days. You can restore them from the Trash page in Settings.

## Train on Dataset

Start training directly from your dataset:

1. Click **Train Model** on the dataset page
2. Select a project or create new
3. Configure training parameters
4. Start training

See [Cloud Training](../train/cloud-training.md) for details.

## FAQ

### What happens to my data after upload?

Your data is processed and stored in your selected region (US, EU, or AP). Images are:

1. Validated for format and size
2. Normalized if larger than 4096px (preserving aspect ratio)
3. Stored using Content-Addressable Storage (CAS) with SHA-256 hashing
4. Thumbnails generated at 256px for fast browsing

### How does storage work?

Ultralytics Platform uses **Content-Addressable Storage (CAS)** for efficient storage:

- **Deduplication**: Identical images uploaded by different users are stored only once
- **Integrity**: SHA-256 hashing ensures data integrity
- **Efficiency**: Reduces storage costs and speeds up processing
- **Regional**: Data stays in your selected region (US, EU, or AP)

### Can I add images to an existing dataset?

Yes, use the **Add Images** button on the dataset page to upload additional images. New statistics will be computed automatically.

### How do I move images between datasets?

Use the bulk selection feature:

1. Select images in the gallery
2. Click **Move** or **Copy**
3. Select destination dataset

### What label formats are supported?

Ultralytics Platform supports YOLO format labels:

| Task     | Format                           | Example                             |
| -------- | -------------------------------- | ----------------------------------- |
| Detect   | `class cx cy w h`                | `0 0.5 0.5 0.2 0.3`                 |
| Segment  | `class x1 y1 x2 y2 ...`          | `0 0.1 0.1 0.9 0.1 0.9 0.9`         |
| Pose     | `class cx cy w h kx1 ky1 v1 ...` | `0 0.5 0.5 0.2 0.3 0.6 0.7 2`       |
| OBB      | `class x1 y1 x2 y2 x3 y3 x4 y4`  | `0 0.1 0.1 0.9 0.1 0.9 0.9 0.1 0.9` |
| Classify | Directory structure              | `train/cats/`, `train/dogs/`        |

All coordinates are normalized (0-1 range). Pose visibility flags: 0=not labeled, 1=labeled but occluded, 2=labeled and visible.
