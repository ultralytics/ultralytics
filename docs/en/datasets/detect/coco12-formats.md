---
comments: true
description: Explore the Ultralytics COCO12-Formats dataset, a test dataset featuring all 12 supported image formats (AVIF, BMP, DNG, HEIC, JP2, JPEG, JPG, MPO, PNG, TIF, TIFF, WebP) for validating image loading pipelines.
keywords: COCO12-Formats, Ultralytics, dataset, image formats, object detection, YOLO, AVIF, BMP, DNG, HEIC, JP2, JPEG, PNG, TIFF, WebP, MPO
---

# COCO12-Formats Dataset

## Introduction

The [Ultralytics](https://www.ultralytics.com/) COCO12-Formats dataset is a specialized test dataset designed to validate image loading across all 12 supported image format extensions. It contains 12 images (6 for training, 6 for validation), each saved in a different format to ensure comprehensive testing of the image loading pipeline.

This dataset is invaluable for:

- **Testing image format support**: Verify that all supported formats load correctly
- **CI/CD pipelines**: Automated testing of format compatibility
- **Debugging**: Isolate format-specific issues in training pipelines
- **Development**: Validate new format additions or changes

## Supported Formats

The dataset includes one image for each of the 12 supported format extensions defined in `ultralytics/data/utils.py`:

| Format | Extension | Description                          | Train/Val |
| ------ | --------- | ------------------------------------ | --------- |
| AVIF   | `.avif`   | AV1 Image File Format (modern)       | Train     |
| BMP    | `.bmp`    | Bitmap - uncompressed raster format  | Train     |
| DNG    | `.dng`    | Digital Negative - Adobe RAW format  | Train     |
| HEIC   | `.heic`   | High Efficiency Image Coding         | Train     |
| JPEG   | `.jpeg`   | JPEG with full extension             | Train     |
| JPG    | `.jpg`    | JPEG with short extension            | Train     |
| JP2    | `.jp2`    | JPEG 2000 - medical/geospatial       | Val       |
| MPO    | `.mpo`    | Multi-Picture Object (stereo images) | Val       |
| PNG    | `.png`    | Portable Network Graphics            | Val       |
| TIF    | `.tif`    | TIFF with short extension            | Val       |
| TIFF   | `.tiff`   | Tagged Image File Format             | Val       |
| WebP   | `.webp`   | Modern web image format              | Val       |

## Dataset Structure

```
coco12-formats/
├── images/
│   ├── train/          # 6 images (avif, bmp, dng, heic, jpeg, jpg)
│   └── val/            # 6 images (jp2, mpo, png, tif, tiff, webp)
├── labels/
│   ├── train/          # Corresponding YOLO format labels
│   └── val/
└── coco12-formats.yaml # Dataset configuration
```

## Dataset YAML

The COCO12-Formats dataset is configured using a YAML file that defines dataset paths and class names. You can review the official `coco12-formats.yaml` file in the [Ultralytics GitHub repository](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco12-formats.yaml).

!!! example "ultralytics/cfg/datasets/coco12-formats.yaml"

    ```yaml
    --8<-- "ultralytics/cfg/datasets/coco12-formats.yaml"
    ```

## Dataset Generation

The dataset can be generated using the provided script that converts source images from COCO8 and COCO128 to all supported formats:

```python
from ultralytics.data.scripts.generate_coco12_formats import generate_coco12_formats

# Generate the dataset
generate_coco12_formats()
```

### Requirements

Some formats require additional dependencies:

```bash
pip install pillow pillow-heif pillow-avif-plugin
```

#### AVIF System Library (Optional)

For OpenCV to read AVIF files directly, `libavif` must be installed **before** building OpenCV:

=== "macOS"

    ```bash
    brew install libavif
    ```

=== "Ubuntu/Debian"

    ```bash
    sudo apt install libavif-dev libavif-bin
    ```

=== "From Source"

    ```bash
    git clone -b v1.2.1 https://github.com/AOMediaCodec/libavif.git
    cd libavif
    cmake -B build -DAVIF_CODEC_AOM=SYSTEM -DAVIF_BUILD_APPS=ON
    cmake --build build --config Release --parallel
    sudo cmake --install build
    ```

!!! note

    The pip-installed `opencv-python` package may not include AVIF support since it's pre-built. Ultralytics uses Pillow with `pillow-avif-plugin` as a fallback for AVIF images when OpenCV lacks support.

## Usage

To train a YOLO model on the COCO12-Formats dataset, use the following examples:

!!! example "Train Example"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a pretrained YOLO model
        model = YOLO("yolo26n.pt")

        # Train on COCO12-Formats to test all image formats
        results = model.train(data="coco12-formats.yaml", epochs=1, imgsz=640)
        ```

    === "CLI"

        ```bash
        # Train YOLO on COCO12-Formats
        yolo detect train data=coco12-formats.yaml model=yolo26n.pt epochs=1 imgsz=640
        ```

## Format-Specific Notes

### AVIF (AV1 Image File Format)

AVIF is a modern image format based on the AV1 video codec, offering excellent compression. Requires `pillow-avif-plugin`:

```bash
pip install pillow-avif-plugin
```

### DNG (Digital Negative)

DNG is Adobe's open RAW format based on TIFF. For testing purposes, the dataset uses TIFF-based files with the `.dng` extension.

### JP2 (JPEG 2000)

JPEG 2000 is a wavelet-based image compression standard offering better compression and quality than traditional JPEG. Commonly used in medical imaging (DICOM), geospatial applications, and digital cinema. Natively supported by both OpenCV and Pillow.

### MPO (Multi-Picture Object)

MPO files are used for stereoscopic (3D) images. The dataset stores standard JPEG data with the `.mpo` extension for format testing.

### HEIC (High Efficiency Image Coding)

HEIC requires the `pillow-heif` package for proper encoding:

```bash
pip install pillow-heif
```

## Use Cases

### CI/CD Testing

```python
from ultralytics import YOLO


def test_all_image_formats():
    """Test that all image formats load correctly."""
    model = YOLO("yolo26n.pt")
    results = model.train(data="coco12-formats.yaml", epochs=1, imgsz=64)
    assert results is not None
```

### Format Validation

```python
from pathlib import Path

from ultralytics.data.utils import IMG_FORMATS

# Verify all formats are represented
dataset_dir = Path("datasets/coco12-formats/images")
found_formats = {f.suffix[1:].lower() for f in dataset_dir.rglob("*.*")}
assert found_formats == IMG_FORMATS, f"Missing formats: {IMG_FORMATS - found_formats}"
```

## Citations and Acknowledgments

If you use the COCO dataset in your research, please cite:

!!! quote ""

    === "BibTeX"

        ```bibtex
        @misc{lin2015microsoft,
              title={Microsoft COCO: Common Objects in Context},
              author={Tsung-Yi Lin and Michael Maire and Serge Belongie and Lubomir Bourdev and Ross Girshick and James Hays and Pietro Perona and Deva Ramanan and C. Lawrence Zitnick and Piotr Doll{\'a}r},
              year={2015},
              eprint={1405.0312},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

## FAQ

### What Is the COCO12-Formats Dataset Used For?

The COCO12-Formats dataset is designed for testing image format compatibility in Ultralytics YOLO training pipelines. It ensures all 12 supported image formats (AVIF, BMP, DNG, HEIC, JP2, JPEG, JPG, MPO, PNG, TIF, TIFF, WebP) load and process correctly.

### Why Test Multiple Image Formats?

Different image formats have unique characteristics (compression, bit depth, color spaces). Testing all formats ensures:

- Robust image loading code
- Compatibility across diverse datasets
- Early detection of format-specific bugs

### Which Formats Require Special Dependencies?

- **AVIF**: Requires `pillow-avif-plugin`
- **HEIC**: Requires `pillow-heif`

### Can I Add New Format Tests?

Yes! Modify the `generate_coco12_formats.py` script to include additional formats. Ensure you also update `IMG_FORMATS` in `ultralytics/data/utils.py`.
