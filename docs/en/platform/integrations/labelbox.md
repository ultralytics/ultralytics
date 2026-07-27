---
plans: [free, pro, enterprise]
comments: true
description: Upload a Labelbox NDJSON export straight to Ultralytics Platform and start editing annotations and training YOLO models.
keywords: Ultralytics Platform, Labelbox, Labelbox export, NDJSON, dataset import, annotation, integrations, YOLO, computer vision
title: Labelbox Dataset Import - Ultralytics Platform
---

# Labelbox Integration

[Ultralytics Platform](https://platform.ultralytics.com) reads Labelbox NDJSON exports directly. There is no key to paste, no connection to keep alive, and no conversion script to maintain — export from Labelbox, upload the file, and your labels arrive as a normal Platform dataset.

## Import from Labelbox

1. **Export from Labelbox.** Export the project or catalog you want to bring over. Labelbox produces an NDJSON file.
2. **Upload to Platform.** Create a new dataset and upload the `.ndjson` file, or paste a direct link to it. You can also start from **Settings > [Integrations](index.md) > Labelbox**.
3. **Train.** Once processing finishes, [edit the annotations](../data/annotation.md) and [train](../train/index.md) exactly like any other Platform dataset.

## What Gets Imported

Platform recognizes the Labelbox format on its own and maps each annotation to the matching [YOLO task type](../data/index.md#supported-tasks):

| Labelbox Annotation | Platform Task                              |
| ------------------- | ------------------------------------------ |
| Bounding box        | [Detect](../../datasets/detect/index.md)   |
| Polygon             | [Segment](../../datasets/segment/index.md) |

Your Labelbox annotation names become the dataset's class names, and pixel coordinates are normalized using the image dimensions recorded in the export. A catalog export with no annotations imports as an unlabeled image dataset, ready to label in Platform's [annotation editor](../data/annotation.md).

!!! note "Image URLs must be reachable"

    A Labelbox export references each image by URL rather than embedding the pixels. Platform downloads the images from those URLs, probing a sample first, so the import fails fast and tells you why if the images cannot be reached.

!!! tip "Mixed annotations import as segment"

    An export containing both bounding boxes and polygons is imported as a segment dataset.
