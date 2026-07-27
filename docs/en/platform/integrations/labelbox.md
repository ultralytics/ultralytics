---
plans: [free, pro, enterprise]
comments: true
description: Upload a Labelbox NDJSON export straight to Ultralytics Platform and start editing annotations and training YOLO models.
keywords: Ultralytics Platform, Labelbox, Labelbox export, NDJSON, dataset import, annotation, integrations, YOLO, computer vision
title: Labelbox Dataset Import - Ultralytics Platform
---

# Labelbox Integration

[Ultralytics Platform](https://platform.ultralytics.com) reads Labelbox NDJSON exports directly, so there is no connection to set up and no API key to paste. Export from Labelbox, upload the file, and the dataset is ready to edit and train.

## Import from Labelbox

1. **Export from Labelbox.** Export the project or catalog you want to bring over. Labelbox produces an NDJSON file.
2. **Upload to Platform.** Create a new dataset and upload the `.ndjson` file, or paste a direct link to it. You can also start from **Settings > [Integrations](index.md) > Labelbox**.
3. **That's it.** Once processing finishes, [edit the annotations](../data/annotation.md) and [train](../train/index.md) the dataset like any other Platform dataset.

## What Gets Imported

Platform detects the Labelbox format automatically and maps each annotation to the matching [YOLO task type](../data/index.md#supported-tasks):

| Labelbox Annotation | Platform Task                              |
| ------------------- | ------------------------------------------ |
| Bounding box        | [Detect](../../datasets/detect/index.md)   |
| Polygon             | [Segment](../../datasets/segment/index.md) |

Class names come from your Labelbox annotation names, and pixel coordinates are normalized using the image dimensions in the export. A catalog export with no annotations imports as an unlabeled image dataset that you can annotate in Platform.

!!! note "Image URLs must be reachable"

    A Labelbox export references each image by URL rather than embedding the pixels. Platform downloads the images from those URLs, probing a sample first, so the export fails fast if the images are not publicly accessible.

!!! tip "Mixed annotations import as segment"

    An export that contains both bounding boxes and polygons is imported as a segment dataset.
