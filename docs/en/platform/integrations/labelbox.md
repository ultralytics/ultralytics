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

1. **Export from Labelbox.** Open the [labeling project](https://docs.labelbox.com/docs/export-labels) or [catalog](https://docs.labelbox.com/docs/export-from-catalog) you want to bring over, go to the **Data Rows** tab, select **All**, and click **Export data**. Labelbox runs the export as a background job — watch for it in [Notifications](https://app.labelbox.com/notifications) and download the NDJSON file once the job finishes.
2. **Upload to Platform.** Create a new dataset and upload the `.ndjson` file, or paste a direct link to it. You can also start from **Settings > [Integrations](index.md) > Labelbox**.
3. **Train.** Once processing finishes, [edit the annotations](../data/annotation.md) and [train](../train/index.md) exactly like any other Platform dataset.

Unlike [CVAT](cvat.md) and [Label Studio](label-studio.md), there is no format to choose — Labelbox's own export is the supported one.

## What Gets Imported

Platform recognizes the Labelbox format on its own and maps bounding boxes and polygons to the matching [YOLO task type](../data/index.md#supported-tasks):

| Labelbox Annotation                              | Imported                                   |
| ------------------------------------------------ | ------------------------------------------ |
| Bounding box                                     | [Detect](../../datasets/detect/index.md)   |
| Polygon                                          | [Segment](../../datasets/segment/index.md) |
| Segmentation mask, point, polyline, relationship | Not yet                                    |

Your Labelbox annotation names become the dataset's class names, and pixel coordinates are normalized using the image dimensions recorded in the export. A catalog export with no annotations imports as an unlabeled image dataset, ready to label in Platform's [annotation editor](../data/annotation.md).

!!! warning "Mask and point projects import without annotations"

    Only bounding boxes and polygons are read today. A project labeled entirely with the segmentation-mask (brush) tool, points, or polylines imports its images and class names but no annotations, and no error is raised. You can spot it on the dataset page: an imported dataset that kept its labels shows a labeled and annotation count next to the image count, and one that lost them shows neither.

!!! note "Upload the export while its links are fresh"

    A Labelbox export references each image by signed URL rather than embedding the pixels, and those signatures expire. Platform downloads the images from those URLs, probing a sample before it starts, so an export whose links have all lapsed fails immediately and tells you why — re-export from Labelbox and upload the new file. An export that is only partly expired imports the images it can still reach and skips the rest, so upload soon after exporting.

!!! tip "Mixed annotations import as segment"

    An export containing both bounding boxes and polygons is imported as a segment dataset. Polygons need at least three points to be read.
