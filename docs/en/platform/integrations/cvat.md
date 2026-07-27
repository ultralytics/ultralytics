---
plans: [free, pro, enterprise]
coming_soon: true
comments: true
description: Move a CVAT task or project into Ultralytics Platform with the Ultralytics YOLO export, then edit annotations, train YOLO models, and deploy from one workspace.
keywords: Ultralytics Platform, CVAT, CVAT export, Ultralytics YOLO format, dataset import, annotation, integrations, YOLO, computer vision
title: CVAT Dataset Import - Ultralytics Platform
---

# CVAT Integration

Direct [CVAT](https://www.cvat.ai/) imports are coming to [Ultralytics Platform](https://platform.ultralytics.com), so that any CVAT export uploads as-is with no format to choose.

Until then there is a short path that works today, because CVAT already exports in the Ultralytics YOLO layout Platform reads.

## Import from CVAT Today

1. In CVAT, open your task or project and choose **Export dataset**.
2. Select the **[Ultralytics YOLO](https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/)** format that matches your task, and include the images.
3. [Upload the exported ZIP](../data/datasets.md) to Platform as a new dataset.
4. [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without leaving the workspace.

CVAT's Ultralytics YOLO export produces the layout Platform expects, so nothing needs converting:

```
archive.zip/
├── data.yaml          # class names Platform reads
├── images/train/
└── labels/train/
```

## Choosing an Export Format

CVAT offers [many export formats](https://docs.cvat.ai/docs/dataset_management/formats/). Three matter here:

| CVAT Format          | Works  | Notes                                                                                        |
| -------------------- | ------ | -------------------------------------------------------------------------------------------- |
| **Ultralytics YOLO** | Best   | Ships `data.yaml`, so your label names come across intact                                    |
| **COCO 1.0**         | Yes    | Platform reads COCO JSON annotations and category names                                      |
| **YOLO 1.1**         | Partly | Boxes import, but its `obj.names` file is not read — classes arrive as `class0`, `class1`, … |

!!! warning "Pascal VOC XML is not supported"

    Platform cannot read XML label files and flags them during upload. Choose Ultralytics YOLO or COCO instead.

## What the Integration Will Add

Picking the right export format is the step the integration removes. Once it ships, you will export from CVAT however you like, upload it, and Platform will map the annotations to the matching [YOLO task](../data/index.md#supported-tasks) itself.

- **No format to choose** — every CVAT export maps to a YOLO dataset with label names preserved
- **One workspace** — labeling, training, and deployment stop spanning separate tools and a conversion script
- **Keep annotating** — imported datasets open in Platform's [annotation editor](../data/annotation.md), including SAM-powered smart annotation
- **Train immediately** — datasets are ready for [cloud training](../train/cloud-training.md) as soon as they finish processing

!!! tip "Available now"

    The [Labelbox](labelbox.md), [Roboflow](roboflow.md), and [Ultralytics HUB](ultralytics-hub.md) integrations work today, and Platform imports YOLO, COCO, and Ultralytics NDJSON datasets directly.
