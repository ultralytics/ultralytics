---
plans: [free, pro, enterprise]
coming_soon: true
comments: true
description: Move a Label Studio project into Ultralytics Platform with the YOLO export, then edit annotations, train YOLO models, and deploy from one workspace.
keywords: Ultralytics Platform, Label Studio, Label Studio export, YOLO export, dataset import, annotation, integrations, YOLO, computer vision
title: Label Studio Dataset Import - Ultralytics Platform
---

# Label Studio Integration

Direct [Label Studio](https://labelstud.io/) imports are coming to [Ultralytics Platform](https://platform.ultralytics.com), so that a raw project export uploads as-is with no format to choose.

Until then there is a short path that works today, because Label Studio exports YOLO directly and Platform reads the class list that export ships.

## Import from Label Studio Today

1. In Label Studio, open your project and click **Export**.
2. Choose the **[YOLO](https://labelstud.io/guide/export)** format. COCO works too.
3. [Upload the exported ZIP](../data/datasets.md) to Platform as a new dataset.
4. [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without leaving the workspace.

A Label Studio YOLO export carries its class list alongside the labels, and Platform reads it:

```
archive.zip/
├── classes.txt        # class names Platform reads first
├── notes.json         # fallback, used only when classes.txt is missing or empty
├── images/
└── labels/
```

Upload the archive exactly as Label Studio produced it. Platform looks for `classes.txt` and `notes.json` at the root of the archive, so re-zipping the export inside another folder loses your label names and the classes import as `class0`, `class1`, and so on.

## Choosing an Export Format

Label Studio offers [several export formats](https://labelstud.io/guide/export). For image detection and segmentation:

| Label Studio Format | Works | Notes                                                   |
| ------------------- | ----- | ------------------------------------------------------- |
| **YOLO**            | Best  | Ships `classes.txt`, so your label names carry over     |
| **COCO**            | Yes   | Platform reads COCO JSON annotations and category names |
| **Pascal VOC XML**  | No    | XML label files cannot be read                          |

!!! warning "Pascal VOC imports without annotations"

    Platform does not read Pascal VOC XML labels, and a VOC export fails quietly rather than loudly: the images import and the annotations do not. Choose YOLO or COCO instead.

## What the Integration Will Add

Picking the right export format is the step the integration removes. Once it ships, you will export from Label Studio however you like, upload it, and Platform will map the annotations to the matching [YOLO task](../data/index.md#supported-tasks) itself.

- **No format to choose** — every Label Studio export maps to a YOLO dataset with label names preserved
- **One workspace** — labeling, training, and deployment stop spanning separate tools and a conversion script
- **Keep annotating** — imported datasets open in Platform's [annotation editor](../data/annotation.md), including SAM-powered smart annotation
- **Train immediately** — datasets are ready for [cloud training](../train/cloud-training.md) as soon as they finish processing

!!! tip "Available now"

    The [Labelbox](labelbox.md), [Roboflow](roboflow.md), and [Ultralytics HUB](ultralytics-hub.md) integrations work today, and Platform imports YOLO, COCO, and Ultralytics NDJSON datasets directly.
