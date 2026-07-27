---
plans: [free, pro, enterprise]
coming_soon: true
comments: true
description: Bring Label Studio annotation projects into Ultralytics Platform to edit annotations, train YOLO models, and deploy them from one workspace.
keywords: Ultralytics Platform, Label Studio, Label Studio export, dataset import, annotation, integrations, YOLO, computer vision
title: Label Studio Dataset Import - Ultralytics Platform
---

# Label Studio Integration

Direct [Label Studio](https://labelstud.io/) imports are coming to [Ultralytics Platform](https://platform.ultralytics.com). Today, moving a Label Studio project into training means choosing an export format and matching it to what your training code expects. The integration removes that step: upload the export and Platform maps the annotations to the matching [YOLO task](../data/index.md#supported-tasks) for you.

## How It Will Work

1. **Export from Label Studio.** Export the project you want to train on.
2. **Upload to Platform.** Create a new dataset from the export — no conversion step and no format to choose.
3. **Train.** [Edit the annotations](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) from the same workspace.

## Why It Helps

- **No conversion scripts** — annotations and class names map to a YOLO dataset automatically, so there is nothing to keep in sync as your label schema changes
- **One workspace** — labeling, training, and deployment live together instead of spanning a labeling tool, a conversion step, and a training environment
- **Keep annotating** — imported datasets open in Platform's [annotation editor](../data/annotation.md), including SAM-powered smart annotation
- **Train immediately** — datasets are ready for [cloud training](../train/cloud-training.md) as soon as they finish processing

## Import from Label Studio Today

Label Studio already exports in formats Platform ingests, so you do not have to wait:

1. In Label Studio, export your project in a **YOLO** or **COCO** format.
2. [Upload the export](../data/datasets.md) to Platform as a new dataset.
3. Edit the annotations and train the dataset like any other Platform dataset.

!!! warning "Pascal VOC XML is not supported"

    Choose a YOLO or COCO export rather than Pascal VOC. Platform flags XML label files during upload because it cannot read them.

!!! tip "Available now"

    The [Labelbox](labelbox.md), [Roboflow](roboflow.md), and [Ultralytics HUB](ultralytics-hub.md) integrations work today, and Platform imports YOLO, COCO, and Ultralytics NDJSON datasets directly.
