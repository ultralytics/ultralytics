---
plans: [free, pro, enterprise]
coming_soon: true
comments: true
description: Bring Label Studio annotation projects into Ultralytics Platform to edit annotations, train YOLO models, and deploy them.
keywords: Ultralytics Platform, Label Studio, Label Studio export, dataset import, annotation, integrations, YOLO, computer vision
title: Label Studio Dataset Import - Ultralytics Platform
---

# Label Studio Integration

Direct [Label Studio](https://labelstud.io/) imports are coming soon to [Ultralytics Platform](https://platform.ultralytics.com). Once available, you will export your Label Studio project and upload it to Platform to [annotate](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without converting the data yourself.

## Import from Label Studio Today

Label Studio exports datasets in several formats, including YOLO, which Platform already supports:

1. In Label Studio, export your project in a **YOLO** format.
2. [Upload the export](../data/datasets.md) to Platform as a new dataset.
3. Edit the annotations and train the dataset like any other Platform dataset.

!!! tip "Already have labels elsewhere?"

    The [Labelbox](labelbox.md) and [Roboflow](roboflow.md) integrations are available today, and Platform also imports YOLO, COCO, and Ultralytics NDJSON datasets directly.
