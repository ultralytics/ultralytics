---
plans: [free, pro, enterprise]
coming_soon: true
comments: true
description: Bring CVAT annotation projects into Ultralytics Platform to edit annotations, train YOLO models, and deploy them.
keywords: Ultralytics Platform, CVAT, CVAT export, dataset import, annotation, integrations, YOLO, computer vision
title: CVAT Dataset Import - Ultralytics Platform
---

# CVAT Integration

Direct [CVAT](https://www.cvat.ai/) imports are coming soon to [Ultralytics Platform](https://platform.ultralytics.com). Once available, you will export your CVAT project and upload it to Platform to [annotate](../data/annotation.md), [train](../train/index.md), and [deploy](../deploy/index.md) without converting the data yourself.

## Import from CVAT Today

CVAT exports datasets in several formats, including YOLO, which Platform already supports:

1. In CVAT, export your task or project in a **YOLO** format.
2. [Upload the export](../data/datasets.md) to Platform as a new dataset.
3. Edit the annotations and train the dataset like any other Platform dataset.

!!! tip "Already have labels elsewhere?"

    The [Labelbox](labelbox.md) and [Roboflow](roboflow.md) integrations are available today, and Platform also imports YOLO, COCO, and Ultralytics NDJSON datasets directly.
