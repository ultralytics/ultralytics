---
plans: [free, pro, enterprise]
comments: true
description: Import every dataset from your Roboflow workspace into Ultralytics Platform with a single API key.
keywords: Ultralytics Platform, Roboflow, Roboflow import, dataset import, integrations, YOLO, computer vision
title: Roboflow Dataset Import - Ultralytics Platform
---

# Roboflow Integration

The Roboflow integration imports every supported dataset in your Roboflow workspace into [Ultralytics Platform](https://platform.ultralytics.com) at its latest version. Re-run it any time to pull in datasets you've added since your last import.

## Import from Roboflow

1. Go to **Settings > [Integrations](index.md)** and select **Roboflow** from the integration list.
2. Paste your **Roboflow API Key** and click **Import**.
3. Review the **Import from Roboflow** preview dialog, which lists:
    - **New datasets** that will be imported
    - **Already imported** datasets that will be skipped
    - Any datasets with a **missing version**, **unsupported tasks**, or that **couldn't be sized**
    - Storage required, checked against your remaining storage
4. Click **Import** to start.

![Ultralytics Platform Settings Integrations Roboflow Import Dialog](https://cdn.ul.run/i/7eaadb8f6bb5bbcd57a6d46f89c524f1.avif)<!-- screenshot -->
Imported datasets appear in your [Datasets](../data/datasets.md) list immediately with a `processing` status and become ready once their images and annotations finish importing.

Any workspace editor can run the import. The preview is the slow step: Roboflow generates each new dataset's export archive before it can report a size, so a workspace with many fresh versions can take a minute or more to preview. If a very large workspace times out, run it again — everything already imported is skipped, so the second pass only has the remainder to do.

## Supported Task Types

Roboflow projects are mapped to the matching [YOLO task type](../data/index.md#supported-tasks):

| Roboflow Project Type       | Platform Task                                |
| --------------------------- | -------------------------------------------- |
| Object Detection            | [Detect](../../datasets/detect/index.md)     |
| Instance Segmentation       | [Segment](../../datasets/segment/index.md)   |
| Single-Label Classification | [Classify](../../datasets/classify/index.md) |
| Keypoint Detection          | [Pose](../../datasets/pose/index.md)         |

!!! note "Where to find your Roboflow API key"

    Follow Roboflow's guide to [find your API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key). Platform uses the key to run the preview and the import and then discards it — nothing is saved, and there is no connection to disconnect later.

!!! note "Unsupported projects are skipped"

    Projects with unsupported task types (such as multi-label classification and semantic segmentation) and projects that don't yet have a generated version are skipped and reported in the preview. A dataset whose export size Roboflow can't report in time is also left out and shown as **Couldn't size — retry later**.

!!! tip "Re-run to sync new datasets"

    Already-imported dataset versions are detected and skipped automatically, so you can safely re-run the import to pull in newly added datasets without creating duplicates. Bumping a project's version in Roboflow makes it a new dataset to import rather than an update to the one you already have.

!!! warning "Trashed imports still count as imported"

    A previously imported dataset that you moved to [Trash](../account/trash.md) is still detected and skipped, because it keeps occupying storage during the retention window. Delete it permanently before re-importing the same Roboflow version.
