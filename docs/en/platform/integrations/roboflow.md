---
plans: [free, pro, enterprise]
comments: true
description: Import datasets from your Roboflow workspace into Ultralytics Platform with a single API key.
keywords: Ultralytics Platform, Roboflow, Roboflow import, dataset import, integrations, YOLO, computer vision
title: Roboflow Dataset Import - Ultralytics Platform
---

# Roboflow Integration

The Roboflow integration imports the supported datasets you select from your Roboflow workspace into [Ultralytics Platform](https://platform.ultralytics.com) at their latest version. Re-run it any time to pull in datasets you've added since your last import.

## Import from Roboflow

1. Go to **Settings > [Integrations](index.md)** and select **Roboflow** from the integration list.
2. Paste your **Roboflow API Key** and click **Connect**. If the workspace has a [saved Roboflow key](../account/api-keys.md#provider-keys-for-agents-and-annotation), the preview opens with it automatically.
3. Review the **Import from Roboflow** preview:
    - A table of datasets (name, task, images, size), with nothing selected by default; select the new datasets you want to import
    - Already-imported versions shown as dimmed, unselectable rows with an **Already imported** tooltip
    - Counts of unsupported tasks and datasets that couldn't be sized
    - The size of your selection against your storage
4. Click **Import N** to import the N selected datasets; it stays disabled until the selection fits your storage.

The preview stays open after an import. Started and already-imported datasets remain visible but cannot be selected, while datasets that could not be started stay selected for a retry. Click **Connect** again to refresh the preview and its storage usage; storage limits are checked again when importing.

![Ultralytics Platform Settings Integrations Roboflow Import Dialog](https://cdn.ul.run/i/e0c09ece8166144e4135db09cbd89cf7.avif)<!-- screenshot -->

Imported datasets take the workspace's [default dataset visibility](../account/settings.md#default-visibility) and appear in your [Datasets](../data/datasets.md) list immediately with a `processing` status and become ready once their images and annotations finish importing.

Any workspace editor can run the import. The preview is the slow step: Roboflow generates each new dataset's export archive before it can report a size, so a workspace with many fresh versions can take a minute or more to preview. If a very large workspace times out, run it again — Roboflow keeps generating the exports it already started, so the next preview has less to wait for.

If some datasets can't be started, for example because one exceeds your plan's per-import size limit, the rest still import and the notification after you confirm names the first failure and its reason. Requests that are rate-limited, fail with a server error, or time out are retried automatically, so one hiccup doesn't fail the whole workspace.

## Supported Task Types

Roboflow projects are mapped to the matching [YOLO task type](../data/index.md#supported-tasks):

| Roboflow Project Type       | Platform Task                                |
| --------------------------- | -------------------------------------------- |
| Object Detection            | [Detect](../../datasets/detect/index.md)     |
| Instance Segmentation       | [Segment](../../datasets/segment/index.md)   |
| Single-Label Classification | [Classify](../../datasets/classify/index.md) |
| Keypoint Detection          | [Pose](../../datasets/pose/index.md)         |

!!! note "Where to find your Roboflow API key"

    Follow Roboflow's guide to [find your API key](https://docs.roboflow.com/reference/authentication/authentication/find-your-roboflow-api-key).

!!! note "Unsupported projects are skipped"

    Projects with unsupported task types (such as multi-label classification and semantic segmentation) and projects that don't yet have a generated version are skipped; the preview counts the unsupported ones. A dataset whose export size Roboflow can't report in time is also left out and counted under **couldn't be sized, retry later**.

!!! tip "Re-run to sync new datasets"

    Already-imported dataset versions are detected and skipped automatically, so you can safely re-run the import to pull in newly added datasets without creating duplicates. A dataset whose import failed is retried in place instead of skipped. Bumping a project's version in Roboflow makes it a new dataset to import rather than an update to the one you already have.

!!! warning "Trashed imports still count as imported"

    A previously imported dataset that you moved to [Trash](../account/trash.md) is still detected and skipped, because it keeps occupying storage during the retention window. Delete it permanently before re-importing the same Roboflow version.

## FAQ

### Does Platform store my Roboflow API key?

Yes, when the workspace owner connects with it; see [saved provider keys](../account/api-keys.md#provider-keys-for-agents-and-annotation).

### Which Roboflow projects are imported?

Object detection, instance segmentation, single-label classification, and keypoint detection projects that have at least one generated version. Multi-label classification and semantic segmentation projects, and projects without a version, are skipped. Select the new datasets you want to import in the preview.

### How do I pick up new datasets after the first import?

Run the import again. Datasets that are already imported, including ones sitting in [Trash](../account/trash.md), are skipped, so only new projects and new versions are transferred.

### What if a dataset import fails?

Open the dataset to see why it failed, then run the import again. The preview lists the failed dataset among the new ones, and the import retries it in place rather than creating a second copy.

### Why does the preview take a long time?

Roboflow generates an export archive for each new dataset version before it can report a size, so workspaces with many fresh versions can take a minute or more to preview. If it times out, run it again — Roboflow keeps generating the exports it already started, so the next preview has less to wait for.
