---
plans: [free, pro, enterprise]
comments: true
description: Import YOLO model checkpoints and training results from Weights & Biases projects into Ultralytics Platform.
keywords: Ultralytics Platform, Weights & Biases, W&B, model import, best.pt, training results, YOLO
title: Weights & Biases Model Import - Ultralytics Platform
---

# Weights & Biases Integration

Import YOLO `best.pt` checkpoints from Weights & Biases (W&B) into [Ultralytics Platform](https://platform.ultralytics.com). Imported checkpoints become regular [Platform models](../train/models.md), ready to inspect, predict, export, or deploy.

To log new training runs to W&B, see the [YOLO experiment tracking guide](../../integrations/weights-biases.md).

## Import from W&B

1. Switch to the Platform workspace that should receive the models.
2. Go to **Settings > [Integrations](index.md)** and select **Weights & Biases**.
3. Paste your [W&B API key](https://wandb.ai/authorize) and click **Connect**. If the workspace has a [saved W&B key](../account/api-keys.md#provider-keys-for-agents-and-annotation), Platform connects with it automatically.
4. Review the projects with logged models from your W&B entity and teams. Model counts, tasks, and sizes load as you browse the table.
5. Select the projects to import. Nothing is selected by default, and a project becomes selectable once its importable models have loaded.
6. Click **Import N**, where N is the number of selected projects. Platform imports all eligible models from each selected project.

Imports appear in the background upload tracker and as in-flight rows in the destination project's models sidebar. Keep the browser tab open until uploads finish; you can navigate within Platform while they continue. The preview remains available afterward so you can select more projects without reconnecting.

![Ultralytics Platform Settings Integrations Weights & Biases Import Preview](https://cdn.ul.run/i/04d4d2c502aab468324713c7f6cfbc3d.avif)<!-- screenshot -->

## Eligible Checkpoints and Training Results

A run is eligible when it has a W&B **model artifact** containing `best.pt`, reports a [supported Platform task](../data/index.md#supported-tasks), and the checkpoint is within the **1 GB model upload limit**. The [Ultralytics W&B integration](../../integrations/weights-biases.md) logs this artifact after training when saving is enabled and a best checkpoint exists. Projects without eligible runs cannot be selected.

Platform reads the checkpoint's training arguments, task, metadata, and available metrics just as it does for a [model upload](../train/models.md#upload-model). When a checkpoint lacks per-epoch results, Platform tries to fill the training curves from the run's W&B history. Metrics that were not logged remain unavailable; a history retrieval failure does not prevent the checkpoint from importing.

## Destination Projects and Re-imports

Each selected W&B project imports into a Platform project of the same name, created if needed. Existing model names receive a numeric suffix when necessary.

!!! warning "Re-importing adds models again"

    W&B runs are not deduplicated against previous imports. Importing the same project again creates additional models rather than updating the existing ones.

If a destination project is in [Trash](../account/trash.md), restore it or permanently delete it before retrying. A W&B project whose name resolves to the reserved Platform project name `deploy` cannot be imported directly; download its checkpoints from W&B and [upload them](../train/models.md#upload-model) into another Platform project instead.

## Storage and Permissions

W&B imports are available on every plan and require the workspace editor [role](../account/teams.md#roles-and-permissions) or higher. Imported checkpoints count against that workspace's [storage and model limits](../account/billing.md).

The preview disables importing when the selected checkpoints exceed the displayed storage headroom. Click **Connect** again to refresh storage usage; limits are checked again during upload. Projects that cannot be queued stay selected for a retry, and a notification reports failed model imports.

A key the workspace owner connects with is saved for the workspace; see [saved provider keys](../account/api-keys.md#provider-keys-for-agents-and-annotation).
