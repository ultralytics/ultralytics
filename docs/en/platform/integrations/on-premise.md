---
title: On Premise - Ultralytics Platform
description: Connect an Enterprise host to Ultralytics Platform for local dataset ingest, previews, and NVIDIA GPU training without sending pixels to the cloud.
keywords: Ultralytics Platform, On Premise, on-premise computer vision, private datasets, local GPU training, data residency, YOLO
---

# On Premise

{% from "macros/platform-plans.md" import plan_badges %} {{ plan_badges(["Enterprise"]) }}

[On Premise](https://platform.ultralytics.com) connects CPU and optional NVIDIA GPU workers on your own Linux host to Ultralytics Platform. Platform remains the hosted control plane for the UI, authentication, metadata, annotations, and job orchestration, while every pixel and trained model artifact stays on your premises.

Your host needs Docker and outbound HTTPS access to Platform. The installer adds Docker automatically when it is missing, so the normal setup is one command.

## Data Boundary

| Stays on your premises                                     | Stored in Platform                                  |
| ---------------------------------------------------------- | --------------------------------------------------- |
| Source images and videos                                   | Dataset names, paths, dimensions, and revisions     |
| Extracted archives, downloaded NDJSON images, video frames | Classes, labels, annotations, and split assignments |
| Training data, checkpoints, weights, and run artifacts     | Job state, scalar metrics, and worker health        |

Dataset folders are mounted read-only. Platform and its hosted workers never receive the source or derived pixels, and On Premise jobs never fall back to Ultralytics or RunPod compute.

!!! note "Connected On Premise"

    Platform, authentication, and metadata remain hosted. Workers initiate outbound HTTPS connections to claim jobs and report metadata. On Premise is not an air-gapped or fully self-hosted Platform installation, and it does not require a local MongoDB instance.

## Connect a Host

1. Open [Ultralytics Platform](https://platform.ultralytics.com) on the Linux host that can access your datasets.
2. Go to `Settings > Integrations` and select **Connect** on the **On premise** card.
3. Keep the prefilled values or change them:
    - **Machine name:** `On Premise host`
    - **Dataset folder:** `/datasets`
    - **Models folder:** `/models`
4. Select **Create install command**.
5. Copy and run the generated command on the host. It installs Docker Compose when needed, creates the selected folders, starts the CPU worker, and starts the GPU worker automatically when Docker's NVIDIA runtime is available.
6. Close the dialog when the host shows as connected.

The command contains a one-time enrollment token that expires after 10 minutes. The installed worker exchanges it for a revocable worker key stored in a mode-`0600` environment file. It never receives Platform MongoDB or cloud-storage credentials.

!!! info "NVIDIA training"

    CPU ingest only needs Docker. On-premise GPU training also requires a supported NVIDIA driver and NVIDIA Container Toolkit on the host.

## Create an On-Premise Dataset

1. Put the dataset beneath the connected dataset folder. For example, `/datasets/warehouse` is `warehouse` inside the default root.
2. In Platform, select **New Dataset > On premise**.
3. Select the connected host, enter the relative dataset path, choose the task and visibility, and create the dataset.
4. The host indexes the dataset and reports metadata. Platform never uploads the images.

On Premise uses the same CPU ingest code as hosted uploads. It supports:

- loose images and videos;
- ZIP, TAR, TAR.GZ, and TGZ archives;
- Ultralytics NDJSON and COCO JSON;
- YOLO datasets and classification folder layouts; and
- detect, segment, pose, OBB, and classify tasks, including the same class mapping, task inference, validation, and split handling.

The storage output is the only difference. Hosted ingestion may resize or normalize images and create thumbnails in Platform storage. On Premise never resizes, re-encodes, edits, or deletes mounted originals. Archive contents, remote NDJSON assets, and 1 FPS video frames are written only to a Docker volume on the host.

## Preview and Annotate

Platform authorizes each preview, then your browser loads the revision-bound file directly from `http://localhost:8765` on the same computer. No hostname, certificate, VPN, proxy, or preview setting is required.

Annotations are stored as Platform metadata. Editing or deleting an image in Platform changes the Platform reference and annotations only; it never changes a source file or label sidecar.

## Train on a Local GPU

Start training from the normal project training dialog. A dataset bound to an On Premise host is claimable only by that host's GPU worker. Training reads the mounted files, writes checkpoints and weights beneath the configured models folder, and returns job state and scalar metrics to Platform.

On-premise training does not consume Platform compute credits. Ultralytics hosted workers and RunPod cannot claim the job or read its pixels or artifacts.

## Manage the Worker

Use the **On premise** card in `Settings > Integrations` to view CPU/GPU availability, reconnect a host, or disconnect it. Reconnecting rotates the worker secret without changing existing dataset identity. Disconnecting revokes future claims and preview access; it does not delete datasets, source files, cached pixels, or model artifacts from the host.

To inspect or stop the installation on the host:

```bash
cd /opt/ultralytics-worker
docker compose logs -f
docker compose down
```

Also see [Datasets](../data/datasets.md), [Annotation](../data/annotation.md), and [Cloud Training](../train/cloud-training.md).
