---
plans: [enterprise]
title: On Premise - Ultralytics Platform
description: Connect an Enterprise host to Ultralytics Platform for local dataset ingest, previews, and NVIDIA GPU training without sending pixels to the cloud.
keywords: Ultralytics Platform, On Premise, on-premise computer vision, private datasets, local GPU training, data residency, YOLO
---

# On Premise

[On Premise](https://platform.ultralytics.com) connects CPU and optional NVIDIA GPU workers on your own Linux, macOS, or Windows host to Ultralytics Platform. Platform remains the hosted control plane for the UI, authentication, metadata, annotations, and job orchestration, while every pixel and trained model artifact stays on your premises.

Your host needs Docker and outbound HTTPS access to Platform. The installer adds Docker automatically when it is missing, so the normal setup is one command.

## System Requirements

|                  | Minimum                                                         | Recommended                                                                      |
| ---------------- | --------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| Operating system | 64-bit Linux, Apple Silicon macOS, or x86-64 Windows with WSL 2 | Current OS and Docker releases                                                   |
| CPU              | 4 cores                                                         | 8 or more cores for CPU training                                                 |
| Memory           | 8 GB RAM                                                        | 16 GB or more                                                                    |
| Storage          | 20 GB free plus space for datasets and models                   | SSD with free space at least twice the working dataset size plus model artifacts |
| Network          | Outbound HTTPS to Platform and container registries             | Stable broadband for initial image pulls                                         |

CPU ingest and training work on all three operating systems. The installer selects the official native arm64 image on Apple Silicon and ARM Linux, so small jobs such as YOLO26n on COCO8 run without x86 emulation. NVIDIA acceleration is optional; when it is unavailable, training runs on CPU.

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

1. Open [Ultralytics Platform](https://platform.ultralytics.com) on the Linux, macOS, or Windows host that can access your datasets.
2. Go to `Settings > Integrations` and select **Connect** on the **On Premise** card.
3. Platform selects the detected Linux, macOS, or Windows command. Keep the prefilled values or change them:
    - **Machine name:** `On Premise host`
    - **Dataset folder:** `/datasets` on Linux or `~/Ultralytics/datasets` on macOS and Windows
    - **Models folder:** `/models` on Linux or `~/Ultralytics/models` on macOS and Windows
4. Select **Create install command**. The dialog tells you which terminal to open for the selected operating system.
5. Copy the complete command, paste it into that terminal, and run it. The command includes the one-time enrollment token, installs and starts Docker when needed, and creates the selected folders.
6. Leave the dialog open. Platform checks every 500 milliseconds and shows the host as connected when the CPU worker starts. A GPU worker starts automatically when Docker exposes a supported NVIDIA runtime.

The enrollment token expires after 10 minutes and can be exchanged only once. The installed worker stores the resulting revocable worker key in a mode-`0600` environment file. It never receives Platform MongoDB or cloud-storage credentials. Compose restarts the workers automatically, and setup configures Docker to start at boot on Linux or sign-in on macOS and Windows.

!!! info "Training hardware"

    CPU ingest and training only need Docker. Optional GPU acceleration also requires a supported NVIDIA driver and container runtime on the host.

## Create an On Premise Dataset

1. Put the dataset beneath the connected dataset folder. For example, `/datasets/warehouse` is `warehouse` inside the default root.
2. In Platform, select **New Dataset > On Premise**.
3. Browse the connected host with the same folder browser used for Google Cloud Storage, Amazon S3, and Azure Blob Storage, select a folder, choose the task, and create the private dataset.
4. The host indexes the dataset and reports metadata. Platform never uploads the images.

On Premise uses the same CPU ingest code as hosted uploads. It supports:

- loose images and videos;
- ZIP, TAR, TAR.GZ, and TGZ archives;
- Ultralytics NDJSON and COCO JSON;
- YOLO datasets and classification folder layouts; and
- detect, segment, pose, OBB, and classify tasks, including the same class mapping, task inference, validation, and split handling.

The storage output is the only difference. Hosted ingestion may resize or normalize images and create thumbnails in Platform storage. On Premise never resizes, re-encodes, edits, or deletes mounted originals. Archive contents, remote NDJSON assets, and video frames sampled at 1 FPS up to 100 frames, then evenly across longer videos, are written only to a Docker volume on the host.

## Preview and Annotate

Platform authorizes each preview, then your browser loads the revision-bound file directly from `http://localhost:8765` on the same computer. No hostname, certificate, VPN, proxy, or preview setting is required.

Annotations are stored as Platform metadata. Editing or deleting an image in Platform changes the Platform reference and annotations only; it never changes a source file or label sidecar.

## Train Locally

Start training from the normal project training dialog. A dataset bound to an On Premise host is claimable only by that host. Platform uses its GPU worker when available and otherwise runs the same training code on its CPU worker. Training reads the mounted files, writes checkpoints and weights beneath the configured models folder, and returns job state, scalar metrics, and the immutable checkpoint reference to Platform. Model downloads use the same signed `localhost` connection as previews, so the weights move directly from your host to your browser.

On Premise training does not consume Platform compute credits. Ultralytics hosted workers and RunPod cannot claim the job or read its pixels or artifacts.

## Manage the Worker

Use the **On Premise** card in `Settings > Integrations` to view CPU/GPU availability, reconnect a host, or disconnect it. Reconnecting rotates the worker secret without changing existing dataset identity. Disconnecting revokes future claims and preview access; it does not delete datasets, source files, cached pixels, or model artifacts from the host.

To inspect or stop the installation on Linux:

```bash
cd /opt/ultralytics-worker
docker compose logs -f
docker compose down
```

On macOS and Windows, the installer prints the equivalent command using `~/.ultralytics/worker`.

Also see [Datasets](../data/datasets.md), [Annotation](../data/annotation.md), and [Cloud Training](../train/cloud-training.md).
