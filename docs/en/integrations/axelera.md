---
comments: true
description: Learn how Ultralytics can run on Axelera Metis accelerators using the Voyager SDK for efficient edge AI inference.
keywords: Ultralytics, Axelera, Metis, Voyager SDK, edge AI, inference accelerator, PCIe, M.2, YOLO
---

# Axelera Integration for Ultralytics YOLO

Axelera builds edge AI accelerators that target high-throughput, low-power inference. The Metis family starts at 214 TOPs on compact M.2 cards, scales up to 629 TOPs on Europa modules, and is paired with the Voyager SDK that ships with 100+ ready-to-run models. This placeholder page outlines what Axelera offers while we finalize step-by-step export and deployment instructions for Ultralytics models.

## Hardware Snapshot

- **Metis M.2 and M.2 MAX**: Compact cards powered by quad-core Metis AIPUs for edge devices and industrial PCs.
- **Metis PCIe (x1/x4)**: Expansion cards ranging from single- to multi-AIPU configurations for higher channel counts.
- **Evaluation systems**: Bundles with popular hosts like RK3588 compute boards, Lenovo ThinkStation, Dell workstations, and Arduino Portenta-based kits to accelerate bring-up.

These options support multi-stream video analytics, quality inspection, and people monitoring while keeping power draw and cost below typical GPU deployments.

## Voyager SDK Overview

Voyager SDK provides the toolchain for compiling, deploying, and monitoring models on Metis accelerators. It includes a catalog of optimized computer vision models and example pipelines to speed up development. The SDK is designed for quick iteration on edge devices, complementing Ultralytics workflows that start with training and validation in Python.

## Using Ultralytics with Axelera (Coming Soon)

We are preparing a full guide that will cover:

1. Exporting YOLO models into an Axelera-ready format.
2. Running single- and multi-stream inference on Metis hardware.
3. Profiling throughput, latency, and power for common camera workloads.

If you'd like early access or to validate a specific use case, reach out to Axelera at `info@axelera.ai` or visit [axelera.ai](https://www.axelera.ai/) to explore hardware availability and SDK downloads.
