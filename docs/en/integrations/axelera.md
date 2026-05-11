---
comments: true
description: Deploy Ultralytics YOLO models on Axelera AI's Metis hardware. Learn how to export, compile, and run high-performance edge inference with up to 856 TOPS.
keywords: Axelera AI, Metis AIPU, Voyager SDK, Edge AI, YOLOv8, YOLO11, YOLO26, Model Export, Computer Vision, PCIe, M.2, Object Detection, quantization
---

# Axelera AI Export and Deployment

Ultralytics partners with [Axelera AI](https://axelera.ai/) to enable high-performance, energy-efficient inference on [Edge AI](https://www.ultralytics.com/glossary/edge-ai) devices. Export and deploy **Ultralytics YOLO models** directly to the **Metis® AIPU** using the **Voyager SDK**.

![Axelera AI edge deployment ecosystem for YOLO](https://github.com/user-attachments/assets/c97a0297-390d-47df-bb13-ff1aa499f34a)

Axelera AI provides dedicated hardware acceleration for [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) at the edge, using a proprietary dataflow architecture and [in-memory computing](https://www.ultralytics.com/glossary/edge-computing) to deliver up to **856 TOPS** with low power consumption.

## Selecting the Right Hardware

Axelera AI offers various form factors to suit different deployment constraints. The chart below helps identify the optimal hardware for your Ultralytics YOLO deployment.

```mermaid
graph TD
    A[Start: Select Deployment Target] --> B{Device Type?}
    B -->|Edge Server / Workstation| C{Throughput Needs?}
    B -->|Embedded / Robotics| D{Space Constraints?}
    B -->|Standalone / R&D| E[Dev Kits & Systems]

    C -->|Max Density <br> 30+ Streams| F[**Metis PCIe x4**<br>856 TOPS]
    C -->|Standard PC <br> Low Profile| G[**Metis PCIe x1**<br>214 TOPS]

    D -->|Drones & Handhelds| H[**Metis M.2**<br>2280 M-Key]
    D -->|High Performance Embedded| I[**Metis M.2 MAX**<br>Extended Thermal]

    E -->|ARM-based All-in-One| J[**Metis Compute Board**<br>RK3588 + AIPU]
    E -->|Prototyping| K[**Arduino Portenta x8**<br>Integration Kit]

    click F "https://store.axelera.ai/"
    click G "https://store.axelera.ai/"
    click H "https://store.axelera.ai/"
    click J "https://store.axelera.ai/"
```

## Hardware Portfolio

The Axelera hardware lineup is optimized to run [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) and legacy versions with high FPS-per-watt efficiency.

### Accelerator Cards

These cards enable AI acceleration in existing host devices, facilitating [brownfield deployments](https://www.ultralytics.com/glossary/edge-computing).

| Product           | Form Factor    | Compute            | Performance (INT8) | Target Application                                                                                                                         |
| :---------------- | :------------- | :----------------- | :----------------- | :----------------------------------------------------------------------------------------------------------------------------------------- |
| **Metis PCIe x4** | PCIe Gen3 x16  | **4x** Metis AIPUs | **856 TOPS**       | High-density [video analytics](https://docs.ultralytics.com/guides/analytics), smart cities                                                |
| **Metis PCIe x1** | PCIe Gen3 x1   | **1x** Metis AIPU  | **214 TOPS**       | Industrial PCs, retail [queue management](https://docs.ultralytics.com/guides/queue-management)                                            |
| **Metis M.2**     | M.2 2280 M-Key | **1x** Metis AIPU  | **214 TOPS**       | [Drones](https://www.ultralytics.com/blog/build-ai-powered-drone-applications-with-ultralytics-yolo11), robotics, portable medical devices |
| **Metis M.2 MAX** | M.2 2280       | **1x** Metis AIPU  | **214 TOPS**       | Environments requiring advanced thermal management                                                                                         |

### Integrated Systems

For turnkey solutions, Axelera partners with manufacturers to provide systems pre-validated for the Metis AIPU.

- **Metis Compute Board**: A standalone edge device pairing the Metis AIPU with a Rockchip RK3588 ARM CPU.
- **Workstations**: Enterprise towers from **Dell** (Precision 3460XE) and **Lenovo** (ThinkStation P360 Ultra).
- **Industrial PCs**: Ruggedized systems from **Advantech** and **Aetina** designed for [manufacturing automation](https://www.ultralytics.com/solutions/ai-in-manufacturing).

## Supported Tasks

The following tasks are supported across YOLOv8, YOLO11, and YOLO26 models.

| Task                                                              | YOLOv8 | YOLO11 | YOLO26              |
| :---------------------------------------------------------------- | :----- | :----- | :------------------ |
| [Object Detection](https://docs.ultralytics.com/tasks/detect)     | ✅     | ✅     | ✅                  |
| [Pose Estimation](https://docs.ultralytics.com/tasks/pose)        | ✅     | ✅     | ✅                  |
| [Segmentation](https://docs.ultralytics.com/tasks/segment)        | ✅     | ✅     | ⚠️ Voyager SDK only |
| [Oriented Bounding Boxes](https://docs.ultralytics.com/tasks/obb) | ✅     | ✅     | ✅                  |
| [Classification](https://docs.ultralytics.com/tasks/classify)     | ✅     | ✅     | ✅                  |

!!! note

    YOLO26 segmentation is not yet supported through the Ultralytics `export` command. Users who need YOLO26-seg can deploy via the [Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk) using `deploy.py`, which provides a user-space workaround. Native compiler support will be added in a future release.

## Installation

!!! warning "Platform Requirements"

    Exporting to Axelera format requires:

    - **Operating System**: Linux only (Ubuntu 22.04/24.04 recommended)
    - **Hardware**: Axelera AI accelerator ([Metis devices](https://store.axelera.ai/))
    - **Python**: Versions 3.10, 3.11, and 3.12
    - **System dependency**: `sudo apt install libgl1` (required by OpenCV, not included via `pip`)

### Ultralytics Installation

```bash
pip install ultralytics
```

For detailed instructions, see our [Ultralytics Installation guide](../quickstart.md). If you encounter difficulties, consult our [Common Issues guide](../guides/yolo-common-issues.md).

### Axelera Driver Installation

1. Add the Axelera repository key:

    ```bash
    sudo sh -c "curl -fsSL https://software.axelera.ai/artifactory/api/security/keypair/axelera/public | gpg --dearmor -o /etc/apt/keyrings/axelera.gpg"
    ```

2. Add the repository to apt:

    Choose the appropriate snippet from below to match the OS being used.

    ```bash
    # Ubuntu 22.04
    sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu22 main' > /etc/apt/sources.list.d/axelera.list"
    ```

    ```bash
    # Ubuntu 24.04
    sudo sh -c "echo 'deb [signed-by=/etc/apt/keyrings/axelera.gpg] https://software.axelera.ai/artifactory/axelera-apt-source ubuntu24 main' > /etc/apt/sources.list.d/axelera.list"
    ```

3. Install the SDK and load the driver:

    ```bash
    sudo apt update
    sudo apt install -y metis-dkms=1.4.16
    sudo modprobe metis
    ```

!!! tip "First run downloads the SDK automatically"

    The first `yolo export format=axelera` or `yolo predict` with an Axelera model will automatically download and install the Axelera SDK packages. This may take several minutes depending on your connection speed, and no progress is shown during the download. To install manually beforehand:

    ```bash
    pip install axelera-devkit==1.6.0 --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple
    pip install axelera-rt==1.6.0 --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple
    ```

## Exporting YOLO Models to Axelera

Export your trained YOLO models using the standard Ultralytics export command.

!!! example "Export to Axelera Format"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export to Axelera format
        model.export(format="axelera")  # creates 'yolo26n_axelera_model' directory
        ```

    === "CLI"

        ```bash
        yolo export model=yolo26n.pt format=axelera
        ```

!!! warning "First export may fail after dependency update"

    The Axelera compiler requires `numpy<2`. If your environment has `numpy>=2`, the first `yolo export` will auto-downgrade it but the export will fail due to the stale module state. Simply run the same export command again — it will succeed on the second run.

### Export Arguments

| Argument   | Type             | Default          | Description                                                                                                                             |
| :--------- | :--------------- | :--------------- | :-------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'axelera'`      | Target format for Axelera Metis AIPU hardware.                                                                                          |
| `imgsz`    | `int` or `tuple` | `640`            | Image size for model input.                                                                                                             |
| `batch`    | `int`            | `1`              | Specifies export model batch inference size or the max number of images the exported model will process concurrently in `predict` mode. |
| `int8`     | `bool`           | `True`           | Enable [INT8 quantization](https://www.ultralytics.com/glossary/model-quantization) for AIPU.                                           |
| `data`     | `str`            | `'coco128.yaml'` | [Dataset](https://docs.ultralytics.com/datasets) config for quantization calibration.                                                   |
| `fraction` | `float`          | `1.0`            | Fraction of dataset for calibration (100-400 images recommended).                                                                       |
| `device`   | `str`            | `None`           | Export device: GPU (`device=0`) or CPU (`device=cpu`).                                                                                  |

For all export options, see the [Export Mode documentation](https://docs.ultralytics.com/modes/export).

### Output Structure

```
yolo26n_axelera_model/
├── yolo26n.axm              # Axelera model file
└── metadata.yaml            # Model metadata (classes, image size, etc.)
```

## Running Inference

Load the exported model with the Ultralytics API and run inference, similar to loading [ONNX](https://docs.ultralytics.com/integrations/onnx) models.

!!! example "Inference with Axelera Model"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported Axelera model
        model = YOLO("yolo26n_axelera_model")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")

        # Process results
        for r in results:
            print(f"Detected {len(r.boxes)} objects")
            r.show()  # Display results
        ```

    === "CLI"

        ```bash
        yolo predict model='yolo26n_axelera_model' source='https://ultralytics.com/images/bus.jpg'
        ```

## Axelera AI Benchmarks

The Metis AIPU maximizes throughput while minimizing energy consumption.

| Model   | Metis PCIe FPS (frames per second) | Metis M.2 FPS (frames per second) |
| ------- | ---------------------------------- | --------------------------------- |
| YOLOv8n | 847                                | 771                               |
| YOLO11n | 746                                | 574                               |
| YOLO26n | 648.6                              | 484.9                             |

_Benchmarks based on Axelera AI data. Actual FPS depends on model size, batching, and input resolution._

## Real-World Applications

Ultralytics YOLO on Axelera hardware enables advanced edge computing solutions:

- **Smart Retail**: Real-time [object counting](https://docs.ultralytics.com/guides/object-counting) and [heatmap analytics](https://docs.ultralytics.com/guides/heatmaps) for store optimization.
- **Industrial Safety**: Low-latency [PPE detection](https://docs.ultralytics.com/datasets/detect/construction-ppe) in manufacturing environments.
- **Drone Analytics**: High-speed [object detection](https://docs.ultralytics.com/tasks/detect) on UAVs for [agriculture](https://www.ultralytics.com/solutions/ai-in-agriculture) and search-and-rescue.
- **Traffic Systems**: Edge-based [license plate recognition](https://www.ultralytics.com/blog/using-ultralytics-yolo11-for-automatic-number-plate-recognition) and [speed estimation](https://docs.ultralytics.com/guides/speed-estimation).

## Recommended Workflow

1. **Train** your model using Ultralytics [Train Mode](https://docs.ultralytics.com/modes/train)
2. **Export** to Axelera format using `model.export(format="axelera")`
3. **Validate** accuracy with `yolo val` to verify minimal quantization loss
4. **Predict** using `yolo predict` for qualitative validation
5. **Deploy** to a high-performance end-to-end pipeline without PyTorch dependency — see the [YOLO on Voyager SDK examples](https://github.com/ultralytics/ultralytics/tree/main/examples/YOLO-Axelera-Python) for composable Python pipelines using `axelera-rt`

## Device Health Check

Verify your Axelera device is functioning properly:

```bash
# if axdevice cannot be found, please run at least one inference (see above) to ensure the required packages are installed
axdevice
```

For detailed diagnostics, see the [AxDevice documentation](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/docs/reference/axdevice.md).

## Maximum Performance

This integration uses single-core configuration for compatibility. For production requiring maximum throughput, the [Axelera Voyager SDK](https://github.com/axelera-ai-hub/voyager-sdk) offers:

- Multi-core utilization (quad-core Metis AIPU)
- Streaming inference pipelines
- Tiled inferencing for higher-resolution cameras

See the [model-zoo](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/docs/reference/model_zoo.md) for FPS benchmarks or [contact Axelera](https://axelera.ai/contact-us) for production support.

## Known Issues

!!! warning "Known Limitations"

    - **M.2 power limitations**: Large or extra-large models may encounter runtime errors on M.2 accelerators due to power supply constraints.

For support, visit the [Axelera Community](https://community.axelera.ai/).

## FAQ

### What YOLO versions are supported on Axelera?

The Voyager SDK supports export of [YOLOv8](https://docs.ultralytics.com/models/yolov8), [YOLO11](https://docs.ultralytics.com/models/yolo11), and [YOLO26](https://docs.ultralytics.com/models/yolo26) models. See [Supported Tasks](#supported-tasks) for per-model task availability.

### Can I deploy custom-trained models?

Yes. Any model trained using [Ultralytics Train Mode](https://docs.ultralytics.com/modes/train) can be exported to the Axelera format, provided it uses supported layers and operations.

### How does INT8 quantization affect accuracy?

Axelera's Voyager SDK automatically quantizes models for the mixed-precision AIPU architecture. For most [object detection](https://www.ultralytics.com/glossary/object-detection) tasks, the performance gains (higher FPS, lower power) significantly outweigh the minimal impact on [mAP](https://docs.ultralytics.com/guides/yolo-performance-metrics). Quantization takes seconds to several hours depending on model size. Run `yolo val` after export to verify accuracy.

### How many calibration images should I use?

We recommend 100 to 400 images. More than 400 provides no additional benefit and increases quantization time. Experiment with 100, 200, and 400 images to find the optimal balance.

### Where can I find the Voyager SDK?

The SDK, drivers, and compiler tools are available via the [Axelera Developer Portal](https://github.com/axelera-ai-hub/voyager-sdk).
