---
title: Luxonis OAK Deployment for Ultralytics YOLO
comments: true
description: Learn how to convert and deploy Ultralytics YOLO models on Luxonis OAK cameras using Luxonis Hub or local Tools and ModelConverter workflows for RVC2 and RVC4 devices.
keywords: Luxonis, OAK camera, DepthAI, RVC2, RVC4, Luxonis Hub, hubai-sdk, Tools, ModelConverter, NN Archive, edge AI, YOLO deployment, Ultralytics
---

# Luxonis OAK Deployment for Ultralytics YOLO

!!! warning "Not a direct Ultralytics export format"

    Luxonis deployment is **not currently supported** as a native Ultralytics `model.export(format="...")` target. This guide documents a manual workflow that converts Ultralytics YOLO models with Luxonis tooling for deployment on Luxonis OAK cameras.

Luxonis [OAK cameras](https://www.luxonis.com/) are [edge AI](https://www.ultralytics.com/glossary/edge-ai) vision devices that combine image sensors with on-device compute for tasks such as [object detection](../tasks/detect.md), [instance segmentation](../tasks/segment.md), and [pose estimation](../tasks/pose.md). Running inference directly on the camera removes the need for a host GPU and reduces latency and bandwidth.

This guide covers the two OAK hardware generations, conversion of a YOLO `.pt` checkpoint with [Luxonis Hub](https://hub.luxonis.com/) or with local Luxonis tooling, and inference on-device with DepthAI.

## Luxonis Hardware Generations

This guide covers the two current OAK hardware generations, [`RVC2`](https://docs.luxonis.com/hardware/platform/rvc/rvc2) and [`RVC4`](https://docs.luxonis.com/hardware/platform/rvc/rvc4). Both run Ultralytics YOLO models, but each requires its own converted artifact. For the older `RVC3` platform, see the Luxonis [hardware documentation](https://docs.luxonis.com/hardware/platform/rvc/rvc3).

- `RVC2` is built around the Intel Movidius Myriad X and powers camera families such as [OAK-D](https://docs.luxonis.com/hardware/products/OAK-D) and [OAK-1](https://docs.luxonis.com/hardware/products/OAK-1). Converted models are `.superblob` artifacts executed through an [OpenVINO](https://docs.openvino.ai/)-based path.
- `RVC4` is built around the [Qualcomm QCS8550](https://www.qualcomm.com/internet-of-things/products/q8-series/qcs8550) and powers the [OAK 4 family](https://www.luxonis.com/oak4). Converted models are `.dlc` artifacts executed through a [SNPE](https://www.qualcomm.com/developer/software/neural-processing-sdk-for-ai)-based path. RVC4 has more on-device compute than RVC2 and supports INT8 [quantization](https://www.ultralytics.com/glossary/model-quantization).

!!! note "Compatibility at a glance"

    - Conversion support covers Ultralytics YOLO families from `YOLOv5` through `YOLO26`; the exact model and task matrix is maintained in the Luxonis [Tools supported models list](https://github.com/luxonis/tools#-supported-models). For example, `YOLO26` lists detection, instance segmentation, pose, and semantic segmentation, while classification and OBB are listed for `YOLOv8` and `YOLO11`.
    - On-device output decoding with `DetectionNetwork` covers detection, instance segmentation, and pose models. Other tasks use a generic `NeuralNetwork` node with a Luxonis parser node or host-side parsing; see the Luxonis [inference documentation](https://docs.luxonis.com/software-v3/ai-inference/inference/).
    - Converted models are generation-specific: an `RVC2` artifact does not run on `RVC4`, and an `RVC4` artifact does not run on `RVC2`.

Ultralytics `.pt` checkpoints must be packaged as an [NN Archive](https://docs.luxonis.com/software-v3/ai-inference/nn-archive), which bundles the compiled model for the target generation with the metadata DepthAI needs to preprocess inputs and decode outputs. Either conversion path below produces this archive from a `.pt` checkpoint, custom-trained with [Train mode](../modes/train.md) or an official Ultralytics model such as those on the [YOLO26 models page](../models/yolo26.md#performance-metrics).

## Conversion Path 1: Luxonis Hub

[Luxonis Hub](https://hub.luxonis.com/) provides a hosted conversion workflow and does not require a local toolchain. [Quick Conversion](https://docs.luxonis.com/cloud/hubai/quick-conversion) is the simplest browser-based route:

1. Choose `YOLO` as the source model type.
2. Select the target platform: `RVC2` or `RVC4`.
3. Upload the `.pt` file.
4. Set the input shape and any advanced options.
5. Submit the conversion and wait for the cloud job to complete.

For model versioning, team sharing, or custom calibration data for `RVC4` INT8 quantization, use the [Model Registry](https://docs.luxonis.com/cloud/hubai/model-registry/concepts) with [Detailed Conversion](https://docs.luxonis.com/cloud/hubai/model-registry/detailed-conversion) or the [HubAI SDK](https://docs.luxonis.com/cloud/hubai/model-registry/hubai-sdk/). Custom quantization datasets are only supported through the SDK.

### Using `hubai-sdk`

[`hubai-sdk`](https://docs.luxonis.com/cloud/hubai/model-registry/hubai-sdk/) is the programmatic Python and CLI interface to Luxonis Hub. It requires Python 3.10 or newer. Create an [API key](https://docs.luxonis.com/cloud/api/api-keys/) in Luxonis Hub and expose it as `HUBAI_API_KEY` before running the example.

!!! example "Convert a `YOLO26` checkpoint to `RVC4` with `hubai-sdk`"

    === "CLI"

        ```bash
        pip install hubai-sdk
        ```

    === "Python"

        ```python
        import os

        from hubai_sdk import HubAIClient

        client = HubAIClient(api_key=os.getenv("HUBAI_API_KEY"))

        response = client.convert.RVC4(
            path="yolo26n.pt",
            name="yolo26n-rvc4",
            quantization_mode="INT8_STANDARD",
            quantization_data="GENERAL",
            yolo_input_shape=[640, 640],
        )

        print(f"Converted model downloaded to: {response.downloaded_path}")
        ```

The result is an NN Archive hosted in Luxonis Hub, which DepthAI can load by its model identifier or from the downloaded `.tar.xz` file.

## Conversion Path 2: Local Conversion with Tools and ModelConverter

Local conversion is the offline alternative and runs in two stages:

1. **Convert** the `.pt` checkpoint into an `ONNX` NN Archive with [Tools](https://github.com/luxonis/tools). Tools restructures the exported YOLO outputs so that DepthAI can decode them on-device, so use it instead of a generic PyTorch-to-ONNX export.
2. **Compile** that archive into an `RVC2` or `RVC4` NN Archive with [ModelConverter](https://docs.luxonis.com/software-v3/ai-inference/conversion/rvc-conversion/offline/modelconverter/), which runs the target-specific toolchain in [Docker](https://docs.docker.com/get-docker/).

### Stage 1: Convert `.pt` to an `ONNX` NN Archive with Tools

Install Tools from source as documented in the [Tools README](https://github.com/luxonis/tools#-how-to-run). The constraints file is required with pip >= 26.2.

!!! example "Install Tools and convert a model"

    === "CLI"

        ```bash
        git clone --recursive https://github.com/luxonis/tools.git
        cd tools
        PIP_CONSTRAINT=constraints.txt PIP_BUILD_CONSTRAINT=constraints.txt pip install .

        tools yolo26n.pt --imgsz "640 640"
        ```

### Stage 2: Compile the `ONNX` NN Archive for `RVC2` or `RVC4` with ModelConverter

ModelConverter requires Python 3.10 or newer, accepts any local path or remote URL for the archive, and writes results to an `output/` directory in the current working directory. `RVC4` conversion quantizes to INT8 by default and needs a directory of representative calibration images passed as the `calibration.path` config override; without it, ModelConverter calibrates on random data.

!!! example "Install ModelConverter and compile for `RVC4`"

    === "CLI"

        ```bash
        pip install modelconv

        modelconverter convert rvc4 --path yolo26n.tar.xz calibration.path calibration_images/
        ```

Use `rvc2` in place of `rvc4` to target `RVC2` devices, which do not use calibration data. For tool versions, quantization modes, and further config overrides, see the [ModelConverter README](https://github.com/luxonis/modelconverter#readme).

## Running Inference on OAK Cameras

Inference uses [DepthAI v3](https://docs.luxonis.com/software-v3/depthai). YOLO detection, instance segmentation, and pose models are decoded on-device by [`dai.node.DetectionNetwork`](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/detection_network), which emits [ImgDetections](https://docs.luxonis.com/software-v3/depthai/depthai-components/messages/img_detections/).

Models hosted on Luxonis Hub are referenced by the model identifier shown next to each conversion. For private models, set `DEPTHAI_HUB_API_KEY` to your [API key](https://docs.luxonis.com/cloud/api/api-keys/) so DepthAI can authenticate.

!!! example "Run a YOLO model with `DetectionNetwork`"

    === "CLI"

        ```bash
        pip install depthai
        ```

    === "Python"

        ```python
        import depthai as dai

        model = "your-model-identifier"
        # Or load a local NN Archive:
        # model = dai.NNArchive("path/to/model.tar.xz")

        visualizer = dai.RemoteConnection()

        with dai.Pipeline() as pipeline:
            camera = pipeline.create(dai.node.Camera).build()
            detection = pipeline.create(dai.node.DetectionNetwork).build(camera, model)

            visualizer.addTopic("rgb", detection.passthrough, group="RGB")
            visualizer.addTopic("detections", detection.out, group="RGB")

            pipeline.start()
            visualizer.registerPipeline(pipeline)

            while pipeline.isRunning():
                if visualizer.waitKey(1) == ord("q"):
                    pipeline.stop()
        ```

Open `http://localhost:8080` in a browser to view the RGB stream and detections in the OAK Visualizer. On cameras with a stereo pair, [`dai.node.SpatialDetectionNetwork`](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/spatial_detection_network) adds 3D coordinates to each detection.

## Examples and Benchmarks

Pre-converted Ultralytics YOLO models are available in the Luxonis [Model Zoo](https://models.luxonis.com/?search=YOLO), and reference pipelines are in the [oak-examples](https://github.com/luxonis/oak-examples) repository. To measure throughput for a specific model and device, follow the Luxonis [benchmarking documentation](https://docs.luxonis.com/software-v3/ai-inference/benchmarking/).

## FAQ

### Which Ultralytics YOLO models are supported on Luxonis OAK cameras?

Conversion covers YOLO families from `YOLOv5` through [YOLO26](../models/yolo26.md), with task coverage that differs by family. The Luxonis [Tools supported models list](https://github.com/luxonis/tools#-supported-models) is the authoritative matrix. On-device decoding with `DetectionNetwork` is available for detection, instance segmentation, and pose models.

### Can I run the same converted model on both `RVC2` and `RVC4` devices?

No. `RVC2` uses `.superblob` artifacts in an OpenVINO-based runtime and `RVC4` uses `.dlc` artifacts in a SNPE-based runtime, so a model must be converted separately for each generation.

### Should I use quantization for `RVC4`?

`RVC4` conversion supports several modes, including `INT8_STANDARD` (default, requires calibration data) and `FP16_STANDARD` (no calibration). INT8 gives higher throughput and a smaller model; FP16 preserves more of the original accuracy. With INT8, calibration images should be representative of the deployment scene.
