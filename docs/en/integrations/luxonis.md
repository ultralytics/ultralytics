---
title: Luxonis OAK Deployment for Ultralytics YOLO
comments: true
description: Convert and deploy Ultralytics YOLO models on Luxonis OAK RVC2 and RVC4 cameras using Luxonis Hub or local Tools and ModelConverter workflows.
keywords: Luxonis, OAK camera, DepthAI, RVC2, RVC4, Luxonis Hub, Tools, ModelConverter, NN Archive, edge AI, YOLO deployment, Ultralytics
---

# Luxonis OAK Deployment for Ultralytics YOLO

[Luxonis OAK cameras](https://www.luxonis.com/) run Ultralytics YOLO models on-device after conversion with Luxonis tooling. This is a manual deployment workflow; Luxonis is not a native Ultralytics `model.export(format="...")` target.

This guide covers [RVC2](https://docs.luxonis.com/hardware/platform/rvc/rvc2) and [RVC4](https://docs.luxonis.com/hardware/platform/rvc/rvc4) devices. Convert for your camera's platform: RVC2 and RVC4 artifacts are not interchangeable. Supported model families and tasks vary; check the [Luxonis Tools support matrix](https://github.com/luxonis/tools#-supported-models) before conversion.

Start with a supported YOLO `.pt` checkpoint, such as `yolo26n.pt`, or your own [trained model](../modes/train.md). The converted [NN Archive](https://docs.luxonis.com/software-v3/ai-inference/nn-archive) packages the model and runtime metadata.

## Cloud Conversion with Luxonis Hub

1. Sign in to [Luxonis Hub](https://hub.luxonis.com/) and open [Quick Conversion](https://docs.luxonis.com/cloud/hubai/quick-conversion).
2. Select **YOLO**, choose **RVC2** or **RVC4**, and upload your `.pt` checkpoint.
3. Set the input shape and conversion parameters, then submit the conversion.
4. After completion, copy the converted model's identifier for the inference example below, or download its NN Archive.

For automated conversion or custom RVC4 calibration data, use the [HubAI SDK](https://docs.luxonis.com/cloud/hubai/model-registry/hubai-sdk/). Follow its installation instructions in a Python 3.10+ environment. [Detailed Conversion](https://docs.luxonis.com/cloud/hubai/model-registry/detailed-conversion) provides registry management and predefined calibration datasets.

## Local Conversion with Tools and ModelConverter

Use Python 3.10+ and [Docker](https://docs.docker.com/get-docker/). Install [Luxonis Tools](https://github.com/luxonis/tools#-how-to-run) using its upstream setup instructions, including the dependency constraints. Tools prepares YOLO outputs for Luxonis parsing; a generic ONNX export does not replace this step.

```bash
# Export the checkpoint to an ONNX NN Archive
tools yolo26n.pt --imgsz "640 640" --output-dir output

# Install the target-platform converter
pip install modelconv==0.6.0
```

Tools writes `yolo26n.tar.xz` inside a timestamped directory under `output/`. Replace the example path below with that file. For RVC4 INT8 conversion, populate `calibration_images/` with representative deployment images; see the [calibration data requirements](https://github.com/luxonis/modelconverter#calibration-data).

```bash
modelconverter convert rvc4 --path output/yolo26n_20260807_104858/yolo26n.tar.xz \
  calibration.path calibration_images/
```

For RVC2, use `rvc2` instead of `rvc4`; calibration images are not needed for its FP16 conversion. Use the resulting **compiled** NN Archive from ModelConverter's output directory for inference, rather than the intermediate ONNX archive. See [ModelConverter](https://github.com/luxonis/modelconverter) for output paths and platform-specific options.

## Run Inference on an OAK Camera

Connect a compatible OAK camera and install DepthAI in your Python environment:

```bash
pip install depthai==3.9.0
```

For a private Hub model, set `DEPTHAI_HUB_API_KEY` to a [Luxonis Hub API key](https://docs.luxonis.com/cloud/api/api-keys/) before running the script. Replace `your-model-identifier` with the identifier from Hub, or use the local compiled NN Archive alternative.

```python
import depthai as dai

model = "your-model-identifier"
# Alternatively, use your converted local artifact:
# model = dai.NNArchive("path/to/model.rvc4.tar.xz")

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

Open `http://localhost:8082` to view the RGB stream and detections. `DetectionNetwork` also supports compatible [instance segmentation](https://github.com/luxonis/depthai-core/blob/v3.9.0/examples/python/DetectionNetwork/detection_and_segmentation.py) and [pose](https://github.com/luxonis/depthai-core/blob/v3.9.0/examples/python/DetectionNetwork/detection_and_keypoints.py) models. For other tasks, including semantic segmentation and classification, follow the [task-specific inference guidance](https://docs.luxonis.com/software-v3/ai-inference/inference/).

For stereo depth pipelines, see the [spatial detection example](https://github.com/luxonis/oak-examples/tree/main/neural-networks/object-detection/spatial-detections). Measure your model on the target camera using the [benchmarking guide](https://docs.luxonis.com/software-v3/ai-inference/benchmarking/); throughput depends on the model, input size, precision, and pipeline configuration.
