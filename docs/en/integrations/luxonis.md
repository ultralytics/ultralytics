---
comments: true
description: Convert Ultralytics YOLO models for Luxonis OAK RVC2 and RVC4 cameras with Luxonis Tools and ModelConverter, then run them on-device with DepthAI.
keywords: Luxonis, OAK camera, OAK-D, RVC2, RVC4, DepthAI, Luxonis Tools, ModelConverter, NN Archive, Luxonis Hub, on-device inference, edge AI, INT8 quantization, object detection, Ultralytics YOLO, YOLO26, YOLO11
---

# Luxonis OAK Deployment for Ultralytics YOLO

Luxonis [OAK cameras](https://www.luxonis.com/) run Ultralytics YOLO models on-device. Luxonis is not a native `model.export()` format: a `.pt` checkpoint is converted with Luxonis tooling into an [NN Archive](https://docs.luxonis.com/software-v3/ai-inference/nn-archive) for one target generation, `RVC2` or `RVC4`. Supported families and tasks are listed in the Luxonis [Tools support matrix](https://github.com/luxonis/tools#-supported-models), and [Luxonis Hub](https://docs.luxonis.com/cloud/hubai/quick-conversion) offers the same conversion as a hosted service. Install [Tools](https://github.com/luxonis/tools#-how-to-run) (Python >= 3.10) and [ModelConverter](https://github.com/luxonis/modelconverter#readme) (requires Docker), convert your checkpoint (`yolo26n.pt` below) to an `ONNX` NN Archive, then compile it for the device. `RVC4` quantizes to INT8 and requires a directory of representative calibration images, while `RVC2` ignores calibration data. Run the compiled archive with [DepthAI v3](https://docs.luxonis.com/software-v3/depthai): [`DetectionNetwork`](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/detection_network) decodes YOLO detection output on-device, and segmentation and pose outputs are decoded by the host-side parsers in [DepthAI Nodes](https://github.com/luxonis/depthai-nodes). See the Luxonis [inference docs](https://docs.luxonis.com/software-v3/ai-inference/inference/), [Model Zoo](https://models.luxonis.com/?search=YOLO), and [oak-examples](https://github.com/luxonis/oak-examples) for full pipelines.

```bash
git clone --recursive https://github.com/luxonis/tools.git && cd tools && PIP_CONSTRAINT=constraints.txt PIP_BUILD_CONSTRAINT=constraints.txt pip install . modelconv depthai
curl -LO https://github.com/ultralytics/assets/releases/latest/download/yolo26n.pt && tools yolo26n.pt --imgsz "640 640" --output-dir . # writes yolo26n_<timestamp>/yolo26n.tar.xz
modelconverter convert rvc4 --path yolo26n_*/yolo26n.tar.xz --output-dir rvc4 calibration.path calibration_images/                      # writes output/rvc4/yolo26n.rvc4.tar.xz
```

```python
import depthai as dai

with dai.Pipeline() as pipeline:
    cam = pipeline.create(dai.node.Camera).build()
    nn = pipeline.create(dai.node.DetectionNetwork).build(cam, dai.NNArchive("output/rvc4/yolo26n.rvc4.tar.xz"))
    queue = nn.out.createOutputQueue()
    pipeline.start()
    print([(d.label, d.confidence) for d in queue.get().detections])
```
