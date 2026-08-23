---
description: Convert Ultralytics YOLO models for Luxonis OAK RVC2 and RVC4 cameras with Luxonis Tools and ModelConverter, then run them on-device with DepthAI.
---

# Luxonis OAK Deployment for Ultralytics YOLO

Luxonis [OAK cameras](https://www.luxonis.com/) run Ultralytics YOLO models on-device. Luxonis is not a native `model.export()` format: a `.pt` checkpoint is converted with Luxonis tooling into an [NN Archive](https://docs.luxonis.com/software-v3/ai-inference/nn-archive) for one target generation, `RVC2` or `RVC4`. Supported families and tasks are listed in the Luxonis [Tools support matrix](https://github.com/luxonis/tools#-supported-models), and [Luxonis Hub](https://docs.luxonis.com/cloud/hubai/quick-conversion) offers the same conversion as a hosted service. Install [Tools](https://github.com/luxonis/tools#-how-to-run) (Python >= 3.10) and [ModelConverter](https://github.com/luxonis/modelconverter#readme) (requires Docker), convert the checkpoint to an `ONNX` NN Archive, then compile it for the device. `RVC4` conversion quantizes to INT8 and needs a directory of representative calibration images; use `rvc2` for `RVC2` devices, which take no calibration data.

```bash
git clone --recursive https://github.com/luxonis/tools.git && cd tools
PIP_CONSTRAINT=constraints.txt PIP_BUILD_CONSTRAINT=constraints.txt pip install . modelconv depthai
tools yolo26n.pt --imgsz "640 640"
modelconverter convert rvc4 --path yolo26n.tar.xz calibration.path calibration_images/
```

Run the compiled archive with [DepthAI v3](https://docs.luxonis.com/software-v3/depthai). Detection, segmentation, and pose models are decoded on-device by [`DetectionNetwork`](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/detection_network); see the Luxonis [inference docs](https://docs.luxonis.com/software-v3/ai-inference/inference/), [Model Zoo](https://models.luxonis.com/?search=YOLO), and [oak-examples](https://github.com/luxonis/oak-examples) for other tasks and full pipelines.

```python
import depthai as dai

with dai.Pipeline() as pipeline:
    camera = pipeline.create(dai.node.Camera).build()
    nn = pipeline.create(dai.node.DetectionNetwork).build(camera, dai.NNArchive("output/yolo26n.tar.xz"))
    queue = nn.out.createOutputQueue()
    pipeline.start()
    print([(d.label, d.confidence) for d in queue.get().detections])
```
