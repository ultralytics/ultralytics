---
title: Export YOLO to Alibaba MNN for Mobile
comments: true
description: Export Ultralytics YOLO26 models to Alibaba MNN with format=mnn. Covers FP16 and INT8 weights, embedded NMS and the NMS-free head, output shapes, Python and C++ inference, CPU threads, and benchmarks.
keywords: YOLO26, MNN, Alibaba MNN, MNN export, mnn file, MNNConvert, FP16, INT8 weight quantization, Android, iOS, on-device inference, mobile deployment, embedded devices, edge AI
---

# Alibaba MNN Export

[MNN](https://github.com/alibaba/MNN) is the deep learning framework from Alibaba for inference and training on phones, embedded boards, and other [edge](https://www.ultralytics.com/glossary/edge-computing) devices. Exporting an [Ultralytics YOLO26](../models/yolo26.md) model with `format="mnn"` converts it through ONNX into a single `yolo26n.mnn` file, with FP32, FP16, or INT8 weights, that the MNN runtime loads on Android, iOS, Linux, macOS, and Windows.

## At a Glance

| Property          | Details                                                                                                                                                     |
| ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format` argument | `mnn`                                                                                                                                                       |
| Output            | `yolo26n.mnn`, one file with the [metadata](#output-format) inside                                                                                          |
| Runs on           | CPUs of phones, embedded boards, and desktops, and GPUs through the Metal, OpenCL, and Vulkan backends of MNN. Ultralytics runs it on the CPU               |
| Runtime           | MNN: the `MNN` Python package, or the MNN C++ library for Android, iOS, Linux, macOS, and Windows                                                           |
| Precision         | FP32 (default), FP16 weights, and INT8 weights. YOLO26n is 9.8 MB in FP32, 5.0 MB with FP16 weights, and 2.7 MB with INT8 weights                           |
| Export host       | Linux and macOS, with PyTorch 1.10 or newer; `MNN` also has packages for Windows, which was not run. See [Quickstart](#quickstart). The export needs no GPU |

!!! tip "Export in the cloud"

    [Ultralytics Platform](../platform/train/models.md#export-model) can export a trained model to MNN, so you do not need a local export environment.

## Quickstart

Install the Ultralytics package in a virtual environment, here on Linux or macOS:

```bash
python3 -m venv mnn-env && source mnn-env/bin/activate
pip install ultralytics
```

The first export installs the [`MNN`](https://pypi.org/project/MNN/) package and the ONNX packages that the conversion uses: `onnx`, `onnxslim`, `onnxruntime`, and `protobuf`, 8 to 10 packages in the environments used for this page, none of which replaced an installed package. The export logs `Restart runtime or rerun command for updates to take effect` and still completes in the same run.

To install them before the first export, run `pip install "ultralytics[export-base]" MNN` instead; the export then installs nothing. For an editable repository install, use `pip install -e ".[export-base]" MNN`. The export has these version requirements:

| Requirement      | Version                                                                                                                                                                                                                                                        |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Python           | 3.8 or newer, as for the Ultralytics package; `MNN` 3.6.2 has packages for Python 3.8 to 3.14. Exports with FP32, FP16, and INT8 weights ran with every version from 3.9 to 3.14                                                                               |
| PyTorch          | 1.10 or newer, the version the export checks for, and 1.13 or newer for `nms=True`. Exports for this page ran with PyTorch 2.15.0, 2.14.1 on Python 3.10, and 2.8.0 on Python 3.9                                                                              |
| `MNN`            | Installed by the first export, which requires 2.9.6 or newer. All exports and inference for this page used 3.6.2, with `protobuf` 5.29.6, below the version 6 limit that the Ultralytics test environment for MNN sets, and with 6.33.6 and 7.36.2             |
| Operating system | `MNN` 3.6.2 has packages for Linux x86-64 and ARM64 (manylinux2014), macOS 11 or newer on Apple silicon and Intel, and Windows x86-64. Exports for this page ran on Ubuntu 24.04 x86-64 and on macOS 27 with an Apple M4; ARM64 Linux and Windows were not run |

For other installation methods, see the [installation guide](../quickstart.md). Then export your model and load the exported file to run inference or validate its accuracy:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a YOLO26 model
        model = YOLO("yolo26n.pt")

        # Export the model to MNN format
        model.export(format="mnn")  # creates 'yolo26n.mnn'
        ```

    === "CLI"

        ```bash
        # Export a YOLO26n PyTorch model to MNN format
        yolo export model=yolo26n.pt format=mnn # creates 'yolo26n.mnn'
        ```

On a shared AMD EPYC 9655 server, the default, FP16, and INT8 exports of YOLO26n took 0.9 to 1.3 s with `MNN` installed, and 3.6 to 7.2 s for the first export in a new environment, which also installs the packages; the FP32 export of YOLO26x took 3.9 s. The exported model runs on the export host, so you can test it there:

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported MNN model
        model = YOLO("yolo26n.mnn")

        # Run inference
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Run inference with the exported MNN model
        yolo predict model=yolo26n.mnn source='https://ultralytics.com/images/bus.jpg'
        ```

!!! example "Validate"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the exported MNN model
        model = YOLO("yolo26n.mnn")

        # Validate accuracy on the COCO8 dataset
        metrics = model.val(data="coco8.yaml")
        ```

    === "CLI"

        ```bash
        # Validate the exported MNN model
        yolo val model=yolo26n.mnn data=coco8.yaml
        ```

## Supported Tasks

MNN export supports all seven Ultralytics tasks: [detection](../tasks/detect.md), [segmentation](../tasks/segment.md), [semantic segmentation](../tasks/semantic.md), [depth estimation](../tasks/depth.md), [classification](../tasks/classify.md), [pose estimation](../tasks/pose.md), and [OBB](../tasks/obb.md). Semantic segmentation and depth estimation are available only with [YOLO26](../models/yolo26.md), the only family that ships those heads. For this page, the YOLO26n checkpoint of every task was exported with FP32, FP16, and INT8 weights and run on the export host, and so were [YOLO11](../models/yolo11.md), [YOLOv10](../models/yolov10.md), and [RT-DETR](../models/rtdetr.md) detection models; [YOLOv8](../models/yolov8.md) and [YOLOv5](../models/yolov5.md) detection models were exported and run in FP32.

## Export Arguments

| Argument   | Type             | Default | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ---------- | ---------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `format`   | `str`            | `'mnn'` | Target format for the exported model, defining compatibility with various deployment environments.                                                                                                                                                                                                                                                                                                                                                          |
| `imgsz`    | `int` or `tuple` | `640`   | Image size of the model input: an integer for square images or a tuple `(height, width)`. Fixed in the `.mnn` file unless `dynamic=True`. When not passed, the export uses the size the checkpoint was trained at. The official YOLO26 checkpoints use `640` for detect and segment, `640` for semantic models pretrained on ADE20K and `1024` for those pretrained on Cityscapes, `768` for depth, `224` for classify, `640` for pose, and `1024` for OBB. |
| `quantize` | `int` or `str`   | `None`  | Precision of the stored weights: `16` (FP16), `8` (INT8, also spelled `'int8'` or `'w8a8'`), or `32`/unset (FP32). Needs no calibration data. `'w8a16'` and `'w8a32'` raise an error, see [Precision](#precision). Replaces the deprecated `half`/`int8` flags.                                                                                                                                                                                             |
| `batch`    | `int`            | `1`     | Batch size of the exported model, fixed in the input shape unless `dynamic=True`.                                                                                                                                                                                                                                                                                                                                                                           |
| `dynamic`  | `bool`           | `False` | Lets the model take other image sizes and batch sizes at inference. Cannot be combined with `nms=True` for YOLO detection, segmentation, pose, and OBB models.                                                                                                                                                                                                                                                                                              |
| `nms`      | `bool`, optional | `None`  | `None` writes the raw output, which needs NMS for YOLO detection, segmentation, pose, and OBB models; `True` embeds NMS in YOLO detection and pose models; `False` keeps the NMS-free head of models that have one, such as YOLO26; a model without one, such as YOLO11, logs `This model has no one-to-one head; using one-to-many outputs.` and writes the raw output. See [Output Format](#output-format).                                               |
| `opset`    | `int`            | `None`  | ONNX opset of the intermediate ONNX model. If not set, the export uses opset 18 (lower with PyTorch older than 2.4), capped at the newest opset that the installed `onnx` supports. The minimum depends on the model, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).                                                                                                                                                              |
| `simplify` | `bool`           | `True`  | Simplifies the intermediate ONNX model with `onnxslim` before the conversion.                                                                                                                                                                                                                                                                                                                                                                               |
| `device`   | `str`            | `None`  | Device that runs the PyTorch model while the export traces it, such as `cpu` or `0` for the first CUDA GPU.                                                                                                                                                                                                                                                                                                                                                 |

For the arguments shared by all formats, see the [Export mode documentation](../modes/export.md).

### Precision

`quantize` selects how the weights are stored in the `.mnn` file. The activations, the values that flow between layers, are not quantized, and no calibration data is needed:

| `quantize`    | Result                                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------------------------- |
| unset or `32` | FP32 weights. YOLO26n is 9.8 MB                                                                                        |
| `16`          | FP16 weights. YOLO26n is 5.0 MB                                                                                        |
| `8`           | INT8 weights. YOLO26n is 2.7 MB                                                                                        |
| `'w8a16'`     | Fails with `quantize='w8a16' is not supported; format='mnn' supports quantize=16 (FP16), 8 (INT8), 32 or None (FP32).` |
| `'w8a32'`     | Fails with `quantize='w8a32' is not supported; format='mnn' supports quantize=16 (FP16), 8 (INT8), 32 or None (FP32).` |

- **FP16 and INT8 weights**: the converter of the `MNN` package (`MNN.tools.mnnconvert`, the Python form of `MNNConvert`) stores the convolution and matrix multiplication weights as 16-bit floats (`--fp16`) or as 8-bit integers (`--weightQuantBits 8`). The [MNN documentation](https://mnn-docs.readthedocs.io/en/latest/tools/compress.html) describes both as compression of the file: by default the runtime turns INT8 weights back into floats when it loads the model, and both run at the speed of the FP32 model. For INT8 models with 8-bit activations, calibrated on images, MNN has a separate offline quantization tool.
- **Calibration**: `data` and `fraction` are not export arguments of this format, and passing them fails with `argument 'data' is not supported for format='mnn'`.
- **Inputs and outputs**: the input and the outputs keep the types of the FP32 model in every precision (`float32`, and `uint8` for the semantic class map), so code written for the FP32 model runs the others unchanged.

!!! example "INT8 weights"

    === "Python"

        ```python
        from ultralytics import YOLO

        model = YOLO("yolo26n.pt")

        # Export with INT8 weights; use quantize=16 for FP16 weights
        model.export(format="mnn", quantize=8)  # creates 'yolo26n.mnn'

        # Load the exported model and run inference
        model = YOLO("yolo26n.mnn")
        results = model("https://ultralytics.com/images/bus.jpg")
        ```

    === "CLI"

        ```bash
        # Export with INT8 weights; use quantize=16 for FP16 weights
        yolo export model=yolo26n.pt format=mnn quantize=8 # creates 'yolo26n.mnn'

        # Run inference with the exported model
        yolo predict model=yolo26n.mnn source='https://ultralytics.com/images/bus.jpg'
        ```

The sizes of the `.mnn` file for the official YOLO26n checkpoints at their default export size, in MB (10^6 bytes). The export log prints the size in MiB, as `(9.4 MB)`, `(4.7 MB)`, and `(2.6 MB)` for the detection model:

| Task     | Default `imgsz` | FP32 | FP16 weights | INT8 weights |
| -------- | --------------- | ---- | ------------ | ------------ |
| Detect   | `640`           | 9.8  | 5.0          | 2.7          |
| Segment  | `640`           | 11.1 | 5.6          | 3.0          |
| Semantic | `1024`          | 6.3  | 3.2          | 1.7          |
| Depth    | `768`           | 20.7 | 10.4         | 5.4          |
| Classify | `224`           | 11.3 | 5.7          | 3.0          |
| Pose     | `640`           | 11.9 | 6.0          | 3.3          |
| OBB      | `1024`          | 10.1 | 5.1          | 2.9          |

## Output Format

The export writes `yolo26n.mnn` next to the `.pt` file, together with the intermediate ONNX model of the conversion, `yolo26n.onnx`, see [Limitations and Troubleshooting](#limitations-and-troubleshooting). The Ultralytics metadata (class names, task, image size, batch size, stride, and export arguments) is stored as a JSON string in the `bizCode` field of the `.mnn` file, where Ultralytics reads it when it loads the model.

The model takes one input, `images`, with shape `(batch, 3, height, width)`: RGB, channels first (NCHW), `float32`, scaled to 0-1. Ultralytics resizes each image to that size with gray padding (letterbox; classification models are resized and center-cropped instead), and the box coordinates in the output refer to the padded image. The shapes below are for the official YOLO26n checkpoints at their default export size, with the Cityscapes checkpoint `yolo26n-sem.pt` for semantic segmentation:

| Task     | Default `imgsz` | `nms=None` (default)                                                | `nms=False`                            | `nms=True`                                                 |
| -------- | --------------- | ------------------------------------------------------------------- | -------------------------------------- | ---------------------------------------------------------- |
| Detect   | `640`           | `(1, 84, 8400)`                                                     | `(1, 300, 6)`                          | `(1, 300, 6)`                                              |
| Segment  | `640`           | `(1, 116, 8400)` and `(1, 32, 160, 160)`                            | `(1, 300, 38)` and `(1, 32, 160, 160)` | Fails, see [Limitations](#limitations-and-troubleshooting) |
| Semantic | `1024`          | `(1, 1024, 1024)`, `uint8` class index for each pixel               | Same as `nms=None`                     | Same as `nms=None`                                         |
| Depth    | `768`           | `(1, 1, 768, 768)`, depth map                                       | Same as `nms=None`                     | Same as `nms=None`                                         |
| Classify | `224`           | `(1, 1000)`, class probabilities                                    | Same as `nms=None`                     | Fails, see [Limitations](#limitations-and-troubleshooting) |
| Pose     | `640`           | `(1, 56, 8400)`: 4 box values, 1 class score, then 17 keypoints x 3 | `(1, 300, 57)`                         | `(1, 300, 57)`                                             |
| OBB      | `1024`          | `(1, 20, 21504)`: 4 box values, 15 class scores, then the angle     | `(1, 300, 7)`                          | Fails, see [Limitations](#limitations-and-troubleshooting) |

- **Raw output (`nms=None`)**: detection, segmentation, pose, and OBB models write shape `(batch, values, anchors)`. For each anchor: the box center x, center y, width, and height in input pixels, then the class scores, then the values of the task (32 mask coefficients, 17 x 3 keypoint values, or the OBB angle in radians). Apply NMS to this output yourself. See the [ONNX output format](onnx.md#output-format) for the layout of each task.
- **NMS-free head (`nms=False`)**: YOLO26 detection, segmentation, pose, and OBB models write up to 300 detections per image, each with the box corners x1, y1, x2, y2 in input pixels, the score, the class index, and then the values of the task, so no NMS is needed. OBB rows hold the rotated box center x, center y, width, and height, the score, the class index, and the angle in radians. Keep the rows whose score is above your confidence threshold.
- **Embedded NMS (`nms=True`)**: YOLO detection and pose models write the same layout after NMS inside the model, with the default `conf=0.25` of the export. Segmentation, OBB, and classification models fail with this setting, see [Limitations and Troubleshooting](#limitations-and-troubleshooting), and semantic segmentation and depth models log a warning such as `'nms=True' is not valid for semantic models. Forcing 'nms=None'.` and export as with `nms=None`.
- **Semantic segmentation and depth**: the semantic output is a class map of `uint8` class indices for each pixel, as in the [ONNX export](onnx.md#output-format), and the depth output a `float32` depth map. Both are the same for every value of `nms`.
- **RT-DETR**: the output of RT-DETR-L has shape `(1, 300, 6)`, which needs no NMS. `nms=True` logs `'nms=True' is not available for end2end models or non-YOLO heads. Forcing 'nms=None'.` and exports the same output.

The table below compares the raw outputs (`nms=None`) on the x86-64 export host with those of the [ONNX](onnx.md) export of the same checkpoint on one image, `bus.jpg`, and for OBB on the aerial image `boats.jpg`. The FP32 column is the largest absolute difference of any output value, where box values are in pixels; the FP16 and INT8 columns are the correlation of the output with the ONNX output:

| Task              | FP32 against ONNX         | FP16 weights against ONNX         | INT8 weights against ONNX         |
| ----------------- | ------------------------- | --------------------------------- | --------------------------------- |
| Detect            | 0.0055                    | 1.0000                            | 0.9999                            |
| Segment           | 0.0045                    | 1.0000, prototypes 1.0000         | 0.9999, prototypes 0.9978         |
| Semantic          | Same class on every pixel | Same class on 99.9% of the pixels | Same class on 99.5% of the pixels |
| Depth             | 0.0001                    | 1.0000                            | 1.0000                            |
| Classify          | below 0.0001              | 1.0000, same top-1 class          | 0.9909, same top-1 class          |
| Pose              | 0.0029                    | 1.0000                            | 1.0000                            |
| OBB (`boats.jpg`) | 0.0102                    | 1.0000                            | 0.9999                            |

On the same images, the detections that the FP32 `nms=False` and `nms=True` exports kept above a score of 0.25 matched those of the FP32 ONNX export with the same setting, with the same classes and values within 0.001. FP16 and INT8 weights change the outputs a little; the detection, segmentation, and pose models found the same objects on `bus.jpg` with every precision. Validate a model on your own data before you deploy it, see [Benchmarks](#benchmarks) for the accuracy on COCO.

`YOLO("yolo26n.mnn")` reads the layout from the metadata, so you only need these details when you run the `.mnn` file in your own code.

### MNN-Only Inference

The following examples run YOLO26 preprocessing, inference, and postprocessing with MNN alone, in both Python and C++, so you can deploy without PyTorch or the Ultralytics package. They read the raw output of an export with `nms` unset, `(1, 84, 8400)` for YOLO26n, assume a 640x640 input and the 80 COCO classes, and apply NMS with MNN; the NMS keeps at most 100 boxes, with the IoU threshold of 0.5 and the score threshold of 0.25 that Ultralytics predict uses by default, and treats all classes together. Ultralytics predict keeps up to 300 boxes and applies NMS to each class separately, so the results can differ. To run the Python example, save it as `main.py` and run `python main.py --model yolo26n.mnn --img bus.jpg`; it ran for this page on the export host with `MNN` 3.6.2 and found the same 5 objects in `bus.jpg` as `YOLO("yolo26n.mnn")`. The C++ example was not built.

!!! example "MNN"

    === "Python"

        ```python
        import argparse

        import MNN
        import MNN.cv as cv2
        import MNN.numpy as np


        def inference(model, img, precision, backend, thread):
            config = {}
            config["precision"] = precision
            config["backend"] = backend
            config["numThread"] = thread
            rt = MNN.nn.create_runtime_manager((config,))
            # net = MNN.nn.load_module_from_file(model, ['images'], ['output0'], runtime_manager=rt)
            net = MNN.nn.load_module_from_file(model, [], [], runtime_manager=rt)
            original_image = cv2.imread(img)
            ih, iw, _ = original_image.shape
            length = max((ih, iw))
            scale = length / 640
            image = np.pad(original_image, [[0, length - ih], [0, length - iw], [0, 0]], "constant")
            image = cv2.resize(
                image, (640, 640), 0.0, 0.0, cv2.INTER_LINEAR, -1, [0.0, 0.0, 0.0], [1.0 / 255.0, 1.0 / 255.0, 1.0 / 255.0]
            )
            image = image[..., ::-1]  # BGR to RGB
            input_var = image[None]
            input_var = MNN.expr.convert(input_var, MNN.expr.NC4HW4)
            output_var = net.forward(input_var)
            output_var = MNN.expr.convert(output_var, MNN.expr.NCHW)
            output_var = output_var.squeeze()
            # output_var shape: [84, 8400]; 84 means: [cx, cy, w, h, prob * 80]
            cx = output_var[0]
            cy = output_var[1]
            w = output_var[2]
            h = output_var[3]
            probs = output_var[4:]
            # [cx, cy, w, h] -> [x0, y0, x1, y1]
            x0 = cx - w * 0.5
            y0 = cy - h * 0.5
            x1 = cx + w * 0.5
            y1 = cy + h * 0.5
            boxes = np.stack([x0, y0, x1, y1], axis=1)
            # get max prob and idx
            scores = np.max(probs, 0)
            class_ids = np.argmax(probs, 0)
            result_ids = MNN.expr.nms(boxes, scores, 100, 0.5, 0.25)
            print(result_ids.shape)
            # nms result box, score, ids
            result_boxes = boxes[result_ids]
            result_scores = scores[result_ids]
            result_class_ids = class_ids[result_ids]
            for i in range(len(result_boxes)):
                x0, y0, x1, y1 = result_boxes[i].read_as_tuple()
                y0 = int(y0 * scale)
                y1 = int(y1 * scale)
                x0 = int(x0 * scale)
                x1 = int(x1 * scale)
                # clamp to the original image size to handle cases where padding was applied
                x1 = min(iw, x1)
                y1 = min(ih, y1)
                print(result_class_ids[i])
                cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.imwrite("res.jpg", original_image)


        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", type=str, required=True, help="the yolo26 model path")
            parser.add_argument("--img", type=str, required=True, help="the input image path")
            parser.add_argument("--precision", type=str, default="normal", help="inference precision: normal, low, high, lowBF")
            parser.add_argument(
                "--backend",
                type=str,
                default="CPU",
                help="inference backend: CPU, OPENCL, OPENGL, NN, VULKAN, METAL, TRT, CUDA, HIAI",
            )
            parser.add_argument("--thread", type=int, default=4, help="inference using thread: int")
            args = parser.parse_args()
            inference(args.model, args.img, args.precision, args.backend, args.thread)
        ```

    === "CPP"

        ```cpp
        #include <stdio.h>
        #include <MNN/ImageProcess.hpp>
        #include <MNN/expr/Module.hpp>
        #include <MNN/expr/Executor.hpp>
        #include <MNN/expr/ExprCreator.hpp>
        #include <MNN/expr/Executor.hpp>

        #include <cv/cv.hpp>

        using namespace MNN;
        using namespace MNN::Express;
        using namespace MNN::CV;

        int main(int argc, const char* argv[]) {
            if (argc < 3) {
                MNN_PRINT("Usage: ./yolo26_demo.out model.mnn input.jpg [forwardType] [precision] [thread]\n");
                return 0;
            }
            int thread = 4;
            int precision = 0;
            int forwardType = MNN_FORWARD_CPU;
            if (argc >= 4) {
                forwardType = atoi(argv[3]);
            }
            if (argc >= 5) {
                precision = atoi(argv[4]);
            }
            if (argc >= 6) {
                thread = atoi(argv[5]);
            }
            MNN::ScheduleConfig sConfig;
            sConfig.type = static_cast<MNNForwardType>(forwardType);
            sConfig.numThread = thread;
            BackendConfig bConfig;
            bConfig.precision = static_cast<BackendConfig::PrecisionMode>(precision);
            sConfig.backendConfig = &bConfig;
            std::shared_ptr<Executor::RuntimeManager> rtmgr = std::shared_ptr<Executor::RuntimeManager>(Executor::RuntimeManager::createRuntimeManager(sConfig));
            if(rtmgr == nullptr) {
                MNN_ERROR("Empty RuntimeManger\n");
                return 0;
            }
            rtmgr->setCache(".cachefile");

            std::shared_ptr<Module> net(Module::load(std::vector<std::string>{}, std::vector<std::string>{}, argv[1], rtmgr));
            auto original_image = imread(argv[2]);
            auto dims = original_image->getInfo()->dim;
            int ih = dims[0];
            int iw = dims[1];
            int len = ih > iw ? ih : iw;
            float scale = len / 640.0;
            std::vector<int> padvals { 0, len - ih, 0, len - iw, 0, 0 };
            auto pads = _Const(static_cast<void*>(padvals.data()), {3, 2}, NCHW, halide_type_of<int>());
            auto image = _Pad(original_image, pads, CONSTANT);
            image = resize(image, Size(640, 640), 0, 0, INTER_LINEAR, -1, {0., 0., 0.}, {1./255., 1./255., 1./255.});
            image = cvtColor(image, COLOR_BGR2RGB);
            auto input = _Unsqueeze(image, {0});
            input = _Convert(input, NC4HW4);
            auto outputs = net->onForward({input});
            auto output = _Convert(outputs[0], NCHW);
            output = _Squeeze(output);
            // output shape: [84, 8400]; 84 means: [cx, cy, w, h, prob * 80]
            auto cx = _Gather(output, _Scalar<int>(0));
            auto cy = _Gather(output, _Scalar<int>(1));
            auto w = _Gather(output, _Scalar<int>(2));
            auto h = _Gather(output, _Scalar<int>(3));
            std::vector<int> startvals { 4, 0 };
            auto start = _Const(static_cast<void*>(startvals.data()), {2}, NCHW, halide_type_of<int>());
            std::vector<int> sizevals { -1, -1 };
            auto size = _Const(static_cast<void*>(sizevals.data()), {2}, NCHW, halide_type_of<int>());
            auto probs = _Slice(output, start, size);
            // [cx, cy, w, h] -> [x0, y0, x1, y1]
            auto x0 = cx - w * _Const(0.5);
            auto y0 = cy - h * _Const(0.5);
            auto x1 = cx + w * _Const(0.5);
            auto y1 = cy + h * _Const(0.5);
            auto boxes = _Stack({x0, y0, x1, y1}, 1);
            auto scores = _ReduceMax(probs, {0});
            auto ids = _ArgMax(probs, 0);
            auto result_ids = _Nms(boxes, scores, 100, 0.5, 0.25);
            auto result_ptr = result_ids->readMap<int>();
            auto box_ptr = boxes->readMap<float>();
            auto ids_ptr = ids->readMap<int>();
            auto score_ptr = scores->readMap<float>();
            for (int i = 0; i < 100; i++) {
                auto idx = result_ptr[i];
                if (idx < 0) break;
                auto x0 = box_ptr[idx * 4 + 0] * scale;
                auto y0 = box_ptr[idx * 4 + 1] * scale;
                auto x1 = box_ptr[idx * 4 + 2] * scale;
                auto y1 = box_ptr[idx * 4 + 3] * scale;
                // clamp to the original image size to handle cases where padding was applied
                x1 = std::min(static_cast<float>(iw), x1);
                y1 = std::min(static_cast<float>(ih), y1);
                auto class_idx = ids_ptr[idx];
                auto score = score_ptr[idx];
                rectangle(original_image, {x0, y0}, {x1, y1}, {0, 0, 255}, 2);
            }
            if (imwrite("res.jpg", original_image)) {
                MNN_PRINT("result image write to `res.jpg`.\n");
            }
            rtmgr->updateCache();
            return 0;
        }
        ```

## Deploy

`YOLO("yolo26n.mnn")` runs the model with the `MNN` Python package on the CPU. This works on the export host too, so you can check a model before you copy it to a device:

- **Precision `low`**: Ultralytics loads every model with the MNN precision `low`. On ARMv8.2 and newer ARM processors, MNN then computes in FP16, according to its documentation. On the x86-64 export host, `low`, `normal`, and `high` gave identical outputs for the FP32, FP16, and INT8 files, so the x86 results on this page were computed in FP32. On Apple silicon, MNN reports FP16 support (`fp16:1`) in its log, and `low` changes the outputs: on an Apple M5 Pro, the raw FP32 YOLO26n output on `bus.jpg` differed from ONNX by up to 4.2 with `low` and 0.03 with `normal`, and `YOLO("yolo26n.mnn")` found the same 5 objects.
- **CPU threads**: Ultralytics asks MNN for half as many threads as `os.cpu_count()` reports, and on Linux for no more than the CPUs the process may use: those that `taskset` or Docker `--cpuset-cpus` allow, and the cgroup v2 CPU quota of its cgroup or a parent cgroup, such as Docker `--cpus` or a Kubernetes CPU limit. On the Linux server with 384 logical CPUs, that gave MNN 192 threads with all CPUs, 16, 8, 4, and 1 when `taskset` limited the process to as many CPUs, and 4 with Docker `--cpus 4`. In two runs, the median of 20 FP32 YOLO26n forwards through the Ultralytics backend on a random 640x640 input took 22 to 24 ms with all CPUs, 21 to 22 ms limited to 16 CPUs on two of the 8-core CCDs of the processor with `taskset -c 40-55`, 10 ms with the 8 CPUs of one CCD, 17 ms with 4, and 53 ms with 1. On a machine with many cores, limit the CPUs of the process, on Linux with `taskset` or Docker `--cpuset-cpus`, and compare.
- **`device`**: the model runs on the CPU. On Apple silicon, `device="mps"` gave the same detections on one image as `device="cpu"`; on a CUDA host, `device=0` was not run for this page.
- **Fixed input shape**: unless the model was exported with `dynamic=True`, it runs at its export `imgsz` and `batch`, which Ultralytics reads from the metadata. An `imgsz` passed to `predict` or `val` is ignored: YOLO26n exported at 640 gave the same detections with `imgsz=320` and `imgsz=1280`. A list of three images ran on a model exported with `batch=1`, and models exported with `batch=2` and `batch=4` reached the same accuracy on `coco8.yaml` as with `batch=1`.
- **`dynamic=True`**: the model runs at the `imgsz` passed to `predict` or `val`; YOLO26n ran at 320, 640, 1280, and 480x640, and `val` ran with `batch=4`.
- **File name**: Ultralytics recognizes the format by `.mnn` in the file name and reads the metadata only from a file whose name ends in `.mnn`, see [Limitations and Troubleshooting](#limitations-and-troubleshooting).
- **Several models**: a detection and a segmentation model ran alternately in one process.

### Mobile and Embedded Devices

In an Android, iOS, or HarmonyOS application, or on an embedded Linux board, build the MNN library for the target, load the `.mnn` file with it, and pass it an image preprocessed as described in [Output Format](#output-format). The C++ example above shows the calls, and the backend, such as CPU, Metal, OpenCL, or Vulkan, is chosen when the model is loaded, not at export. According to the MNN documentation, MNN supports iOS 8.0 and newer and Android 4.3 and newer, and with precision `low` the CPU computes in FP16 on ARMv8.2 and newer processors when MNN is built with the `MNN_ARM82` option. These devices were not run for this page.

- **[Build MNN](https://mnn-docs.readthedocs.io/en/latest/compile/engine.html)**: build instructions for Linux, macOS, Windows, Android, iOS, and HarmonyOS.
- **[MNN Android project](https://github.com/alibaba/MNN/tree/master/project/android)**, **[MNN iOS project](https://github.com/alibaba/MNN/tree/master/project/ios)**, and **[MNN HarmonyOS project](https://github.com/alibaba/MNN/tree/master/project/harmony)**: build scripts and demo applications in the MNN repository.
- **[MNN Python API](https://mnn-docs.readthedocs.io/en/latest/start/python.html)**: the runtime configuration of the `MNN` package, including precision and threads.
- **[Raspberry Pi](../guides/raspberry-pi.md)** and **[NVIDIA Jetson](../guides/nvidia-jetson.md)**: set up Ultralytics on these boards.
- **[Ultralytics and MNN](https://www.ultralytics.com/blog/seamlessly-deploy-ultralytics-yolo11-using-the-mnn-integration)**: an overview of the integration on the Ultralytics blog.
- **[Model deployment options](../guides/model-deployment-options.md)**: compare MNN with the other export formats.

## Limitations and Troubleshooting

- **Automatic installation turned off**: with `YOLO_AUTOINSTALL=false` and a plain `pip install ultralytics`, the export logs `Ultralytics requirements ['onnx>=1.12.0,<2.0.0', 'onnxruntime', 'onnxslim>=0.1.82'] not found, AutoUpdate disabled by YOLO_AUTOINSTALL=False` and fails with `No module named 'onnx'`. Install the packages first with `pip install "ultralytics[export-base]" MNN`; the export then needs no installation.
- **PyTorch older than 1.10**: the export fails with `MNN export requires torch>=1.10.0 to avoid segmentation faults`.
- **PyTorch older than 1.13 with `nms=True`**: the Ultralytics code stops the ONNX step of the export with `'nms=True' ONNX export requires torch>=1.13 (found torch==...)`; this was not run.
- **Embedded NMS for other tasks**: `nms=True` works for YOLO detection and pose models only. Segmentation and OBB models fail with `Alibaba MNN export with 'nms=True' only supports detect and pose models.`, and classification models with `'nms=True' is not valid for classification models.`. For YOLO26 segmentation and OBB models, `nms=False` gives an output that needs no NMS, see [Output Format](#output-format).
- **`dynamic=True` with `nms=True`**: for YOLO detection, segmentation, pose, and OBB models, the export fails with `Alibaba MNN export does not support combining 'dynamic=True' with 'nms=True'.` Use one of the two.
- **Intermediate ONNX file**: every export leaves `yolo26n.onnx` next to `yolo26n.mnn`, and replaces an existing `yolo26n.onnx`, such as one from an earlier ONNX export with other settings. The `.mnn` file does not need it, and you can delete it.
- **Output file**: every export of one checkpoint writes the same `yolo26n.mnn`, whatever its `imgsz`, `batch`, or `quantize`, and replaces an existing file. Rename the file before you export a second variant, and keep `.mnn` at the end of the name, for example `yolo26n_int8.mnn`: a copy named `model.mnn.bak` loaded with the class names `class0`, `class1`, and so on, and one named `detector.bin` failed with `is not a supported model format`.
- **Minimum opset**: the export fails with `ONNX export requires opset>=9 for this model, but got opset=8` when `opset` is too low. The minimum is `7` for classification, `9` for detection, segmentation, semantic segmentation, pose, and OBB, `11` for depth estimation and for models exported with `nms=True`, and `16` for RT-DETR. YOLO26n detection exported with opset 12 and the default 18, and the classification model with opset 8.
- **Converter messages**: the export log contains lines from MNN such as `Check it out ==> /model.11/Resize_output_0 has empty input, the index is 1`, and on Linux also `Can't open file:/sys/devices/system/cpu/cpufreq/schedutil/affected_cpus` and `CPU Group: [ ... ]`. The exports for this page that logged them completed and gave the same outputs as ONNX.
- **Checkpoints trained with `quantize=8`**: a checkpoint fine-tuned with [quantization-aware training](../modes/export.md#quantization-aware-training) fails with `format='mnn' cannot export a QAT model: its Q/DQ ranges are only read by the 'onnx' and 'engine' backends. Export a non-QAT checkpoint to this format instead.` Export a checkpoint trained without `quantize=8`, or export the QAT checkpoint to [ONNX](onnx.md) or [TensorRT](tensorrt.md).
- **Windows and ARM64 Linux**: `MNN` has packages for both, but exports there were not run for this page.

For other errors, see the [Common Issues guide](../guides/yolo-common-issues.md).

## Benchmarks

MNN models were not benchmarked on a phone for this page. The [Raspberry Pi guide](../guides/raspberry-pi.md#raspberry-pi-5-yolo26-benchmarks) and the [NVIDIA Jetson guide](../guides/nvidia-jetson.md#nvidia-jetson-yolo26-benchmarks) compare MNN with the other export formats on those ARM boards.

### FP32, FP16, and INT8 Weights on a Server CPU

The tables below compare PyTorch, the ONNX export, and MNN models with FP32, FP16, and INT8 weights of every YOLO26 detection model on the CPU of the export host, an **AMD EPYC 9655** server shared with other jobs. For the latency table, every process was limited to the 8 CPUs of one 8-core CCD of the processor with `taskset -c 40-47`, where PyTorch, ONNX Runtime, and MNN each ran on the 8 threads that Ultralytics sets. The results apply to this x86 setup, where MNN computes in FP32 for every precision, see [Deploy](#deploy); on ARM devices the ranking can differ.

=== "Latency"

    Milliseconds per image, lower is better, as the mean of two runs. The fastest format for each model is in bold.

    | Model   | `imgsz` | PyTorch  | ONNX FP32 | MNN FP32 | MNN FP16 weights | MNN INT8 weights |
    | ------- | ------- | -------- | --------- | -------- | ---------------- | ---------------- |
    | YOLO26n | 640     | **10.0** | 16.6      | 16.2     | 16.0             | 15.9             |
    | YOLO26s | 640     | **25.5** | 31.3      | 34.3     | 34.2             | 34.0             |
    | YOLO26m | 640     | 65.1     | **57.9**  | 81.8     | 81.5             | 81.6             |
    | YOLO26l | 640     | 74.1     | **70.7**  | 98.9     | 100.3            | 102.5            |
    | YOLO26x | 640     | 144.4    | **130.4** | 204.8    | 206.6            | 203.1            |

=== "Accuracy"

    mAP<sup>val</sup> 50-95 on 500 images of COCO val2017, higher is better.

    | Model   | PyTorch | ONNX FP32 | MNN FP32 | MNN FP16 weights | MNN INT8 weights |
    | ------- | ------- | --------- | -------- | ---------------- | ---------------- |
    | YOLO26n | 42.0    | 42.0      | 42.0     | 42.0             | 41.7             |
    | YOLO26s | 49.4    | 49.4      | 49.4     | 49.5             | 49.0             |
    | YOLO26m | 53.4    | 53.4      | 53.4     | 53.5             | 53.2             |
    | YOLO26l | 55.4    | 55.4      | 55.4     | 55.4             | 55.1             |
    | YOLO26x | 55.6    | 55.6      | 55.6     | 55.6             | 55.6             |

=== "Size"

    Size of the `.pt` checkpoint, the ONNX file, and the `.mnn` file in MB (10^6 bytes). The smallest file for each model is in bold.

    | Model   | PyTorch | ONNX FP32 | MNN FP32 | MNN FP16 weights | MNN INT8 weights |
    | ------- | ------- | --------- | -------- | ---------------- | ---------------- |
    | YOLO26n | 5.5     | 9.9       | 9.8      | 5.0              | **2.7**          |
    | YOLO26s | 20.4    | 38.3      | 38.2     | 19.2             | **9.9**          |
    | YOLO26m | 44.3    | 82.0      | 81.8     | 41.0             | **20.9**         |
    | YOLO26l | 53.2    | 99.6      | 99.4     | 49.8             | **25.4**         |
    | YOLO26x | 118.7   | 223.3     | 223.1    | 111.7            | **56.5**         |

- **MNN FP32** reproduced the accuracy of the ONNX export for every model. Through Ultralytics, it took 1.3 to 1.6 times as long as PyTorch on this CPU, about as long as ONNX for YOLO26n, and 1.1 to 1.6 times as long as ONNX for the larger models.
- **FP16 and INT8 weights** made the file 2.0 and 3.6 to 3.9 times smaller than FP32. The accuracy changed by 0.0 to +0.1 mAP with FP16 weights and by -0.4 to 0.0 with INT8 weights, and the latency stayed within 4% of FP32, because the runtime computes with float weights in both cases.
- **Threads**: without `taskset`, on all 384 logical CPUs, Ultralytics gave MNN 192 threads, and YOLO26n took 24.3 ms in MNN FP32, against 14.7 ms in PyTorch and 14.6 ms in ONNX, which stayed at 8 threads, as the mean of two runs. Run directly with the `MNN` package in a process limited to 8 CPUs, one YOLO26m FP32 forward took 76 ms with `numThread` 8 and 96 ms with 192 (YOLO26n: 10 against 15 to 16 ms).

Measured with `ultralytics` 8.4.218 plus the MNN thread limit described in [Deploy](#deploy), `MNN` 3.6.2, PyTorch 2.15.0, and ONNX Runtime 1.32.0 at `batch=1`. PyTorch ran with `rect=False`, at the same 640x640 input as the exported models. Accuracy is on the last 500 images of COCO val2017 in sorted order, so it differs from the values for the full validation set on the [YOLO26 model page](../models/yolo26.md). Latency is the inference stage that `val` reports and does not include preprocessing or postprocessing; the two runs differed by up to 26%, and runs that overlapped a burst of other jobs on the server were repeated.

## About MNN

<p align="center">
  <img width="100%" src="https://mnn-docs.readthedocs.io/en/latest/_images/architecture.png" alt="MNN mobile neural network inference framework">
</p>

[MNN](https://github.com/alibaba/MNN) is an open-source deep learning framework from Alibaba, released under the Apache 2.0 license, for inference and training on devices. Its converter turns TensorFlow, Caffe, ONNX, and TorchScript models into `.mnn` files, and Alibaba uses it in more than 30 of its apps, such as Taobao, Tmall, Youku, DingTalk, and Xianyu.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_fsABGtSbqo"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Speed Up Ultralytics YOLO26 Inference with MNN Export | PyTorch vs MNN 🚀
</p>

## FAQ

### Does an MNN model from Ultralytics use the GPU or NPU of a phone?

The `.mnn` file does not fix the backend: the application chooses it when it loads the model, see the `backend` argument of the [MNN-only example](#mnn-only-inference). According to its documentation, MNN has Metal, OpenCL, and Vulkan backends for GPUs and CoreML, NNAPI, HIAI, and QNN backends for NPUs. `YOLO("yolo26n.mnn")` always uses the CPU, and the GPU and NPU backends were not run for this page. To use the Neural Engine of Apple devices or Snapdragon NPUs with an Ultralytics export, see [Core ML](coreml.md) and [Qualcomm QNN](qnn.md).

### Should I export an MNN model with FP16 or INT8 weights?

Both make the file smaller and need no calibration data: YOLO26n is 9.8 MB in FP32, 5.0 MB with FP16 weights, and 2.7 MB with INT8 weights. In the [benchmarks](#benchmarks), FP16 weights changed the accuracy by 0.0 to +0.1 mAP and INT8 weights by -0.4 to 0.0. Neither makes inference faster by itself, because MNN turns the weights back into floats. According to the MNN documentation, the speed comes from the runtime instead: precision `low` computes in FP16 on ARM CPUs that support it, with any of the three files, and INT8 weights are computed in INT8 only when the application enables the dynamic quantization of MNN, which Ultralytics does not.

### How do I run a YOLO26 MNN model in an Android or iOS app?

Build the MNN library for the platform, load the `.mnn` file in the application, and preprocess each image and read the output as described in [Output Format](#output-format); export with `nms=False` to get an output that needs no NMS. The MNN repository has Android and iOS projects to start from, see [Mobile and Embedded Devices](#mobile-and-embedded-devices). This was not run on a phone for this page.

### Why is my MNN model slower than PyTorch or ONNX on a server CPU?

On the x86 server used for this page, MNN FP32 through Ultralytics took 1.3 to 1.6 times as long as PyTorch with 8 CPUs, and up to 1.6 times as long as ONNX, see [Benchmarks](#benchmarks). More threads did not help: without a CPU limit, Ultralytics gave MNN 192 threads, half of the logical CPUs, and YOLO26n took 24.3 ms instead of 16.2 ms on 8 CPUs, see [Deploy](#deploy). In your own MNN code, set `numThread` to the number of CPUs the process can use: limited to 8 CPUs, a raw MNN forward of YOLO26m took 76 ms with `numThread` 8 and 96 ms with 192. MNN also targets phones and embedded boards: on this x86 CPU the precisions `low`, `normal`, and `high` gave identical outputs, computed in FP32. For x86 servers, see [ONNX](onnx.md) or [OpenVINO](openvino.md).
