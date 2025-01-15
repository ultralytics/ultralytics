---
comments: true
description: Optimize YOLO11 models for mobile and embedded devices by exporting to MNN format.
keywords: Ultralytics, YOLO11, MNN, model export, machine learning, deployment, mobile, embedded systems, deep learning, AI models
---

# MNN Export for YOLO11 Models and Deploy

## MNN

<p align="center">
  <img width="100%" src="https://mnn-docs.readthedocs.io/en/latest/_images/architecture.png" alt="MNN architecture">
</p>

[MNN](https://github.com/alibaba/MNN) is a highly efficient and lightweight deep learning framework. It supports inference and training of deep learning models and has industry-leading performance for inference and training on-device. At present, MNN has been integrated into more than 30 apps of Alibaba Inc, such as Taobao, Tmall, Youku, DingTalk, Xianyu, etc., covering more than 70 usage scenarios such as live broadcast, short video capture, search recommendation, product searching by image, interactive marketing, equity distribution, security risk control. In addition, MNN is also used on embedded devices, such as IoT.

## Export to MNN: Converting Your YOLO11 Model

You can expand model compatibility and deployment flexibility by converting YOLO11 models to MNN format.

### Installation

To install the required packages, run:

!!! tip "Installation"

    === "CLI"

        ```bash
        # Install the required package for YOLO11 and MNN
        pip install ultralytics
        pip install MNN
        ```

### Usage

Before diving into the usage instructions, it's important to note that while all [Ultralytics YOLO11 models](../models/index.md) are available for exporting, you can ensure that the model you select supports export functionality [here](../modes/export.md).

!!! example "Usage"

    === "Python"

          ```python
          from ultralytics import YOLO

          # Load the YOLO11 model
          model = YOLO("yolo11n.pt")

          # Export the model to MNN format
          model.export(format="mnn")  # creates 'yolo11n.mnn'

          # Load the exported MNN model
          mnn_model = YOLO("yolo11n.mnn")

          # Run inference
          results = mnn_model("https://ultralytics.com/images/bus.jpg")
          ```

    === "CLI"

          ```bash
          # Export a YOLO11n PyTorch model to MNN format
          yolo export model=yolo11n.pt format=mnn  # creates 'yolo11n.mnn'

          # Run inference with the exported model
          yolo predict model='yolo11n.mnn' source='https://ultralytics.com/images/bus.jpg'
          ```

For more details about supported export options, visit the [Ultralytics documentation page on deployment options](../guides/model-deployment-options.md).

### MNN-Only Inference

A function that relies solely on MNN for YOLO11 inference and preprocessing is implemented, providing both Python and C++ versions for easy deployment in any scenario.

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
            input_var = np.expand_dims(image, 0)
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
            # [cx, cy, w, h] -> [y0, x0, y1, x1]
            x0 = cx - w * 0.5
            y0 = cy - h * 0.5
            x1 = cx + w * 0.5
            y1 = cy + h * 0.5
            boxes = np.stack([x0, y0, x1, y1], axis=1)
            # get max prob and idx
            scores = np.max(probs, 0)
            class_ids = np.argmax(probs, 0)
            result_ids = MNN.expr.nms(boxes, scores, 100, 0.45, 0.25)
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
                print(result_class_ids[i])
                cv2.rectangle(original_image, (x0, y0), (x1, y1), (0, 0, 255), 2)
            cv2.imwrite("res.jpg", original_image)


        if __name__ == "__main__":
            parser = argparse.ArgumentParser()
            parser.add_argument("--model", type=str, required=True, help="the yolo11 model path")
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
                MNN_PRINT("Usage: ./yolo11_demo.out model.mnn input.jpg [forwardType] [precision] [thread]\n");
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
            // [cx, cy, w, h] -> [y0, x0, y1, x1]
            auto x0 = cx - w * _Const(0.5);
            auto y0 = cy - h * _Const(0.5);
            auto x1 = cx + w * _Const(0.5);
            auto y1 = cy + h * _Const(0.5);
            auto boxes = _Stack({x0, y0, x1, y1}, 1);
            auto scores = _ReduceMax(probs, {0});
            auto ids = _ArgMax(probs, 0);
            auto result_ids = _Nms(boxes, scores, 100, 0.45, 0.25);
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

## Summary

In this guide, we introduce how to export the Ultralytics YOLO11 model to MNN and use MNN for inference.

For more usage, please refer to the [MNN documentation](https://mnn-docs.readthedocs.io/en/latest).

## FAQ

### How do I export Ultralytics YOLO11 models to MNN format?

To export your Ultralytics YOLO11 model to MNN format, follow these steps:

!!! example "Export"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 model
        model = YOLO("yolo11n.pt")

        # Export to MNN format
        model.export(format="mnn")  # creates 'yolo11n.mnn' with fp32 weight
        model.export(format="mnn", half=True)  # creates 'yolo11n.mnn' with fp16 weight
        model.export(format="mnn", int8=True)  # creates 'yolo11n.mnn' with int8 weight
        ```

    === "CLI"

        ```bash
        yolo export model=yolo11n.pt format=mnn            # creates 'yolo11n.mnn' with fp32 weight
        yolo export model=yolo11n.pt format=mnn half=True  # creates 'yolo11n.mnn' with fp16 weight
        yolo export model=yolo11n.pt format=mnn int8=True  # creates 'yolo11n.mnn' with int8 weight
        ```

For detailed export options, check the [Export](../modes/export.md) page in the documentation.

### How do I predict with an exported YOLO11 MNN model?

To predict with an exported YOLO11 MNN model, use the `predict` function from the YOLO class.

!!! example "Predict"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load the YOLO11 MNN model
        model = YOLO("yolo11n.mnn")

        # Export to MNN format
        results = mnn_model("https://ultralytics.com/images/bus.jpg")  # predict with `fp32`
        results = mnn_model("https://ultralytics.com/images/bus.jpg", half=True)  # predict with `fp16` if device support

        for result in results:
            result.show()  # display to screen
            result.save(filename="result.jpg")  # save to disk
        ```

    === "CLI"

        ```bash
        yolo predict model='yolo11n.mnn' source='https://ultralytics.com/images/bus.jpg'              # predict with `fp32`
        yolo predict model='yolo11n.mnn' source='https://ultralytics.com/images/bus.jpg' --half=True  # predict with `fp16` if device support
        ```

### What platforms are supported for MNN?

MNN is versatile and supports various platforms:

- **Mobile**: Android, iOS, Harmony.
- **Embedded Systems and IoT Devices**: Devices like Raspberry Pi and NVIDIA Jetson.
- **Desktop and Servers**: Linux, Windows, and macOS.

### How can I deploy Ultralytics YOLO11 MNN models on Mobile Devices?

To deploy your YOLO11 models on Mobile devices:

1. **Build for Android**: Follow the [MNN Android](https://github.com/alibaba/MNN/tree/master/project/android).
2. **Build for iOS**: Follow the [MNN iOS](https://github.com/alibaba/MNN/tree/master/project/ios).
3. **Build for Harmony**: Follow the [MNN Harmony](https://github.com/alibaba/MNN/tree/master/project/harmony).
