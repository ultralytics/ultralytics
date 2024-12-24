## Ultralytics Examples

This directory features a collection of real-world applications and walkthroughs, provided as either Python files or notebooks. Explore the examples below to see how YOLO can be integrated into various applications.

### Ultralytics YOLO Example Applications

| Title                                                                                                                                     | Format             | Contributor                                                                               |
| ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------- |
| [YOLO ONNX Detection Inference with C++](./YOLOv8-CPP-Inference)                                                                          | C++/ONNX           | [Justas Bartnykas](https://github.com/JustasBart)                                         |
| [YOLO OpenCV ONNX Detection Python](./YOLOv8-OpenCV-ONNX-Python)                                                                          | OpenCV/Python/ONNX | [Farid Inawan](https://github.com/frdteknikelektro)                                       |
| [YOLO C# ONNX-Runtime](https://github.com/dme-compunet/YoloSharp)                                                                         | .NET/ONNX-Runtime  | [Compunet](https://github.com/dme-compunet)                                               |
| [YOLO .Net ONNX Detection C#](https://www.nuget.org/packages/Yolov8.Net)                                                                  | C# .Net            | [Samuel Stainback](https://github.com/sstainba)                                           |
| [YOLOv8 on NVIDIA Jetson(TensorRT and DeepStream)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)                            | Python             | [Lakshantha](https://github.com/lakshanthad)                                              |
| [YOLOv8 ONNXRuntime Python](./YOLOv8-ONNXRuntime)                                                                                         | Python/ONNXRuntime | [Semih Demirel](https://github.com/semihhdemirel)                                         |
| [RTDETR ONNXRuntime Python](./RTDETR-ONNXRuntime-Python)                                                                                  | Python/ONNXRuntime | [Semih Demirel](https://github.com/semihhdemirel)                                         |
| [YOLOv8 ONNXRuntime CPP](./YOLOv8-ONNXRuntime-CPP)                                                                                        | C++/ONNXRuntime    | [DennisJcy](https://github.com/DennisJcy), [Onuralp Sezer](https://github.com/onuralpszr) |
| [RTDETR ONNXRuntime C#](https://github.com/Kayzwer/yolo-cs/blob/master/RTDETR.cs)                                                         | C#/ONNX            | [Kayzwer](https://github.com/Kayzwer)                                                     |
| [YOLOv8 SAHI Video Inference](https://github.com/RizwanMunawar/ultralytics/blob/main/examples/YOLOv8-SAHI-Inference-Video/yolov8_sahi.py) | Python             | [Muhammad Rizwan Munawar](https://github.com/RizwanMunawar)                               |
| [YOLOv8 Region Counter](https://github.com/RizwanMunawar/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py)   | Python             | [Muhammad Rizwan Munawar](https://github.com/RizwanMunawar)                               |
| [YOLOv8 Segmentation ONNXRuntime Python](./YOLOv8-Segmentation-ONNXRuntime-Python)                                                        | Python/ONNXRuntime | [jamjamjon](https://github.com/jamjamjon)                                                 |
| [YOLOv8 LibTorch CPP](./YOLOv8-LibTorch-CPP-Inference)                                                                                    | C++/LibTorch       | [Myyura](https://github.com/Myyura)                                                       |
| [YOLOv8 OpenCV INT8 TFLite Python](./YOLOv8-TFLite-Python)                                                                                | Python             | [Wamiq Raza](https://github.com/wamiqraza)                                                |
| [YOLOv8 All Tasks ONNXRuntime Rust](./YOLOv8-ONNXRuntime-Rust)                                                                            | Rust/ONNXRuntime   | [jamjamjon](https://github.com/jamjamjon)                                                 |
| [YOLOv8 OpenVINO CPP](./YOLOv8-OpenVINO-CPP-Inference)                                                                                    | C++/OpenVINO       | [Erlangga Yudi Pradana](https://github.com/rlggyp)                                        |
| [YOLOv5-YOLO11 ONNXRuntime Rust](./YOLO-Series-ONNXRuntime-Rust)                                                                          | Rust/ONNXRuntime   | [jamjamjon](https://github.com/jamjamjon)                                                 |

### How to Contribute

We greatly appreciate contributions from the community, including examples, applications, and guides. If you'd like to contribute, please follow these guidelines:

1. **Create a pull request (PR)** with the title prefix `[Example]`, adding your new example folder to the `examples/` directory within the repository.
2. **Ensure your project adheres to the following standards:**
   - Makes use of the `ultralytics` package.
   - Includes a `README.md` with clear instructions for setting up and running the example.
   - Avoids adding large files or dependencies unless they are absolutely necessary for the example.
   - Contributors should be willing to provide support for their examples and address related issues.

For more detailed information and guidance on contributing, please visit our [contribution documentation](https://docs.ultralytics.com/help/contributing/).

If you encounter any questions or concerns regarding these guidelines, feel free to open a PR or an issue in the repository, and we will assist you in the contribution process.
