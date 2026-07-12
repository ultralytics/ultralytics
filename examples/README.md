<a href="https://www.ultralytics.com/" target="_blank"><img src="https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg" width="320" alt="Ultralytics logo"></a>

# Ultralytics YOLO Examples

> [!WARNING]
> The examples in this directory are community-contributed and showcase creative ways to use Ultralytics YOLO models. While we truly appreciate these contributions, they may not always reflect the latest best practices or receive regular updates. To help streamline our codebase and focus our resources on maintaining comprehensive, up-to-date official documentation and guides, we plan to retire these examples in a future Ultralytics release.

Welcome to the Ultralytics examples directory! This collection showcases practical applications and detailed walkthroughs for integrating [Ultralytics YOLO models](https://docs.ultralytics.com/models) into various real-world projects. Explore Python scripts and Jupyter notebooks designed to help you leverage the power of models like [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26) for tasks like [object detection](https://docs.ultralytics.com/tasks/detect), [instance segmentation](https://docs.ultralytics.com/tasks/segment), [semantic segmentation](https://docs.ultralytics.com/tasks/semantic), [pose estimation](https://docs.ultralytics.com/tasks/pose), and more.

Whether you're deploying models on [edge devices](https://www.ultralytics.com/glossary/edge-ai) using formats like [ONNX](https://docs.ultralytics.com/integrations/onnx) with [ONNX Runtime](https://onnxruntime.ai/), optimizing with [TensorRT](https://docs.ultralytics.com/integrations/tensorrt) on NVIDIA Jetson, using [OpenVINO](https://docs.ultralytics.com/integrations/openvino) for Intel hardware, or integrating with frameworks like [OpenCV](https://opencv.org/), these examples provide valuable insights and code snippets. Find inspiration for your next [computer vision](https://www.ultralytics.com/glossary/computer-vision-cv) project and see how others are using Ultralytics YOLO to build innovative [AI solutions](https://www.ultralytics.com/solutions) on platforms ranging from C++ and C# to Python and Rust.

## 💡 Example Applications

Browse through the community-contributed examples below. These projects demonstrate various use cases and deployment strategies for Ultralytics YOLO models across different platforms and programming languages. All C++ examples are grouped together under [`./cpp`](./cpp) with a combined build-and-test guide.

### C++

All C++ examples live under [`./cpp`](./cpp) and support every YOLO task and generation. See the [C++ examples guide](./cpp) for a combined build-and-test walkthrough.

| Title                                      | Backend      | Contributor                                                                                                                                                                                       |
| ------------------------------------------ | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [YOLO OpenCV DNN CPP](./cpp/OpenCV-DNN)    | OpenCV DNN   | [Justas Bartnykas](https://github.com/JustasBart)                                                                                                                                                 |
| [YOLO ONNX Runtime CPP](./cpp/ONNXRuntime) | ONNX Runtime | [DennisJcy](https://github.com/DennisJcy), [Onuralp Sezer](https://github.com/onuralpszr)                                                                                                         |
| [YOLO LibTorch CPP](./cpp/LibTorch)        | LibTorch     | [Myyura](https://github.com/Myyura)                                                                                                                                                               |
| [YOLO OpenVINO CPP](./cpp/OpenVINO)        | OpenVINO     | [Erlangga Yudi Pradana](https://github.com/rlggyp)                                                                                                                                                |
| [YOLO MNN CPP](./cpp/MNN)                  | MNN          | [Khoi VN](https://github.com/vnk8071)                                                                                                                                                             |
| [YOLO Triton CPP](./cpp/Triton)            | Triton       | [Ahmet Selim Demirel](https://github.com/asdemirel), [Doğan Mehmet Başoğlu](https://github.com/doganmb), [Enes Uzun](https://github.com/uzunenes), [Mevlüt Ardıç](https://github.com/mevlutardic) |

### Python

| Title                                                                                                                                     | Backend               | Contributor                                                                                                                                         |
| ----------------------------------------------------------------------------------------------------------------------------------------- | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| [YOLO OpenCV ONNX Detection](./YOLOv8-OpenCV-ONNX-Python)                                                                                 | OpenCV / ONNX         | [Farid Inawan](https://github.com/frdteknikelektro)                                                                                                 |
| [YOLOv8 ONNX Runtime](./YOLOv8-ONNXRuntime)                                                                                               | ONNX Runtime          | [Semih Demirel](https://github.com/semihhdemirel)                                                                                                   |
| [RTDETR ONNX Runtime](./RTDETR-ONNXRuntime-Python)                                                                                        | ONNX Runtime          | [Semih Demirel](https://github.com/semihhdemirel)                                                                                                   |
| [YOLOv8 Segmentation ONNX Runtime](./YOLOv8-Segmentation-ONNXRuntime-Python)                                                              | ONNX Runtime          | [jamjamjon](https://github.com/jamjamjon)                                                                                                           |
| [YOLO SKU Recognition (Detect + ReID)](./YOLO-SKU-Recognition)                                                                            | Detection + ReID      | [Fatih Akyon](https://github.com/fcakyon)                                                                                                           |
| [YOLOv8 SAHI Video Inference](https://github.com/RizwanMunawar/ultralytics/blob/main/examples/YOLOv8-SAHI-Inference-Video/yolov8_sahi.py) | SAHI                  | [Muhammad Rizwan Munawar](https://github.com/RizwanMunawar) ([See also SAHI Guide](https://docs.ultralytics.com/guides/sahi-tiled-inference))       |
| [YOLOv8 Region Counter](https://github.com/RizwanMunawar/ultralytics/blob/main/examples/YOLOv8-Region-Counter/yolov8_region_counter.py)   | Region Counting       | [Muhammad Rizwan Munawar](https://github.com/RizwanMunawar) ([See also Region Counting Guide](https://docs.ultralytics.com/guides/region-counting)) |
| [YOLOv8 on NVIDIA Jetson (TensorRT and DeepStream)](https://wiki.seeedstudio.com/YOLOv8-DeepStream-TRT-Jetson/)                           | TensorRT / DeepStream | [Lakshantha](https://github.com/lakshanthad) ([See also DeepStream Guide](https://docs.ultralytics.com/guides/deepstream-nvidia-jetson))            |

### Rust

| Title                                                             | Backend      | Contributor                               |
| ----------------------------------------------------------------- | ------------ | ----------------------------------------- |
| [YOLOv8 All Tasks ONNX Runtime Rust](./YOLOv8-ONNXRuntime-Rust)   | ONNX Runtime | [jamjamjon](https://github.com/jamjamjon) |
| [YOLOv5-YOLO11 ONNX Runtime Rust](./YOLO-Series-ONNXRuntime-Rust) | ONNX Runtime | [jamjamjon](https://github.com/jamjamjon) |

### C# / .NET

| Title                                                                              | Backend      | Contributor                                     |
| ---------------------------------------------------------------------------------- | ------------ | ----------------------------------------------- |
| [YOLO C# ONNX Runtime (YoloSharp)](https://github.com/dme-compunet/YoloSharp)      | ONNX Runtime | [Compunet](https://github.com/dme-compunet)     |
| [YOLO .NET ONNX Detection (Yolov8.Net)](https://www.nuget.org/packages/Yolov8.Net) | ONNX         | [Samuel Stainback](https://github.com/sstainba) |
| [RTDETR ONNX Runtime C#](https://github.com/Kayzwer/yolo-cs/blob/master/RTDETR.cs) | ONNX Runtime | [Kayzwer](https://github.com/Kayzwer)           |

## 🤝 How to Contribute

We actively encourage contributions from our vibrant community! Sharing your examples, applications, and guides helps others learn and build amazing things with [Ultralytics](https://www.ultralytics.com/). If you have a project you'd like to share, please follow these steps:

1.  **Fork the Repository:** Start by forking the main [Ultralytics repository](https://github.com/ultralytics/ultralytics) on [GitHub](https://github.com/).
2.  **Create Your Example:** Add your project folder within the `examples/` directory of your forked repository.
3.  **Prepare Your Submission:** Ensure your project meets the following criteria:
    - It utilizes the `ultralytics` pip package.
    - Includes a `README.md` file with clear, step-by-step instructions for setup and execution. Explain the purpose of the example and any prerequisites.
    - Avoid committing large files or extensive dependencies. If necessary, provide instructions for users to download them separately (e.g., using `ultralytics.utils.downloads.safe_download()`).
    - As a contributor, be prepared to offer support and address [issues](https://github.com/ultralytics/ultralytics/issues) related to your example.
4.  **Submit a Pull Request:** Create a [pull request (PR)](https://github.com/ultralytics/ultralytics/pulls) targeting the `main` branch of the official Ultralytics repository. Use the title prefix `[Example]` (e.g., `[Example] Add YOLOv8 Pose Estimation on Raspberry Pi`).

For more comprehensive guidelines on contributing code, documentation, or examples, please refer to our [Contributing Guide](https://docs.ultralytics.com/help/contributing). We appreciate your efforts to enhance the Ultralytics ecosystem! If you have questions, feel free to open an issue or PR, and the team will be happy to assist. Check out the [Ultralytics Blog](https://www.ultralytics.com/blog) for more insights and updates, and explore [Ultralytics Platform](https://platform.ultralytics.com/ultralytics/yolo26) for streamlined model training and deployment.
