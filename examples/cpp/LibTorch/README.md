# Ultralytics YOLO LibTorch Inference C++

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="LibTorch" src="https://img.shields.io/badge/LibTorch-EE4C2C.svg?logo=pytorch&logoColor=white"> <img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-5C3EE8.svg?logo=opencv&logoColor=white">

This example demonstrates how to perform inference using [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11) and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8) models in C++ with the [LibTorch (PyTorch C++ API)](https://docs.pytorch.org/cppdocs/). This allows for deploying Ultralytics YOLO models in C++ environments for efficient execution.

## ⚙️ Dependencies

Ensure you have the following dependencies installed before proceeding:

| Dependency   | Version  | Resource                                     |
| :----------- | :------- | :------------------------------------------- |
| OpenCV       | >=4.0.0  | [https://opencv.org/](https://opencv.org/)   |
| C++ Standard | >=17     | [https://isocpp.org/](https://isocpp.org/)   |
| CMake        | >=3.18   | [https://cmake.org/](https://cmake.org/)     |
| Libtorch     | >=1.12.1 | [https://pytorch.org/](https://pytorch.org/) |

You can download the required version of LibTorch from the official [PyTorch](https://pytorch.org/) website. Make sure to select the correct version corresponding to your system and CUDA version (if using GPU).

## 🚀 Usage

Follow these steps to run the C++ inference example:

1.  **Clone the Ultralytics Repository:**
    Use [Git](https://git-scm.com/) to clone the repository containing the example code and necessary files.

    ```bash
    git clone https://github.com/ultralytics/ultralytics
    ```

2.  **Install Ultralytics:**
    Navigate to the cloned directory and install the `ultralytics` package using [pip](https://pip.pypa.io/en/stable/). This step is necessary for exporting the model. Refer to the [Ultralytics Quickstart Guide](https://docs.ultralytics.com/quickstart) for detailed installation instructions.

    ```bash
    cd ultralytics
    pip install .
    ```

3.  **Navigate to the Example Directory:**
    Change the directory to the C++ LibTorch inference example.

    ```bash
    cd examples/cpp/LibTorch
    ```

4.  **Build the Project:**
    Create a build directory, use [CMake](https://cmake.org/) to configure the project, and then compile it using [Make](https://www.gnu.org/software/make/). You might need to specify the path to your LibTorch and OpenCV installations if they are not found automatically by CMake.

    ```bash
    mkdir build
    cd build
    cmake .. # Add -DCMAKE_PREFIX_PATH=/path/to/libtorch;/path/to/opencv if needed
    make
    ```

5.  **Run the Inference:**
    The binary loads `yolo11s.torchscript` and an image named `bus.jpg` from the **current working directory**. Copy both next to the binary (or `cd` into the folder that holds them) and run it. To use different filenames, edit the `model_path` and `cv::imread(...)` values near the top of `main()` in `main.cc`.
    ```bash
    cp /path/to/yolo11s.torchscript .
    cp /path/to/bus.jpg .
    ./yolo_libtorch
    ```

    Expected output (using `yolo11s` on the sample `bus.jpg`):

    ```
    Rect: [20,227,796,737]  Conf: 0.9457  Class: bus
    Rect: [48,401,246,905]  Conf: 0.8988  Class: person
    Rect: [222,405,344,857]  Conf: 0.8961  Class: person
    Rect: [668,390,809,880]  Conf: 0.8852  Class: person
    Rect: [0,545,78,874]  Conf: 0.7306  Class: person
    ```

## ✨ Exporting Ultralytics YOLO

To use an Ultralytics YOLO model with LibTorch, you first need to export it to the [TorchScript](https://docs.pytorch.org/docs/stable/jit.html) format. TorchScript is a way to create serializable and optimizable models from PyTorch code.

Use the `yolo` [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli) provided by the `ultralytics` package to export the model. For example, to export the `yolo11s.pt` model with an input image size of 640x640:

```bash
yolo export model=yolo11s.pt imgsz=640 format=torchscript
```

This command will generate a `yolo11s.torchscript` file in the model's directory. This file contains the serialized model that can be loaded and executed by the C++ application using LibTorch. For more details on exporting models to various formats, see the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export).

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
