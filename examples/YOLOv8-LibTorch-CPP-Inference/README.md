# YOLOv8 LibTorch Inference C++

This example demonstrates how to perform inference using [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) models in C++ with the [LibTorch (PyTorch C++ API)](https://docs.pytorch.org/cppdocs/). This allows for deploying YOLOv8 models in C++ environments for efficient execution.

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
    Navigate to the cloned directory and install the `ultralytics` package using [pip](https://pip.pypa.io/en/stable/). This step is necessary for exporting the model. Refer to the [Ultralytics Quickstart Guide](https://docs.ultralytics.com/quickstart/) for detailed installation instructions.

    ```bash
    cd ultralytics
    pip install .
    ```

3.  **Navigate to the Example Directory:**
    Change the directory to the C++ LibTorch inference example.

    ```bash
    cd examples/YOLOv8-LibTorch-CPP-Inference
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
    Execute the compiled binary. The application will load the exported YOLOv8 model and perform inference on a sample image (`zidane.jpg` included in the root `ultralytics` directory) or video.
    ```bash
    ./yolov8_libtorch_inference
    ```

## ✨ Exporting Ultralytics YOLOv8

To use an Ultralytics YOLOv8 model with LibTorch, you first need to export it to the [TorchScript](https://docs.pytorch.org/docs/stable/jit.html) format. TorchScript is a way to create serializable and optimizable models from PyTorch code.

Use the `yolo` [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli/) provided by the `ultralytics` package to export the model. For example, to export the `yolov8s.pt` model with an input image size of 640x640:

```bash
yolo export model=yolov8s.pt imgsz=640 format=torchscript
```

This command will generate a `yolov8s.torchscript` file in the model's directory. This file contains the serialized model that can be loaded and executed by the C++ application using LibTorch. For more details on exporting models to various formats, see the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/).

## 🤝 Contributing

Contributions to enhance this example or add new features are welcome! Please see the [Ultralytics Contributing Guide](https://docs.ultralytics.com/help/contributing/) for guidelines on how to contribute to the project. Thank you for helping make Ultralytics YOLO the best Vision AI tool!
