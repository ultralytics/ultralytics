# YOLOv11 TensorRT CUDA Inference C++

This example demonstrates how to perform inference using [Ultralytics YOLOv11](https://docs.ultralytics.com/models/yolov11/) models in CUDA with the [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/index.html). 

## ⚙️ Dependencies

Ensure you have the following dependencies installed before proceeding:

| Dependency   | Version  | Resource                                     |
| :----------- | :------- | :------------------------------------------- |
| OpenCV       | >=4.0.0  | [https://opencv.org/](https://opencv.org/)   |
| C++ Standard | >=17     | [https://isocpp.org/](https://isocpp.org/)   |
| CMake        | >=3.18   | [https://cmake.org/](https://cmake.org/)     |
| TensorRT     | >=10.12.0.36 | [https://developer.nvidia.com/tensorrt](https://developer.nvidia.com/tensorrt) |


## 🚀 Usage

Follow these steps to run the TensorRT inference example:

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
    Change the directory to the TensorRT inference example.

    ```bash
    cd examples/YOLO11-TensorRT-CUDA-Inference
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
    Execute the compiled binary. The application will load the exported YOLOv11 model and perform inference on a sample image (`zidane.jpg` included in the root `ultralytics` directory) or video.
    ```bash
    ./YOLOTensorRTInference -input "zidane.jpg" -model "/path/to/yolov11s.engine"
    ```

## ✨ Exporting Ultralytics YOLOv11

To use an Ultralytics YOLOv11 model with TensorRT, you first need to convert it to the Engine format.

Use the `yolo` [Command Line Interface (CLI)](https://docs.ultralytics.com/usage/cli/) provided by the `ultralytics` package to export the model to ONNX format first. For example, to export the `yolov11s.pt` model:

```bash
yolo export model=yolov11s.pt format=onnx nms=True
```

This command will generate a `yolov11s.onnx` file in the model's directory. After this you need to convert it to TensorRT Engine which can be done using `trtexec` program that can be found in the `bin` folder of `TensorRT-10.12.0.36` which you must have downloaded in the previous steps.

```bash
./trtexec --onnx=/path/to/yolov11s.onnx --saveEngine=yolov11s.engine --fp16 --device=0 --verbose
```
