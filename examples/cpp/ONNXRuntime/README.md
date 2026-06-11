# Ultralytics YOLO ONNX Runtime C++ Example

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white">

This example provides a practical guide on performing inference with [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11) and [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8) models using [C++](https://isocpp.org/), leveraging the capabilities of the [ONNX Runtime](https://onnxruntime.ai/) and the [OpenCV](https://opencv.org/) library. It's designed for developers looking to integrate Ultralytics YOLO into C++ applications for efficient object detection.

## ✨ Benefits

- **Deployment-Friendly:** Well-suited for deployment in industrial and production environments.
- **Performance:** Offers faster [inference latency](https://www.ultralytics.com/glossary/inference-latency) compared to OpenCV's DNN module on both CPU and [GPU](https://www.ultralytics.com/glossary/gpu-graphics-processing-unit).
- **Acceleration:** Supports FP32 and [FP16 (Half Precision)](https://www.ultralytics.com/glossary/half-precision) inference acceleration using [NVIDIA CUDA](https://developer.nvidia.com/cuda/toolkit).

## ☕ Note

Thanks to recent updates in Ultralytics, YOLO models include a `Transpose` operation that aligns their output shape with YOLOv5. This allows the C++ code in this project to run inference seamlessly for YOLOv5, YOLOv7, YOLOv8, and YOLO11 models exported to the [ONNX format](https://onnx.ai/). For YOLO26, export with NMS disabled (`nms=False`) so the raw output shape matches what this code parses.

## 📦 Exporting Ultralytics YOLO Models

You can export your trained [Ultralytics YOLO](https://docs.ultralytics.com/) models to the ONNX format required by this project. Use the Ultralytics `export` mode for this.

### Python

```python
from ultralytics import YOLO

# Load an Ultralytics YOLO model (e.g., yolo11n.pt)
model = YOLO("yolo11n.pt")

# Export the model to ONNX format
# opset=12 is recommended for compatibility
# simplify=True optimizes the model graph
# dynamic=False ensures fixed input size, often better for C++ deployment
# imgsz=640 sets the input image size
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
print("Model exported successfully to yolo11n.onnx")
```

### CLI

```bash
# Export the model using the command line
yolo export model=yolo11n.pt format=onnx opset=12 simplify=True dynamic=False imgsz=640
```

For more details on exporting models, refer to the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export).

## 📦 Exporting FP16 Models

To potentially gain further performance on compatible hardware (like NVIDIA GPUs), you can convert the exported FP32 ONNX model to FP16.

```python
import onnx
from onnxconverter_common import (
    float16,
)  # Ensure you have onnxconverter-common installed: pip install onnxconverter-common

# Load your FP32 ONNX model
fp32_model_path = "yolo11n.onnx"
model = onnx.load(fp32_model_path)

# Convert the model to FP16
model_fp16 = float16.convert_float_to_float16(model)

# Save the FP16 model
fp16_model_path = "yolo11n_fp16.onnx"
onnx.save(model_fp16, fp16_model_path)
print(f"Model converted and saved to {fp16_model_path}")
```

## 🏷️ Class Names

Class names are read directly from the model's metadata, which [Ultralytics](https://docs.ultralytics.com/) bakes into the exported `.onnx` file. No external `coco.yaml` is required for detection. If a model has no `names` metadata, the example falls back to the standard 80 [COCO](https://docs.ultralytics.com/datasets/detect/coco) names from `../common/coco_names.hpp`.

## ⚙️ Dependencies

Ensure you have the following dependencies installed:

| Dependency                                                           | Version       | Notes                                                                                                                                                                       |
| :------------------------------------------------------------------- | :------------ | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [ONNX Runtime](https://onnxruntime.ai/docs/install/)                 | >=1.14.1      | Download pre-built binaries or build from source. Ensure GPU version if using CUDA.                                                                                         |
| [OpenCV](https://opencv.org/releases/)                               | >=4.0.0       | Required for image loading and preprocessing.                                                                                                                               |
| C++ Compiler                                                         | C++17 Support | Needed for features like `<filesystem>`. ([GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), [MSVC](https://visualstudio.microsoft.com/vs/features/cplusplus/)) |
| [CMake](https://cmake.org/download/)                                 | >=3.5         | Cross-platform build system generator. Version 3.18+ recommended for better CUDA support discovery.                                                                         |
| [CUDA Toolkit](https://developer.nvidia.com/cuda/toolkit) (Optional) | >=11.4, <12.0 | Required for GPU acceleration via ONNX Runtime's CUDA Execution Provider. **Must be CUDA 11.x**.                                                                            |
| [cuDNN](https://developer.nvidia.com/cudnn) (CUDA required)          | =8.x          | Required by CUDA Execution Provider. **Must be cuDNN 8.x** compatible with your CUDA 11.x version.                                                                          |

**Important Notes:**

1.  **C++17:** The requirement stems from using the `<filesystem>` library introduced in C++17 for path handling.
2.  **CUDA/cuDNN Versions:** ONNX Runtime's CUDA execution provider currently has strict version requirements (CUDA 11.x, cuDNN 8.x). Check the latest [ONNX Runtime documentation](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html) for any updates to these constraints. Using incompatible versions will lead to runtime errors.

## 🛠️ Build Instructions

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/ultralytics/ultralytics.git
    cd ultralytics/examples/cpp/ONNXRuntime
    ```

2.  **Create Build Directory:**

    ```bash
    mkdir build && cd build
    ```

3.  **Configure with CMake:**
    Run CMake to generate build files. You **must** specify the path to your ONNX Runtime installation directory using `ONNXRUNTIME_ROOT`. Adjust the path according to where you downloaded or built ONNX Runtime.

    ```bash
    # Example for Linux/macOS (adjust path as needed)
    cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime

    # Example for Windows (adjust path as needed, use backslashes or forward slashes)
    cmake .. -DONNXRUNTIME_ROOT="C:/path/to/onnxruntime"
    ```

    **CMake Options:**
    - `-DONNXRUNTIME_ROOT=<path>`: **(Required)** Path to the extracted ONNX Runtime library.
    - `-DCMAKE_BUILD_TYPE=Release`: (Optional) Build in Release mode for optimizations.
    - If CMake struggles to find OpenCV, you might need to set `-DOpenCV_DIR=/path/to/opencv/build`.

4.  **Build the Project:**
    Use the build tool generated by CMake (e.g., Make, Ninja, Visual Studio).

    ```bash
    # Using Make (common on Linux/macOS)
    make

    # Using CMake's generic build command (works with Make, Ninja, etc.)
    cmake --build . --config Release
    ```

5.  **Locate Executable:**
    The compiled executable (e.g., `yolo_onnxruntime`) will be located in the `build` directory.

## 🚀 Usage

For a **CPU-only** build, disable CUDA at configure time:

```bash
cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime -DUSE_CUDA=OFF
cmake --build . --config Release
```

The program looks for its assets relative to the working directory:

- `models/yolo11n.onnx` — the exported model (CMake auto-stages it here if you place `yolo11n.onnx` in the example source directory before configuring).
- `images/detect/` — input images (`.jpg`/`.png`); annotated `*_result.jpg` files are written back alongside them.

Run it from the `build` directory:

```bash
# add the ONNX Runtime libs to the loader path if they are not installed system-wide
LD_LIBRARY_PATH=/path/to/onnxruntime/lib ./yolo_onnxruntime
```

The detection parameters live in `DetectTest()` in `main.cpp`:

```cpp
DL_INIT_PARAM params;
params.rectConfidenceThreshold = 0.1;
params.iouThreshold = 0.5;
params.modelPath = "./models/yolo11n.onnx";
params.imgSize = { 640, 640 };
params.cudaEnable = false;       // set true for the CUDA execution provider
params.modelType = YOLO_DETECT;  // YOLO_DETECT / YOLO_POSE / YOLO_CLS (+ *_HALF for FP16)
```

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the main [Ultralytics repository](https://github.com/ultralytics/ultralytics).
