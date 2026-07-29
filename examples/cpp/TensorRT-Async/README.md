# YOLOv8 TensorRT C++ 三线程异步实时检测

该项目面向配备 NVIDIA GPU 的本地 PC：取流、TensorRT 推理、画面显示分别运行在三个线程中。两个固定容量队列在积压时丢弃旧数据，消费者每次也只取最新数据，避免系统以高吞吐量持续处理已经过时的画面。

## 流水线

1. **取流线程**从摄像头读取 BGR 图像，并将其移动到输入队列。OpenCV 后端支持时，摄像头内部缓冲设置为一帧。
2. **推理线程**完成居中 Letterbox、BGR→RGB、HWC→NCHW 和 `[0, 1]` 归一化；随后通过固定的页锁定主机内存、GPU 输入/输出缓冲及非阻塞 CUDA Stream 执行异步 H2D、`enqueueV3` 和异步 D2H；最后在 CPU 上执行类别感知 NMS。
3. **显示线程**在原图上绘制检测框、类别、置信度和各阶段耗时，并处理 `Q`/`Esc` 退出事件。

项目仅支持 YOLOv8 **目标检测**模型的 batch=1、固定尺寸、单输入/单输出原始输出 Engine。为使用下述 `--fp16` 构建方式，TensorRT 需为 8.6–10.x；TensorRT 11 已移除该精度参数。

## 1. 导出 ONNX

在 Ultralytics 仓库根目录执行：

```powershell
yolo export model=yolov8n.pt format=onnx imgsz=640 batch=1 dynamic=False simplify=True
```

不要为 ONNX 设置 `half=True`。TensorRT 内部 FP16 由下一步开启，Engine 输入输出继续保持 FP32，避免 CPU 侧额外的 FP16 转换。

## 2. 使用 trtexec 构建 FP16 Engine

在部署程序的同一台电脑、同一块 GPU 上执行：

```powershell
trtexec --onnx=yolov8n.onnx --saveEngine=yolov8n_fp16.engine --fp16 --skipInference
```

确认日志中出现 FP16 tactic。TensorRT Engine 与构建时的 TensorRT 版本和 GPU 架构相关，不应直接复制到不同型号的电脑使用。

此运行器不接受动态 Engine；请保持上一步的 `dynamic=False`。TensorRT 11 采用强类型网络，若必须使用 TensorRT 11，需要先对 ONNX 做显式 FP16 转换，并调整本项目的版本约束。

## 3. 编译

依赖：

- Visual Studio 2019 或更高版本，支持 C++17
- CMake 3.18+
- CUDA Toolkit
- TensorRT 8.6–10.x
- OpenCV 4.x

使用 Visual Studio 生成器：

```powershell
cd examples\cpp\TensorRT-Async
cmake -S . -B build -A x64 -DTENSORRT_ROOT="C:\TensorRT-10.x" -DOpenCV_DIR="C:\opencv\build"
cmake --build build --config Release
```

Linux：

```bash
cd examples/cpp/TensorRT-Async
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DTENSORRT_ROOT=/opt/TensorRT
cmake --build build -j
```

## 4. 运行

Windows Visual Studio 构建：

```powershell
.\build\Release\yolo_tensorrt_async.exe --engine .\yolov8n_fp16.engine --camera 0 --width 1280 --height 720
```

Linux：

```bash
./build/yolo_tensorrt_async --engine ./yolov8n_fp16.engine --camera 0 --width 1280 --height 720
```

自定义数据集需提供每行一个类别名的 UTF-8 文本，行数必须与模型类别数一致，并用于确定 TensorRT 输出布局：

```powershell
.\build\Release\yolo_tensorrt_async.exe --engine .\best_fp16.engine --labels .\classes.txt
```

可调参数：

```text
--conf 0.25          置信度阈值
--iou 0.45           NMS IoU 阈值
--input-queue 2      输入队列容量
--output-queue 2     输出队列容量
```

窗口状态栏的 `drop 输入/输出` 计数表示主动舍弃的过时数据。该数值增长是低延迟策略正常工作，并非推理错误。
