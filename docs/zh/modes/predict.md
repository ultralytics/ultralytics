---
comments: true
description: 了解如何使用 YOLOv8 预测模式进行各种任务。学习关于不同推理源如图像，视频和数据格式的内容。
keywords: Ultralytics, YOLOv8, 预测模式, 推理源, 预测任务, 流式模式, 图像处理, 视频处理, 机器学习, 人工智能
---

# 使用 Ultralytics YOLO 进行模型预测

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO 生态系统和集成">

## 引言

在机器学习和计算机视觉领域，将视觉数据转化为有用信息的过程被称为'推理'或'预测'。Ultralytics YOLOv8 提供了一个强大的功能，称为 **预测模式**，它专为各种数据来源的高性能实时推理而设计。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/QtsI0TnwDZs?si=ljesw75cMO2Eas14"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> 如何从 Ultralytics YOLOv8 模型中提取输出，用于自定义项目。
</p>

## 实际应用领域

|                                                            制造业                                                            |                                                             体育                                                              |                                                           安全                                                            |
|:-----------------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------:|
| ![车辆零部件检测](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a0f802a8-0776-44cf-8f17-93974a4a28a1) | ![足球运动员检测](https://github.com/RizwanMunawar/ultralytics/assets/62513924/7d320e1f-fc57-4d7f-a691-78ee579c3442) | ![人员摔倒检测](https://github.com/RizwanMunawar/ultralytics/assets/62513924/86437c4a-3227-4eee-90ef-9efb697bdb43) |
|                                                    车辆零部件检测                                                    |                                                    足球运动员检测                                                    |                                                    人员摔倒检测                                                    |

## 为何使用 Ultralytics YOLO 进行推理？

以下是考虑使用 YOLOv8 的预测模式满足您的各种推理需求的几个原因：

- **多功能性：** 能够对图像、视频乃至实时流进行推理。
- **性能：** 工程化为实时、高速处理而设计，不牺牲准确性。
- **易用性：** 直观的 Python 和 CLI 接口，便于快速部署和测试。
- **高度可定制性：** 多种设置和参数可调，依据您的具体需求调整模型的推理行为。

### 预测模式的关键特性

YOLOv8 的预测模式被设计为强大且多功能，包括以下特性：

- **兼容多个数据来源：** 无论您的数据是单独图片，图片集合，视频文件，还是实时视频流，预测模式都能胜任。
- **流式模式：** 使用流式功能生成一个内存高效的 `Results` 对象生成器。在调用预测器时，通过设置 `stream=True` 来启用此功能。
- **批处理：** 能够在单个批次中处理多个图片或视频帧，进一步加快推理时间。
- **易于集成：** 由于其灵活的 API，易于与现有数据管道和其他软件组件集成。

Ultralytics YOLO 模型在进行推理时返回一个 Python `Results` 对象列表，或者当传入 `stream=True` 时，返回一个内存高效的 Python `Results` 对象生成器：

!!! 示例 "预测"

    === "使用 `stream=False` 返回列表"
        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 预训练的 YOLOv8n 模型

        # 在图片列表上运行批量推理
        results = model(['im1.jpg', 'im2.jpg'])  # 返回 Results 对象列表

        # 处理结果列表
        for result in results:
            boxes = result.boxes  # 边界框输出的 Boxes 对象
            masks = result.masks  # 分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
            probs = result.probs  # 分类输出的 Probs 对象
        ```

    === "使用 `stream=True` 返回生成器"
        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 预训练的 YOLOv8n 模型

        # 在图片列表上运行批量推理
        results = model(['im1.jpg', 'im2.jpg'], stream=True)  # 返回 Results 对象生成器

        # 处理结果生成器
        for result in results:
            boxes = result.boxes  # 边界框输出的 Boxes 对象
            masks = result.masks  # 分割掩码输出的 Masks 对象
            keypoints = result.keypoints  # 姿态输出的 Keypoints 对象
            probs = result.probs  # 分类输出的 Probs 对象
        ```

## 推理来源

YOLOv8 可以处理推理输入的不同类型，如下表所示。来源包括静态图像、视频流和各种数据格式。表格还表示了每种来源是否可以在流式模式下使用，使用参数 `stream=True` ✅。流式模式对于处理视频或实时流非常有利，因为它创建了结果的生成器，而不是将所有帧加载到内存。

!!! 提示 "提示"

    使用 `stream=True` 处理长视频或大型数据集来高效地管理内存。当 `stream=False` 时，所有帧或数据点的结果都将存储在内存中，这可能很快导致内存不足错误。相对地，`stream=True` 使用生成器，只保留当前帧或数据点的结果在内存中，显著减少了内存消耗，防止内存不足问题。

| 来源           | 参数                                       | 类型            | 备注                                                                                        |
|----------------|---------------------------------------------|-----------------|--------------------------------------------------------------------------------------------|
| 图像           | `'image.jpg'`                              | `str` 或 `Path` | 单个图像文件。                                                                                |
| URL            | `'https://ultralytics.com/images/bus.jpg'` | `str`           | 图像的 URL 地址。                                                                            |
| 截屏           | `'screen'`                                 | `str`           | 截取屏幕图像。                                                                              |
| PIL            | `Image.open('im.jpg')`                     | `PIL.Image`     | RGB 通道的 HWC 格式图像。                                                                    |
| OpenCV         | `cv2.imread('im.jpg')`                     | `np.ndarray`    | BGR 通道的 HWC 格式图像 `uint8 (0-255)`。                                                    |
| numpy          | `np.zeros((640,1280,3))`                   | `np.ndarray`    | BGR 通道的 HWC 格式图像 `uint8 (0-255)`。                                                    |
| torch          | `torch.zeros(16,3,320,640)`                | `torch.Tensor`  | RGB 通道的 BCHW 格式图像 `float32 (0.0-1.0)`。                                               |
| CSV            | `'sources.csv'`                            | `str` 或 `Path` | 包含图像、视频或目录路径的 CSV 文件。                                                        |
| 视频 ✅          | `'video.mp4'`                              | `str` 或 `Path` | 如 MP4, AVI 等格式的视频文件。                                                               |
| 目录 ✅          | `'path/'`                                  | `str` 或 `Path` | 包含图像或视频文件的目录路径。                                                                |
| 通配符 ✅       | `'path/*.jpg'`                             | `str`           | 匹配多个文件的通配符模式。使用 `*` 字符作为通配符。                                          |
| YouTube ✅      | `'https://youtu.be/LNwODJXcvt4'`           | `str`           | YouTube 视频的 URL 地址。                                                                     |
| 流媒体 ✅        | `'rtsp://example.com/media.mp4'`           | `str`           | RTSP, RTMP, TCP 或 IP 地址等流协议的 URL 地址。                                              |
| 多流媒体 ✅      | `'list.streams'`                           | `str` 或 `Path` | 一个流 URL 每行的 `*.streams` 文本文件，例如 8 个流将以 8 的批处理大小运行。                |

下面为每种来源类型使用代码的示例：

!!! 示例 "预测来源"

    === "图像"
        对图像文件进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义图像文件的路径
        source = 'path/to/image.jpg'

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "截屏"
        对当前屏幕内容作为截屏进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义当前截屏为来源
        source = 'screen'

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "URL"
        对通过 URL 远程托管的图像或视频进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义远程图像或视频 URL
        source = 'https://ultralytics.com/images/bus.jpg'

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "PIL"
        对使用 Python Imaging Library (PIL) 打开的图像进行推理。
        ```python
        from PIL import Image
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 使用 PIL 打开图像
        source = Image.open('path/to/image.jpg')

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "OpenCV"
        对使用 OpenCV 读取的图像进行推理。
        ```python
        import cv2
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 使用 OpenCV 读取图像
        source = cv2.imread('path/to/image.jpg')

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "numpy"
        对表示为 numpy 数组的图像进行推理。
        ```python
        import numpy as np
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 创建一个 HWC 形状 (640, 640, 3) 的随机 numpy 数组，数值范围 [0, 255] 类型为 uint8
        source = np.random.randint(low=0, high=255, size=(640, 640, 3), dtype='uint8')

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "torch"
        对表示为 PyTorch 张量的图像进行推理。
        ```python
        import torch
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 创建一个 BCHW 形状 (1, 3, 640, 640) 的随机 torch 张量，数值范围 [0, 1] 类型为 float32
        source = torch.rand(1, 3, 640, 640, dtype=torch.float32)

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "CSV"
        对 CSV 文件中列出的图像、URLs、视频和目录进行推理。
        ```python
        import torch
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义一个包含图像、URLs、视频和目录路径的 CSV 文件路径
        source = 'path/to/file.csv'

        # 对来源进行推理
        results = model(source)  # Results 对象列表
        ```

    === "视频"
        对视频文件进行推理。使用 `stream=True` 时，可以创建一个 Results 对象的生成器，减少内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义视频文件路径
        source = 'path/to/video.mp4'

        # 对来源进行推理
        results = model(source, stream=True)  # Results 对象的生成器
        ```

    === "目录"
        对目录中的所有图像和视频进行推理。要包含子目录中的图像和视频，使用通配符模式，例如 `path/to/dir/**/*`。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义包含图像和视频文件用于推理的目录路径
        source = 'path/to/dir'

        # 对来源进行推理
        results = model(source, stream=True)  # Results 对象的生成器
        ```

    === "通配符"
        对与 `*` 字符匹配的所有图像和视频进行推理。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义一个目录下所有 JPG 文件的通配符搜索
        source = 'path/to/dir/*.jpg'

        # 或定义一个包括子目录的所有 JPG 文件的递归通配符搜索
        source = 'path/to/dir/**/*.jpg'

        # 对来源进行推理
        results = model(source, stream=True)  # Results 对象的生成器
        ```

    === "YouTube"
        对 YouTube 视频进行推理。使用 `stream=True` 时，可以为长视频创建一个 Results 对象的生成器，减少内存使用。
        ```python
        from ultralytics import YOLO

        # 加载预训练的 YOLOv8n 模型
        model = YOLO('yolov8n.pt')

        # 定义 YouTube 视频 URL 作为来源
        source = 'https://youtu.be/LNwODJXcvt4'

        # 对来源进行推理
        results = model(source, stream=True)  # Results 对象的生成器
        ```

    === "流媒体"
        使用 RTSP, RTMP, TCP 和 IP 地址
