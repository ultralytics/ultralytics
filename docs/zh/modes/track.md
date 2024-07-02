---
comments: true
description: 学习如何使用Ultralytics YOLO进行视频流中的物体追踪。指南包括使用不同的追踪器和自定义追踪器配置。
keywords: Ultralytics, YOLO, 物体追踪, 视频流, BoT-SORT, ByteTrack, Python 指南, CLI 指南
---

# 使用Ultralytics YOLO进行多物体追踪

<img width="1024" src="https://user-images.githubusercontent.com/26833433/243418637-1d6250fd-1515-4c10-a844-a32818ae6d46.png" alt="多物体追踪示例">

视频分析领域的物体追踪是一项关键任务，它不仅能标识出帧内物体的位置和类别，还能在视频进行过程中为每个检测到的物体保持一个唯一的ID。应用场景无限广阔——从监控与安全到实时体育分析。

## 为什么选择Ultralytics YOLO进行物体追踪？

Ultralytics 追踪器的输出与标准的物体检测结果一致，但增加了物体ID的附加值。这使其易于追踪视频流中的物体并进行后续分析。以下是您应考虑使用Ultralytics YOLO来满足您物体追踪需求的原因：

- **效率：** 实时处理视频流，同时保持准确性。
- **灵活性：** 支持多种追踪算法和配置。
- **易用性：** 简单的Python API和CLI选项，便于快速集成和部署。
- **可定制性：** 易于使用自定义训练的YOLO模型，允许集成到特定领域的应用中。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/hHyHmOtmEgs?si=VNZtXmm45Nb9s-N-"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>使用Ultralytics YOLOv8的物体检测与追踪。
</p>

## 实际应用场景

|                                                    交通运输                                                    |                                                     零售                                                     |                                                    水产养殖                                                    |
|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|:----------------------------------------------------------------------------------------------------------:|
| ![车辆追踪](https://github.com/RizwanMunawar/ultralytics/assets/62513924/ee6e6038-383b-4f21-ac29-b2a1c7d386ab) | ![人员追踪](https://github.com/RizwanMunawar/ultralytics/assets/62513924/93bb4ee2-77a0-4e4e-8eb6-eb8f527f0527) | ![鱼类追踪](https://github.com/RizwanMunawar/ultralytics/assets/62513924/a5146d0f-bfa8-4e0a-b7df-3c1446cd8142) |
|                                                    车辆追踪                                                    |                                                    人员追踪                                                    |                                                    鱼类追踪                                                    |

## 一瞥特点

Ultralytics YOLO扩展了其物体检测功能，以提供强大且多功能的物体追踪：

- **实时追踪：** 在高帧率视频中无缝追踪物体。
- **支持多个追踪器：** 从多种成熟的追踪算法中选择。
- **自定义追踪器配置：** 通过调整各种参数来定制追踪算法，以满足特定需求。

## 可用的追踪器

Ultralytics YOLO支持以下追踪算法。可以通过传递相关的YAML配置文件如`tracker=tracker_type.yaml`来启用：

* [BoT-SORT](https://github.com/NirAharon/BoT-SORT) - 使用 `botsort.yaml` 启用此追踪器。
* [ByteTrack](https://github.com/ifzhang/ByteTrack) - 使用 `bytetrack.yaml` 启用此追踪器。

默认追踪器是BoT-SORT。

## 追踪

要在视频流中运行追踪器，请使用已训练的检测、分割或姿态模型，例如YOLOv8n、YOLOv8n-seg和YOLOv8n-pose。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载官方或自定义模型
        model = YOLO('yolov8n.pt')  # 加载一个官方的检测模型
        model = YOLO('yolov8n-seg.pt')  # 加载一个官方的分割模型
        model = YOLO('yolov8n-pose.pt')  # 加载一个官方的姿态模型
        model = YOLO('path/to/best.pt')  # 加载一个自定义训练的模型

        # 使用模型进行追踪
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True)  # 使用默认追踪器进行追踪
        results = model.track(source="https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml")  # 使用ByteTrack追踪器进行追踪
        ```

    === "CLI"

        ```bash
        # 使用命令行界面进行各种模型的追踪
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4"  # 官方检测模型
        yolo track model=yolov8n-seg.pt source="https://youtu.be/LNwODJXcvt4"  # 官方分割模型
        yolo track model=yolov8n-pose.pt source="https://youtu.be/LNwODJXcvt4"  # 官方姿态模型
        yolo track model=path/to/best.pt source="https://youtu.be/LNwODJXcvt4"  # 自定义训练模型

        # 使用ByteTrack追踪器进行追踪
        yolo track model=path/to/best.pt tracker="bytetrack.yaml"
        ```

如上所述，Detect、Segment和Pose模型在视频或流媒体源上运行时均可进行追踪。

## 配置

### 追踪参数

追踪配置与预测模式共享一些属性，如`conf`、`iou`和`show`。有关进一步配置，请参见[预测](https://docs.ultralytics.com/modes/predict/)模型页面。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 配置追踪参数并运行追踪器
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", conf=0.3, iou=0.5, show=True)
        ```

    === "CLI"

        ```bash
        # 使用命令行界面配置追踪参数并运行追踪器
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" conf=0.3, iou=0.5 show
        ```

### 选择追踪器

Ultralytics还允许您使用修改后的追踪器配置文件。要执行此操作，只需从[ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)中复制一个追踪器配置文件（例如，`custom_tracker.yaml`）并根据您的需求修改任何配置（除了`tracker_type`）。

!!! Example "示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型并使用自定义配置文件运行追踪器
        model = YOLO('yolov8n.pt')
        results = model.track(source="https://youtu.be/LNwODJXcvt4", tracker='custom_tracker.yaml')
        ```

    === "CLI"

        ```bash
        # 使用命令行界面加载模型并使用自定义配置文件运行追踪器
        yolo track model=yolov8n.pt source="https://youtu.be/LNwODJXcvt4" tracker='custom_tracker.yaml'
        ```

有关追踪参数的全面列表，请参考[ultralytics/cfg/trackers](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/cfg/trackers)页面。

## Python示例

### 持续追踪循环

这是一个使用OpenCV（`cv2`）和YOLOv8在视频帧上运行物体追踪的Python脚本。此脚本假设您已经安装了必要的包（`opencv-python`和`ultralytics`）。参数`persist=True`告诉追踪器当前的图像或帧是序列中的下一个，并且期望在当前图像中从上一个图像中获得追踪路径。

!!! Example "带追踪功能的流循环"

    ```python
    import cv2
    from ultralytics import YOLO

    # 加载YOLOv8模型
    model = YOLO('yolov8n.pt')

    # 打开视频文件
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
            results = model.track(frame, persist=True)

            # 在帧上展示结果
            annotated_frame = results[0].plot()

            # 展示带注释的帧
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # 如果按下'q'则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果视频结束则退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

请注意从`model(frame)`更改为`model.track(frame)`的变化，这使能够启用物体追踪而不只是简单的检测。这个修改的脚本将在视频的每一帧上运行追踪器，可视化结果，并在窗口中显示它们。通过按'q'可以退出循环。

### 随时间绘制追踪路径

在连续帧上可视化物体追踪路径可以提供有关视频中检测到的物体的运动模式和行为的有价值的洞见。使用Ultralytics YOLOv8，绘制这些路径是一个无缝且高效的过程。

在以下示例中，我们演示了如何利用YOLOv8的追踪功能在多个视频帧上绘制检测物体的移动。这个脚本涉及打开视频文件、逐帧读取，并使用YOLO模型识别并追踪各种物体。通过保留检测到的边界框的中心点并连接它们，我们可以绘制表示跟踪物体路径的线条。

!!! Example "在多个视频帧上绘制追踪路径"

    ```python
    from collections import defaultdict

    import cv2
    import numpy as np

    from ultralytics import YOLO

    # 加载YOLOv8模型
    model = YOLO('yolov8n.pt')

    # 打开视频文件
    video_path = "path/to/video.mp4"
    cap = cv2.VideoCapture(video_path)

    # 存储追踪历史
    track_history = defaultdict(lambda: [])

    # 循环遍历视频帧
    while cap.isOpened():
        # 从视频读取一帧
        success, frame = cap.read()

        if success:
            # 在帧上运行YOLOv8追踪，持续追踪帧间的物体
            results = model.track(frame, persist=True)

            # 获取框和追踪ID
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # 在帧上展示结果
            annotated_frame = results[0].plot()

            # 绘制追踪路径
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                track.append((float(x), float(y)))  # x, y中心点
                if len(track) > 30:  # 在90帧中保留90个追踪点
                    track.pop(0)

                # 绘制追踪线
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)

            # 展示带注释的帧
            cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # 如果按下'q'则退出循环
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # 如果视频结束则退出循环
            break

    # 释放视频捕获对象并关闭显示窗口
    cap.release()
    cv2.destroyAllWindows()
    ```

### 多线程追踪

多线程追踪提供了同时在多个视频流上运行物体追踪的能力。当处理多个视频输入，例如来自多个监控摄像头时，这一功能特别有用，其中并发处理可以大大提高效率和性能。

在提供的Python脚本中，我们利用Python的`threading`模块来同时运行多个追踪器实例。每个线程负责在一个视频文件上运行追踪器，所有线程在后台同时运行。

为了确保每个线程接收到正确的参数（视频文件、要使用的模型和文件索引），我们定义了一个函数`run_tracker_in_thread`，它接受这些参数并包含主追踪循环。此函数逐帧读取视频，运行追踪器，并显示结果。

在这个例子中，两个不同的模型被使用：`yolov8n.pt`和`yolov8n-seg.pt`，每个模型都在不同的视频文件中追踪物体。视频文件分别指定在`video_file1`和`video_file2`中。

在`threading.Thread`中参数`daemon=True`表示，这些线程会在主程序结束时关闭。然后我们用`start()`来开始线程，并使用`join()`来使主线程等待，直到两个追踪线程都结束。

最后，在所有线程完成任务后，使用`cv2.destroyAllWindows()`关闭显示结果的窗口。
