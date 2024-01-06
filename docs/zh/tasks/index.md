---
comments: true
description: 了解 YOLOv8 能够执行的基础计算机视觉任务，包括检测、分割、分类和姿态估计。理解它们在你的 AI 项目中的应用。
keywords: Ultralytics, YOLOv8, 检测, 分割, 分类, 姿态估计, AI 框架, 计算机视觉任务
---

# Ultralytics YOLOv8 任务

<br>
<img width="1024" src="https://raw.githubusercontent.com/ultralytics/assets/main/im/banner-tasks.png" alt="Ultralytics YOLO 支持的任务">

YOLOv8 是一个支持多种计算机视觉**任务**的 AI 框架。该框架可用于执行[检测](detect.md)、[分割](segment.md)、[分类](classify.md)和[姿态](pose.md)估计。每项任务都有不同的目标和用例。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/NAs-cfq9BDw"
    title="YouTube 视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong>探索 Ultralytics YOLO 任务：对象检测、分割、追踪和姿态估计。
</p>

## [检测](detect.md)

检测是 YOLOv8 支持的主要任务。它涉及在图像或视频帧中检测对象并围绕它们绘制边界框。侦测到的对象根据其特征被归类到不同的类别。YOLOv8 能够在单个图像或视频帧中检测多个对象，具有高准确性和速度。

[检测示例](detect.md){ .md-button }

## [分割](segment.md)

分割是一项涉及将图像分割成基于图像内容的不同区域的任务。每个区域根据其内容被分配一个标签。该任务在应用程序中非常有用，如图像分割和医学成像。YOLOv8 使用 U-Net 架构的变体来执行分割。

[分割示例](segment.md){ .md-button }

## [分类](classify.md)

分类是一项涉及将图像归类为不同类别的任务。YOLOv8 可用于根据图像内容对图像进行分类。它使用 EfficientNet 架构的变体来执行分类。

[分类示例](classify.md){ .md-button }

## [姿态](pose.md)

姿态/关键点检测是一项涉及在图像或视频帧中检测特定点的任务。这些点被称为关键点，用于跟踪移动或姿态估计。YOLOv8 能够在图像或视频帧中准确迅速地检测关键点。

[姿态示例](pose.md){ .md-button }

## 结论

YOLOv8 支持多个任务，包括检测、分割、分类和关键点检测。这些任务都具有不同的目标和用例。通过理解这些任务之间的差异，您可以为您的计算机视觉应用选择合适的任务。
