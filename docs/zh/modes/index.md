---
comments: true
description: 从训练到跟踪，充分利用Ultralytics的YOLOv8。获取支持的每种模式的见解和示例，包括验证、导出和基准测试。
keywords: Ultralytics, YOLOv8, 机器学习, 目标检测, 训练, 验证, 预测, 导出, 跟踪, 基准测试
---

# Ultralytics YOLOv8 模式

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO生态系统及整合">

## 简介

Ultralytics YOLOv8不仅仅是另一个目标检测模型；它是一个多功能框架，旨在涵盖机器学习模型的整个生命周期——从数据摄取和模型训练到验证、部署和实际跟踪。每种模式都服务于一个特定的目的，并设计为提供您在不同任务和用例中所需的灵活性和效率。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/j8uQc0qB91s?si=dhnGKgqvs7nPgeaM"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看：</strong> Ultralytics模式教程：训练、验证、预测、导出和基准测试。
</p>

### 模式概览

理解Ultralytics YOLOv8所支持的不同**模式**对于充分利用您的模型至关重要：

- **训练（Train）**模式：在自定义或预加载的数据集上微调您的模型。
- **验证（Val）**模式：训练后进行校验，以验证模型性能。
- **预测（Predict）**模式：在真实世界数据上释放模型的预测能力。
- **导出（Export）**模式：以各种格式使模型准备就绪，部署至生产环境。
- **跟踪（Track）**模式：将您的目标检测模型扩展到实时跟踪应用中。
- **基准（Benchmark）**模式：在不同部署环境中分析模型的速度和准确性。

本综合指南旨在为您提供每种模式的概览和实用见解，帮助您充分发挥YOLOv8的全部潜力。

## [训练](train.md)

训练模式用于在自定义数据集上训练YOLOv8模型。在此模式下，模型将使用指定的数据集和超参数进行训练。训练过程包括优化模型的参数，使其能够准确预测图像中对象的类别和位置。

[训练示例](train.md){ .md-button }

## [验证](val.md)

验证模式用于训练YOLOv8模型后进行验证。在此模式下，模型在验证集上进行评估，以衡量其准确性和泛化能力。此模式可以用来调整模型的超参数，以改善其性能。

[验证示例](val.md){ .md-button }

## [预测](predict.md)

预测模式用于使用训练好的YOLOv8模型在新图像或视频上进行预测。在此模式下，模型从检查点文件加载，用户可以提供图像或视频以执行推理。模型预测输入图像或视频中对象的类别和位置。

[预测示例](predict.md){ .md-button }

## [导出](export.md)

导出模式用于将YOLOv8模型导出为可用于部署的格式。在此模式下，模型被转换为其他软件应用或硬件设备可以使用的格式。当模型部署到生产环境时，此模式十分有用。

[导出示例](export.md){ .md-button }

## [跟踪](track.md)

跟踪模式用于使用YOLOv8模型实时跟踪对象。在此模式下，模型从检查点文件加载，用户可以提供实时视频流以执行实时对象跟踪。此模式适用于监控系统或自动驾驶汽车等应用。

[跟踪示例](track.md){ .md-button }

## [基准](benchmark.md)

基准模式用于对YOLOv8的各种导出格式的速度和准确性进行评估。基准提供了有关导出格式大小、其针对目标检测、分割和姿态的`mAP50-95`指标，或针对分类的`accuracy_top5`指标，以及每张图像跨各种导出格式（如ONNX、OpenVINO、TensorRT等）的推理时间（以毫秒为单位）的信息。此信息可以帮助用户根据对速度和准确性的具体需求，选择最佳的导出格式。

[基准示例](benchmark.md){ .md-button }
