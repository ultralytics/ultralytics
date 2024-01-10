---
comments: true
description: 使用Ultralytics YOLO训练YOLOv8模型的逐步指南，包括单GPU和多GPU训练示例
keywords: Ultralytics, YOLOv8, YOLO, 目标检测, 训练模式, 自定义数据集, GPU训练, 多GPU, 超参数, CLI示例, Python示例
---

# 使用Ultralytics YOLO进行模型训练

<img width="1024" src="https://github.com/ultralytics/assets/raw/main/yolov8/banner-integrations.png" alt="Ultralytics YOLO生态系统与集成">

## 引言

训练深度学习模型涉及向其输入数据并调整参数，以便准确预测。Ultralytics YOLOv8的训练模式旨在有效高效地训练目标检测模型，充分利用现代硬件功能。本指南旨在涵盖使用YOLOv8的强大功能集训练自己模型的所有细节。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/LNwODJXcvt4?si=7n1UvGRLSd9p5wKs"
    title="YouTube视频播放器" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> 如何在Google Colab中用你的自定义数据集训练一个YOLOv8模型。
</p>

## 为什么选择Ultralytics YOLO进行训练？

以下是选择YOLOv8训练模式的一些有力理由：

- **效率:** 充分利用您的硬件资源，无论您是使用单GPU设置还是跨多个GPU扩展。
- **多功能:** 除了可随时获取的数据集（如COCO、VOC和ImageNet）之外，还可以对自定义数据集进行训练。
- **用户友好:** 简单而强大的CLI和Python接口，为您提供直接的训练体验。
- **超参数灵活性:** 可定制的广泛超参数范围，以微调模型性能。

### 训练模式的关键特性

以下是YOLOv8训练模式的一些显著特点：

- **自动数据集下载:** 标准数据集如COCO、VOC和ImageNet将在首次使用时自动下载。
- **多GPU支持:** 无缝地跨多个GPU扩展您的训练工作，以加快过程。
- **超参数配置:** 通过YAML配置文件或CLI参数修改超参数的选项。
- **可视化和监控:** 实时跟踪训练指标并可视化学习过程，以获得更好的洞察力。

!!! 小贴士 "小贴士"

    * 如COCO、VOC、ImageNet等YOLOv8数据集在首次使用时会自动下载，即 `yolo train data=coco.yaml`

## 使用示例

在COCO128数据集上训练YOLOv8n模型100个时期，图像大小为640。可以使用`device`参数指定训练设备。如果没有传递参数，并且有可用的GPU，则将使用GPU `device=0`，否则将使用`device=cpu`。有关完整列表的训练参数，请参见下面的参数部分。

!!! Example "单GPU和CPU训练示例"

    设备将自动确定。如果有可用的GPU，那么将使用它，否则将在CPU上开始训练。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载一个模型
        model = YOLO('yolov8n.yaml')  # 从YAML建立一个新模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）
        model = YOLO('yolov8n.yaml').load('yolov8n.pt')  # 从YAML建立并转移权重

        # 训练模型
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640)
        ```

    === "CLI"

        ```bash
        # 从YAML构建新模型，从头开始训练
        yolo detect train data=coco128.yaml model=yolov8n.yaml epochs=100 imgsz=640

        # 从预训练*.pt模型开始训练
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640

        # 从YAML构建一个新模型，转移预训练权重，然后开始训练
        yolo detect train data=coco128.yaml model=yolov8n.yaml pretrained=yolov8n.pt epochs=100 imgsz=640
        ```

### 多GPU训练

多GPU训练通过在多个GPU上分布训练负载，实现对可用硬件资源的更有效利用。无论是通过Python API还是命令行界面，都可以使用此功能。 若要启用多GPU训练，请指定您希望使用的GPU设备ID。

!!! Example "多GPU训练示例"

    要使用2个GPU进行训练，请使用CUDA设备0和1，使用以下命令。根据需要扩展到更多GPU。

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

        # 使用2个GPU训练模型
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device=[0, 1])
        ```

    === "CLI"

        ```bash
        # 使用GPU 0和1从预训练*.pt模型开始训练
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=0,1
        ```

### 苹果M1和M2 MPS训练

通过Ultralytics YOLO模型集成对Apple M1和M2芯片的支持，现在可以在使用强大的Metal性能着色器（MPS）框架的设备上训练模型。MPS为在Apple的定制硅上执行计算和图像处理任务提供了一种高性能的方法。

要在Apple M1和M2芯片上启用训练，您应该在启动训练过程时将设备指定为'mps'。以下是Python和命令行中如何做到这点的示例：

!!! Example "MPS训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('yolov8n.pt')  # 加载预训练模型（推荐用于训练）

        # 使用2个GPU训练模型
        results = model.train(data='coco128.yaml', epochs=100, imgsz=640, device='mps')
        ```

    === "CLI"

        ```bash
        # 使用GPU 0和1从预训练*.pt模型开始训练
        yolo detect train data=coco128.yaml model=yolov8n.pt epochs=100 imgsz=640 device=mps
        ```

利用M1/M2芯片的计算能力，这使得训练任务的处理更加高效。有关更详细的指南和高级配置选项，请参阅[PyTorch MPS文档](https://pytorch.org/docs/stable/notes/mps.html)。

### 恢复中断的训练

在处理深度学习模型时，从之前保存的状态恢复训练是一个关键特性。在各种情况下，这可能很方便，比如当训练过程意外中断，或者当您希望用新数据或更多时期继续训练模型时。

恢复训练时，Ultralytics YOLO将加载最后保存的模型的权重，并恢复优化器状态、学习率调度器和时期编号。这允许您无缝地从离开的地方继续训练过程。

在Ultralytics YOLO中，您可以通过在调用`train`方法时将`resume`参数设置为`True`并指定包含部分训练模型权重的`.pt`文件路径来轻松恢复训练。

下面是使用Python和命令行恢复中断训练的示例：

!!! Example "恢复训练示例"

    === "Python"

        ```python
        from ultralytics import YOLO

        # 加载模型
        model = YOLO('path/to/last.pt')  # 加载部分训练的模型

        # 恢复训练
        results = model.train(resume=True)
        ```

    === "CLI"

        ```bash
        # 恢复中断的训练
        yolo train resume model=path/to/last.pt
        ```

通过设置`resume=True`，`train`函数将从'path/to/last.pt'文件中存储的状态继续训练。如果省略`resume`参数或将其设置为`False`，`train`函数将启动新的训练会话。

请记住，默认情况下，检查点会在每个时期结束时保存，或者使用`save_period`参数以固定间隔保存，因此您必须至少完成1个时期才能恢复训练运行。

## 参数

YOLO模型的训练设置是指用于对数据集进行模型训练的各种超参数和配置。这些设置会影响模型的性能、速度和准确性。一些常见的YOLO训练设置包括批大小、学习率、动量和权重衰减。其他可能影响训练过程的因素包括优化器的选择、损失函数的选择以及训练数据集的大小和组成。仔细调整和实验这些设置以实现给定任务的最佳性能是非常重要的。

| 键                 | 值        | 描述                                                                  |
|-------------------|----------|---------------------------------------------------------------------|
| `model`           | `None`   | 模型文件路径，例如 yolov8n.pt, yolov8n.yaml                                  |
| `data`            | `None`   | 数据文件路径，例如 coco128.yaml                                              |
| `epochs`          | `100`    | 训练的轮次数量                                                             |
| `patience`        | `50`     | 早停训练的等待轮次                                                           |
| `batch`           | `16`     | 每批图像数量（-1为自动批大小）                                                    |
| `imgsz`           | `640`    | 输入图像的大小，以整数表示                                                       |
| `save`            | `True`   | 保存训练检查点和预测结果                                                        |
| `save_period`     | `-1`     | 每x轮次保存检查点（如果<1则禁用）                                                  |
| `cache`           | `False`  | True/ram, disk 或 False。使用缓存加载数据                                     |
| `device`          | `None`   | 运行设备，例如 cuda device=0 或 device=0,1,2,3 或 device=cpu                 |
| `workers`         | `8`      | 数据加载的工作线程数（如果DDP则为每个RANK）                                           |
| `project`         | `None`   | 项目名称                                                                |
| `name`            | `None`   | 实验名称                                                                |
| `exist_ok`        | `False`  | 是否覆盖现有实验                                                            |
| `pretrained`      | `True`   | (bool 或 str) 是否使用预训练模型（bool）或从中加载权重的模型（str）                         |
| `optimizer`       | `'auto'` | 使用的优化器，选择范围=[SGD, Adam, Adamax, AdamW, NAdam, RAdam, RMSProp, auto] |
| `verbose`         | `False`  | 是否打印详细输出                                                            |
| `seed`            | `0`      | 随机种子，用于可重复性                                                         |
| `deterministic`   | `True`   | 是否启用确定性模式                                                           |
| `single_cls`      | `False`  | 将多类数据作为单类训练                                                         |
| `rect`            | `False`  | 矩形训练，每批为最小填充整合                                                      |
| `cos_lr`          | `False`  | 使用余弦学习率调度器                                                          |
| `close_mosaic`    | `10`     | (int) 最后轮次禁用马赛克增强（0为禁用）                                             |
| `resume`          | `False`  | 从最后检查点恢复训练                                                          |
| `amp`             | `True`   | 自动混合精度（AMP）训练，选择范围=[True, False]                                    |
| `fraction`        | `1.0`    | 训练的数据集比例（默认为1.0，即训练集中的所有图像）                                         |
| `profile`         | `False`  | 在训练期间为记录器分析ONNX和TensorRT速度                                          |
| `freeze`          | `None`   | (int 或 list, 可选) 在训练期间冻结前n层，或冻结层索引列表                                |
| `lr0`             | `0.01`   | 初始学习率（例如 SGD=1E-2, Adam=1E-3）                                       |
| `lrf`             | `0.01`   | 最终学习率 (lr0 * lrf)                                                   |
| `momentum`        | `0.937`  | SGD动量/Adam beta1                                                    |
| `weight_decay`    | `0.0005` | 优化器权重衰减5e-4                                                         |
| `warmup_epochs`   | `3.0`    | 热身轮次（小数ok）                                                          |
| `warmup_momentum` | `0.8`    | 热身初始动量                                                              |
| `warmup_bias_lr`  | `0.1`    | 热身初始偏差lr                                                            |
| `box`             | `7.5`    | 框损失增益                                                               |
| `cls`             | `0.5`    | cls损失增益（根据像素缩放）                                                     |
| `dfl`             | `1.5`    | dfl损失增益                                                             |
| `pose`            | `12.0`   | 姿态损失增益（仅限姿态）                                                        |
| `kobj`            | `2.0`    | 关键点obj损失增益（仅限姿态）                                                    |
| `label_smoothing` | `0.0`    | 标签平滑（小数）                                                            |
| `nbs`             | `64`     | 标称批大小                                                               |
| `overlap_mask`    | `True`   | 训练期间掩码应重叠（仅限分割训练）                                                   |
| `mask_ratio`      | `4`      | 掩码降采样比率（仅限分割训练）                                                     |
| `dropout`         | `0.0`    | 使用dropout正则化（仅限分类训练）                                                |
| `val`             | `True`   | 训练期间验证/测试                                                           |

## 记录

在训练YOLOv8模型时，跟踪模型随时间的性能变化可能非常有价值。这就是记录发挥作用的地方。Ultralytics的YOLO提供对三种类型记录器的支持 - Comet、ClearML和TensorBoard。

要使用记录器，请在上面的代码片段中的下拉菜单中选择它并运行。所选的记录器将被安装和初始化。

### Comet

[Comet](https://www.comet.ml/site/)是一个平台，允许数据科学家和开发人员跟踪、比较、解释和优化实验和模型。它提供了实时指标、代码差异和超参数跟踪等功能。

使用Comet：

!!! Example "示例"

    === "Python"
        ```python
        # pip install comet_ml
        import comet_ml

        comet_ml.init()
        ```

记得在他们的网站上登录您的Comet账户并获取您的API密钥。您需要将此添加到您的环境变量或脚本中，以记录您的实验。

### ClearML

[ClearML](https://www.clear.ml/) 是一个开源平台，自动跟踪实验并帮助有效共享资源。它旨在帮助团队更有效地管理、执行和复现他们的ML工作。

使用ClearML：

!!! Example "示例"

    === "Python"
        ```python
        # pip install clearml
        import clearml

        clearml.browser_login()
        ```

运行此脚本后，您需要在浏览器中登录您的ClearML账户并认证您的会话。

### TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) 是TensorFlow的可视化工具包。它允许您可视化TensorFlow图表，绘制有关图表执行的定量指标，并展示通过它的附加数据，如图像。

在[Google Colab](https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb)中使用TensorBoard：

!!! Example "示例"

    === "CLI"
        ```bash
        load_ext tensorboard
        tensorboard --logdir ultralytics/runs  # 替换为'runs'目录
        ```

在本地使用TensorBoard，运行下面的命令并在 http://localhost:6006/ 查看结果。

!!! Example "示例"

    === "CLI"
        ```bash
        tensorboard --logdir ultralytics/runs  # 替换为'runs'目录
        ```

这将加载TensorBoard并将其定向到保存训练日志的目录。

在设置好日志记录器后，您可以继续进行模型训练。所有训练指标将自动记录在您选择的平台中，您可以访问这些日志以监控模型随时间的表现，比较不同模型，并识别改进的领域。
