---
comments: true
description: 探索来自Ultralytics的最前沿的Segment Anything Model (SAM)，它可以进行实时图像分割。了解其可提示分割、零样本性能以及如何使用它。
keywords: Ultralytics，图像分割，Segment Anything Model，SAM，SA-1B数据集，实时性能，零样本转移，目标检测，图像分析，机器学习
---

# Segment Anything Model (SAM)

欢迎来到使用Segment Anything Model (SAM) 进行图像分割的前沿。这个革命性的模型通过引入可以提示的实时图像分割，树立了领域新的标准。

## SAM的介绍：Segment Anything Model

Segment Anything Model (SAM) 是一种先进的图像分割模型，可以进行可提示的分割，为图像分析任务提供了前所未有的灵活性。SAM 是Segment Anything 项目的核心，该项目引入了一种新颖的模型、任务和图像分割数据集。

SAM 先进的设计允许它在没有先验知识的情况下适应新的图像分布和任务，这个特点被称为零样本转移。SAM 在包含11亿个掩模的SA-1B数据集上进行训练，该数据集包含超过1100万张精心策划的图像，SAM 在零样本任务中表现出色，许多情况下超过了之前的完全监督结果。

![数据集示例图像](https://user-images.githubusercontent.com/26833433/238056229-0e8ffbeb-f81a-477e-a490-aff3d82fd8ce.jpg)
从我们新引入的SA-1B数据集中选择的示例图像，显示了覆盖的掩模。SA-1B包含了1100万个多样化、高分辨率、许可的图像和11亿个高质量的分割掩模。这些掩模由SAM完全自动地进行了注释，经过人工评级和大量实验的验证，它们具有高质量和多样性。图像按每个图像的掩模数量进行分组以进行可视化（平均每个图像有∼100个掩模）。

## Segment Anything Model (SAM) 的主要特点

- **可提示的分割任务**：SAM 的设计考虑了可提示的分割任务，它可以从给定的提示中生成有效的分割掩模，例如指示对象的空间或文本线索。
- **先进的架构**：Segment Anything Model 使用强大的图像编码器、提示编码器和轻量的掩模解码器。这种独特的架构使得分段任务中的提示灵活、实时掩模计算和模糊感知成为可能。
- **SA-1B 数据集**：由Segment Anything 项目引入的 SA-1B 数据集包含超过11亿个掩模的1100万张图像。作为迄今为止最大的分割数据集，它为 SAM 提供了一个多样化的大规模训练数据源。
- **零样本性能**：SAM 在各种分割任务中展现出出色的零样本性能，使得它成为一个可以立即使用的工具，对于各种应用来说，对提示工程的需求很小。

如果您想了解更多关于Segment Anything Model 和 SA-1B 数据集的详细信息，请访问[Segment Anything 网站](https://segment-anything.com)并查看研究论文[Segment Anything](https://arxiv.org/abs/2304.02643)。

## 可用模型、支持的任务和操作模式

这个表格展示了可用模型及其特定的预训练权重，它们支持的任务，以及它们与不同操作模式（[Inference](../modes/predict.md)、[Validation](../modes/val.md)、[Training](../modes/train.md) 和 [Export](../modes/export.md)）的兼容性，用 ✅ 表示支持的模式，用 ❌ 表示不支持的模式。

| 模型类型  | 预训练权重 | 支持的任务                      | 推断 | 验证 | 训练 | 导出 |
| --------- | ---------- | ------------------------------- | ---- | ---- | ---- | ---- |
| SAM base  | `sam_b.pt` | [实例分割](../tasks/segment.md) | ✅   | ❌   | ❌   | ✅   |
| SAM large | `sam_l.pt` | [实例分割](../tasks/segment.md) | ✅   | ❌   | ❌   | ✅   |

## 如何使用 SAM: 图像分割的多功能和强大

Segment Anything Model 可以用于多种下游任务，超越训练数据的范围。这包括边缘检测，目标提案生成，实例分割和初步的文本到掩模预测。通过 prompt 工程，SAM 可以快速适应新的任务和数据分布，以零样本的方式，确立其作为图像分割需求的多功能和强大工具。

### SAM 预测示例

!!! Example "使用提示进行分割"

    使用给定的提示对图像进行分割。

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("sam_b.pt")

        # 显示模型信息（可选）
        model.info()

        # 使用边界框提示进行推断
        model("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])

        # 使用点提示进行推断
        model("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
        ```

!!! Example "分割整个图像"

    分割整个图像。

    === "Python"

        ```python
        from ultralytics import SAM

        # 加载模型
        model = SAM("sam_b.pt")

        # 显示模型信息（可选）
        model.info()

        # 进行推断
        model("path/to/image.jpg")
        ```

    === "CLI"

        ```bash
        # 使用 SAM 模型进行推断
        yolo predict model=sam_b.pt source=path/to/image.jpg
        ```

- 这里的逻辑是，如果您没有传入任何提示（边界框/点/掩模），则对整个图像进行分割。

!!! Example "SAMPredictor 示例"

    这种方法可以设置图像一次，然后多次运行提示推断，而无需多次运行图像编码器。

    === "提示推断"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # 创建 SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # 设置图像
        predictor.set_image("ultralytics/assets/zidane.jpg")  # 使用图像文件设置
        predictor.set_image(cv2.imread("ultralytics/assets/zidane.jpg"))  # 使用 np.ndarray 设置
        results = predictor(bboxes=[439, 437, 524, 709])
        results = predictor(points=[900, 370], labels=[1])

        # 重置图像
        predictor.reset_image()
        ```

    通过附加参数对整个图像分割。

    === "分割整个图像"

        ```python
        from ultralytics.models.sam import Predictor as SAMPredictor

        # 创建 SAMPredictor
        overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model="mobile_sam.pt")
        predictor = SAMPredictor(overrides=overrides)

        # 使用附加参数进行分割整个图像
        results = predictor(source="ultralytics/assets/zidane.jpg", crop_n_layers=1, points_stride=64)
        ```

- 更多关于`分割整个图像`的附加参数，请查看[`Predictor/generate` 参考](../../../reference/models/sam/predict.md)。

## SAM 与 YOLOv8 的对比

在这里，我们将 Meta 最小的 SAM 模型 SAM-b 与 Ultralytics 的最小分割模型 [YOLOv8n-seg](../tasks/segment.md) 进行对比：

| 模型                                           | 大小                      | 参数                     | 速度 (CPU)                 |
| ---------------------------------------------- | ------------------------- | ------------------------ | -------------------------- |
| Meta's SAM-b                                   | 358 MB                    | 94.7 M                   | 51096 ms/im                |
| [MobileSAM](mobile-sam.md)                     | 40.7 MB                   | 10.1 M                   | 46122 ms/im                |
| [FastSAM-s](fast-sam.md) with YOLOv8 backbone  | 23.7 MB                   | 11.8 M                   | 115 ms/im                  |
| Ultralytics [YOLOv8n-seg](../tasks/segment.md) | **6.7 MB** (缩小了53.4倍) | **3.4 M** (缩小了27.9倍) | **59 ms/im** (加速了866倍) |

这个对比显示了不同模型之间的模型大小和速度上数量级的差异。虽然 SAM 提供了自动分割的独特能力，但它不是与 YOLOv8 分割模型直接竞争的产品，后者体积更小、速度更快、效率更高。

在配备有16GB RAM的2023年 Apple M2 MacBook 上进行了测试。要重现这个测试：

!!! Example "示例"

    === "Python"
        ```python
        from ultralytics import SAM, YOLO, FastSAM

        # 分析 SAM-b
        model = SAM("sam_b.pt")
        model.info()
        model("ultralytics/assets")

        # 分析 MobileSAM
        model = SAM("mobile_sam.pt")
        model.info()
        model("ultralytics/assets")

        # 分析 FastSAM-s
        model = FastSAM("FastSAM-s.pt")
        model.info()
        model("ultralytics/assets")

        # 分析 YOLOv8n-seg
        model = YOLO("yolov8n-seg.pt")
        model.info()
        model("ultralytics/assets")
        ```

## 自动注释：创建分割数据集的快速路径

自动注释是 SAM 的一个关键功能，它允许用户使用预训练的检测模型生成一个[分割数据集](https://docs.ultralytics.com/datasets/segment)。这个功能可以通过自动生成大量图像的准确注释，绕过耗时的手动标注过程，从而快速获得高质量的分割数据集。

### 使用检测模型生成分割数据集

要使用Ultralytics框架对数据集进行自动注释，可以使用如下所示的 `auto_annotate` 函数：

!!! Example "示例"

    === "Python"
        ```python
        from ultralytics.data.annotator import auto_annotate

        auto_annotate(data="path/to/images", det_model="yolov8x.pt", sam_model="sam_b.pt")
        ```

| 参数       | 类型            | 描述                                                                | 默认值       |
| ---------- | --------------- | ------------------------------------------------------------------- | ------------ |
| data       | str             | 包含要进行注释的图像的文件夹的路径。                                |              |
| det_model  | str, 可选       | 预训练的 YOLO 检测模型，默认为 'yolov8x.pt'。                       | 'yolov8x.pt' |
| sam_model  | str, 可选       | 预训练的 SAM 分割模型，默认为 'sam_b.pt'。                          | 'sam_b.pt'   |
| device     | str, 可选       | 在其上运行模型的设备，默认为空字符串（如果可用，则为 CPU 或 GPU）。 |              |
| output_dir | str, None, 可选 | 保存注释结果的目录。默认为与 'data' 目录同级的 'labels' 目录。      | None         |

`auto_annotate` 函数接受您图像的路径，并提供了可选的参数用于指定预训练的检测和 SAM 分割模型、运行模型的设备，以及保存注释结果的输出目录。

使用预训练模型进行自动注释可以大大减少创建高质量分割数据集所需的时间和工作量。这个功能特别对于处理大量图像集合的研究人员和开发人员非常有益，因为它允许他们专注于模型的开发和评估，而不是手动注释。

## 引用和鸣谢

如果您在研究或开发中发现 SAM 对您有用，请考虑引用我们的论文：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @misc{kirillov2023segment,
              title={Segment Anything},
              author={Alexander Kirillov and Eric Mintun and Nikhila Ravi and Hanzi Mao and Chloe Rolland and Laura Gustafson and Tete Xiao and Spencer Whitehead and Alexander C. Berg and Wan-Yen Lo and Piotr Dollár and Ross Girshick},
              year={2023},
              eprint={2304.02643},
              archivePrefix={arXiv},
              primaryClass={cs.CV}
        }
        ```

我们要向 Meta AI 表示感谢，感谢他们为计算机视觉社区创建和维护了这个宝贵的资源。

_keywords: Segment Anything，Segment Anything Model，SAM，Meta SAM，图像分割，可提示分割，零样本性能，SA-1B数据集，先进架构，自动注释，Ultralytics，预训练模型，SAM base，SAM large，实例分割，计算机视觉，AI，人工智能，机器学习，数据注释，分割掩模，检测模型，YOLO检测模型，bibtex，Meta AI。_
