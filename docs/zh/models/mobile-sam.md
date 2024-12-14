---
comments: true
description: 了解有关MobileSAM的更多信息，包括其实现、与原始SAM的比较，以及在Ultralytics框架中如何下载和测试它。立即改进您的移动应用程序。
keywords: MobileSAM, Ultralytics, SAM, 移动应用, Arxiv, GPU, API, 图像编码器, 蒙版解码器, 模型下载, 测试方法
---

![MobileSAM Logo](https://github.com/ChaoningZhang/MobileSAM/blob/master/assets/logo2.png?raw=true)

# 移动端细分模型（MobileSAM）

MobileSAM 论文现在可以在 [arXiv](https://arxiv.org/pdf/2306.14289.pdf) 上找到。

可以通过此 [演示链接](https://huggingface.co/spaces/dhkim2810/MobileSAM) 访问在 CPU 上运行的 MobileSAM 演示。在 Mac i5 CPU 上，性能大约需要 3 秒。在 Hugging Face 的演示中，界面和性能较低的 CPU 导致响应较慢，但它仍然能有效地工作。

MobileSAM 已在 Grounding-SAM、AnyLabeling 和 Segment Anything in 3D 等多个项目中实施。您可以在 [Grounding-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything)、[AnyLabeling](https://github.com/vietanhdev/anylabeling) 和 [Segment Anything in 3D](https://github.com/Jumpat/SegmentAnythingin3D) 上找到这些项目。

MobileSAM 使用单个 GPU 在不到一天的时间内对 10 万个数据集（原始图像的 1%）进行训练。关于此训练的代码将在将来提供。

## 可用模型、支持的任务和操作模式

以下表格显示了可用模型及其具体的预训练权重，它们支持的任务以及与不同操作模式（[预测](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md) 和 [导出](../modes/export.md)）的兼容性，其中支持的模式用 ✅ 表示，不支持的模式用 ❌ 表示。

| 模型类型  | 预训练权重      | 支持的任务                      | 预测 | 验证 | 训练 | 导出 |
| --------- | --------------- | ------------------------------- | ---- | ---- | ---- | ---- |
| MobileSAM | `mobile_sam.pt` | [实例分割](../tasks/segment.md) | ✅   | ❌   | ❌   | ✅   |

## 从 SAM 迁移到 MobileSAM

由于 MobileSAM 保留了与原始 SAM 相同的流程，我们已将原始 SAM 的预处理、后处理和所有其他接口整合到 MobileSAM 中。因此，目前使用原始 SAM 的用户可以以最小的努力迁移到 MobileSAM。

MobileSAM 在性能上与原始 SAM 相当，并保留了相同的流程，只是更改了图像编码器。具体而言，我们用较小的 Tiny-ViT（5M）替换了原始的笨重的 ViT-H 编码器（632M）。在单个 GPU 上，MobileSAM 每张图片的运行时间约为 12 毫秒：图像编码器约 8 毫秒，蒙版解码器约 4 毫秒。

以下表格比较了基于 ViT 的图像编码器：

| 图像编码器 | 原始 SAM | MobileSAM |
| ---------- | -------- | --------- |
| 参数       | 611M     | 5M        |
| 速度       | 452ms    | 8ms       |

原始 SAM 和 MobileSAM 均使用相同的提示引导蒙版解码器：

| 蒙版解码器 | 原始 SAM | MobileSAM |
| ---------- | -------- | --------- |
| 参数       | 3.876M   | 3.876M    |
| 速度       | 4ms      | 4ms       |

以下是整个流程的比较：

| 整个流程（编码器+解码器） | 原始 SAM | MobileSAM |
| ------------------------- | -------- | --------- |
| 参数                      | 615M     | 9.66M     |
| 速度                      | 456ms    | 12ms      |

MobileSAM 和原始 SAM 的性能通过使用点和框作为提示进行演示。

![点作为提示的图像](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

![框作为提示的图像](https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/assets/mask_box.jpg?raw=true)

MobileSAM 的性能优于当前的 FastSAM，尺寸大约减小了 5 倍，速度快了约 7 倍。有关更多详细信息，请访问 [MobileSAM 项目页面](https://github.com/ChaoningZhang/MobileSAM)。

## 在 Ultralytics 中测试 MobileSAM

与原始 SAM 一样，我们在 Ultralytics 中提供了一种简单的测试方法，包括点提示和框提示的模式。

### 模型下载

您可以在 [这里](https://github.com/ChaoningZhang/MobileSAM/blob/master/weights/mobile_sam.pt) 下载模型。

### 点提示

!!! Example "示例"

    === "Python"
        ```python
        from ultralytics import SAM

        # 载入模型
        model = SAM("mobile_sam.pt")

        # 基于点提示预测一个分段
        model.predict("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])
        ```

### 框提示

!!! Example "示例"

    === "Python"
        ```python
        from ultralytics import SAM

        # 载入模型
        model = SAM("mobile_sam.pt")

        # 基于框提示预测一个分段
        model.predict("ultralytics/assets/zidane.jpg", bboxes=[439, 437, 524, 709])
        ```

我们使用相同的 API 实现了 `MobileSAM` 和 `SAM`。有关更多用法信息，请参阅 [SAM 页面](sam.md)。

## 引用和鸣谢

如果您在研究或开发工作中发现 MobileSAM 对您有用，请考虑引用我们的论文：

!!! Quote ""

    === "BibTeX"

        ```bibtex
        @article{mobile_sam,
          title={Faster Segment Anything: Towards Lightweight SAM for Mobile Applications},
          author={Zhang, Chaoning and Han, Dongshen and Qiao, Yu and Kim, Jung Uk and Bae, Sung Ho and Lee, Seungkyu and Hong, Choong Seon},
          journal={arXiv preprint arXiv:2306.14289},
          year={2023}
        }
