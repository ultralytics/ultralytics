---
comments: true
description: FastSAM是一种基于卷积神经网络的实时图像对象分割解决方案。它提供了卓越的用户交互功能、计算效率以及适用于多种视觉任务的特性。
keywords: FastSAM, 机器学习, 基于卷积神经网络的解决方案, 图像对象分割, 实时解决方案, Ultralytics, 视觉任务, 图像处理, 工业应用, 用户交互
---

# Fast Segment Anything Model（FastSAM）

Fast Segment Anything Model（FastSAM）是一种创新的实时卷积神经网络（CNN）模型，用于图像中的任意对象分割任务。该任务旨在根据各种可能的用户交互提示，对图像中的任意对象进行分割。FastSAM在保持具备竞争性能的同时，显著降低了计算需求，使其成为各种视觉任务的实用选择。

![Fast Segment Anything Model（FastSAM）架构概述](https://user-images.githubusercontent.com/26833433/248551984-d98f0f6d-7535-45d0-b380-2e1440b52ad7.jpg)

## 概述

FastSAM旨在解决[Segment Anything Model（SAM）](sam.md)的局限性，SAM是一种计算资源需求很高的Transformer模型。FastSAM将任意对象分割任务拆分为两个顺序阶段：所有实例分割和提示引导选择。第一阶段使用[YOLOv8-seg](../tasks/segment.md)生成图像中所有实例的分割掩码。在第二阶段，输出与提示对应的感兴趣区域。

## 主要特点

1. **实时解决方案：** FastSAM利用CNN的计算效率提供了图像中任意对象分割任务的实时解决方案，适用于需要快速结果的工业应用。

2. **高效和高性能：** FastSAM在显著降低计算和资源需求的同时，不会降低性能质量。它与SAM具有相当的性能，但计算资源大幅减少，能够实现实时应用。

3. **提示引导分割：** FastSAM可以通过各种可能的用户交互提示来分割图像中的任意对象，提供了不同场景下的灵活性和适应性。

4. **基于YOLOv8-seg：** FastSAM基于[YOLOv8-seg](../tasks/segment.md)，是一种配备实例分割分支的目标检测器。这使得它能够有效地生成图像中所有实例的分割掩码。

5. **基准测试中具有竞争力的结果：** 在MS COCO的对象提议任务中，FastSAM在单个NVIDIA RTX 3090上以显著更快的速度获得高分，与[SAM](sam.md)相比，显示出其效率和能力。

6. **实际应用：** 提出的方法以非常高的速度为大量视觉任务提供了一种新的实用解决方案，比当前方法快十几倍乃至数百倍。

7. **模型压缩的可行性：** FastSAM通过引入人工先验到结构中，展示了通过路径显著减少计算工作量的可行性，为通用视觉任务的大型模型架构开辟了新的可能性。

## 可用模型、支持的任务和操作模式

该表格列出了可用的模型及其特定的预训练权重，它们支持的任务以及它们与不同操作模式（如[推断](../modes/predict.md)、[验证](../modes/val.md)、[训练](../modes/train.md)和[导出](../modes/export.md)）的兼容性，由支持的模式用✅表示，不支持的模式用❌表示。

| 模型类型      | 预训练权重          | 支持的任务                       | 推断 | 验证 | 训练 | 导出 |
|-----------|----------------|-----------------------------|----|----|----|----|
| FastSAM-s | `FastSAM-s.pt` | [实例分割](../tasks/segment.md) | ✅  | ❌  | ❌  | ✅  |
| FastSAM-x | `FastSAM-x.pt` | [实例分割](../tasks/segment.md) | ✅  | ❌  | ❌  | ✅  |

## 用法示例

FastSAM模型很容易集成到Python应用程序中。Ultralytics提供了用户友好的Python API和CLI命令以简化开发。

### 预测用法

要对图像进行对象检测，可以使用下面的`predict`方法：

!!! Example "示例"

    === "Python"
        ```python
        from ultralytics import FastSAM
        from ultralytics.models.fastsam import FastSAMPrompt

        # 定义推断源
        source = 'path/to/bus.jpg'

        # 创建FastSAM模型
        model = FastSAM('FastSAM-s.pt')  # 或 FastSAM-x.pt

        # 在图像上运行推断
        everything_results = model(source, device='cpu', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

        # 准备Prompt Process对象
        prompt_process = FastSAMPrompt(source, everything_results, device='cpu')

        # Everything提示
        ann = prompt_process.everything_prompt()

        # Bbox默认形状[0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # 文本提示
        ann = prompt_process.text_prompt(text='a photo of a dog')

        # 点提示
        # 默认点[[0,0]] [[x1,y1],[x2,y2]]
        # 默认point_label [0] [1,0] 0：背景，1：前景
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output='./')
        ```

    === "CLI"
        ```bash
        # 加载FastSAM模型并使用该模型分割图像中的所有对象
        yolo segment predict model=FastSAM-s.pt source=path/to/bus.jpg imgsz=640
        ```

此片段演示了加载预训练模型并在图像上进行预测的简单性。

### 验证用法

可以采用以下方式对数据集上的模型进行验证：

!!! Example "示例"

    === "Python"
        ```python
        from ultralytics import FastSAM

        # 创建FastSAM模型
        model = FastSAM('FastSAM-s.pt')  # 或 FastSAM-x.pt

        # 验证模型
        results = model.val(data='coco8-seg.yaml')
        ```

    === "CLI"
        ```bash
        # 加载FastSAM模型，并在COCO8示例数据集上进行验证，图像大小为640
        yolo segment val model=FastSAM-s.pt data=coco8.yaml imgsz=640
        ```

请注意，FastSAM仅支持检测和分割单个类别的对象。这意味着它将识别和分割所有对象为相同的类别。因此，在准备数据集时，需要将所有对象的类别ID转换为0。

## FastSAM官方用法

FastSAM也可以直接从[https://github.com/CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)存储库中获取。以下是您可能采取的使用FastSAM的典型步骤的简要概述：

### 安装

1. 克隆FastSAM存储库：
   ```shell
   git clone https://github.com/CASIA-IVA-Lab/FastSAM.git
   ```

2. 创建并激活一个带有Python 3.9的Conda环境：
   ```shell
   conda create -n FastSAM python=3.9
   conda activate FastSAM
   ```

3. 进入克隆的存储库并安装所需的软件包：
   ```shell
   cd FastSAM
   pip install -r requirements.txt
   ```

4. 安装CLIP模型：
   ```shell
   pip install git+https://github.com/openai/CLIP.git
   ```

### 示例用法

1. 下载[模型检查点](https://drive.google.com/file/d/1m1sjY4ihXBU1fZXdQ-Xdj-mDltW-2Rqv/view?usp=sharing)。

2. 使用FastSAM进行推断。示例命令：

    - 在图像中分割所有内容：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg
      ```

    - 使用文本提示分割特定对象：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --text_prompt "the yellow dog"
      ```

    - 在边界框中分割对象（以xywh格式提供边界框坐标）：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --box_prompt "[570,200,230,400]"
      ```

    - 在特定点附近分割对象：
      ```shell
      python Inference.py --model_path ./weights/FastSAM.pt --img_path ./images/dogs.jpg --point_prompt "[[520,360],[620,300]]" --point_label "[1,0]"
      ```

此外，您可以在[Colab演示](https://colab.research.google.com/drive/1oX14f6IneGGw612WgVlAiy91UHwFAvr9?usp=sharing)上尝试FastSAM，或在[HuggingFace Web演示](https://huggingface.co/spaces/An-619/FastSAM)上进行可视化体验。

## 引用和致谢

我们要感谢FastSAM作者在实时实例分割领域作出的重要贡献：

!!! Quote ""

    === "BibTeX"

      ```bibtex
      @misc{zhao2023fast,
            title={Fast Segment Anything},
            author={Xu Zhao and Wenchao Ding and Yongqi An and Yinglong Du and Tao Yu and Min Li and Ming Tang and Jinqiao Wang},
            year={2023},
            eprint={2306.12156},
            archivePrefix={arXiv},
            primaryClass={cs.CV}
      }
      ```

可在[arXiv](https://arxiv.org/abs/2306.12156)上找到原始的FastSAM论文。作者已经公开了他们的工作，代码库可以在[GitHub](https://github.com/CASIA-IVA-Lab/FastSAM)上获取。我们感谢他们在推动该领域以及使他们的工作对更广泛的社区可访问方面所做的努力。
