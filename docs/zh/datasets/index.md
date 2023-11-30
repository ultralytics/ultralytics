---
comments: true
description: 探索 Ultralytics 支持的多种计算机视觉数据集，适用于对象检测、分割、姿态估计、图像分类和多对象跟踪。
keywords: 计算机视觉, 数据集, Ultralytics, YOLO, 对象检测, 实例分割, 姿态估计, 图像分类, 多对象跟踪
---

# 数据集概览

Ultralytics 支持多种数据集，方便开展计算机视觉任务，诸如检测、实例分割、姿态估计、分类和多对象跟踪。以下是主要 Ultralytics 数据集的列表，以及每个计算机视觉任务及其相应数据集的概述。

!!! Note "笔记"

    Ultralytics 团队正在努力将文档翻译成多种语言。目前，本页面上的链接可能会直接指向英文文档页面，因为我们正在扩展多语言文档支持。感谢您的耐心等待 🙏！

## [检测数据集](/../datasets/detect/index.md)

边界框对象检测是一种计算机视觉技术，涉及通过在图像中的每个对象周围绘制边界框来检测和定位对象。

- [Argoverse](/../datasets/detect/argoverse.md)：包含城市环境中的 3D 追踪和运动预测数据，并提供丰富的注释。
- [COCO](/../datasets/detect/coco.md)：一个大型数据集，专为对象检测、分割和描述设计，包含 20 多万带有标签的图像。
- [COCO8](/../datasets/detect/coco8.md)：包含 COCO 训练集和 COCO 验证集的前 4 张图像，适合快速测试。
- [Global Wheat 2020](/../datasets/detect/globalwheat2020.md)：一个小麦头部图像的数据集，收集自世界各地，用于对象检测和定位任务。
- [Objects365](/../datasets/detect/objects365.md)：一个高质量的大规模对象检测数据集，含 365 个对象类别和逾 60 万个注释图像。
- [OpenImagesV7](/../datasets/detect/open-images-v7.md)：谷歌提供的综合数据集，包含 170 万训练图像和 4.2 万验证图像。
- [SKU-110K](/../datasets/detect/sku-110k.md)：一个特点是在零售环境中进行密集对象检测的数据集，包含 1.1 万图像和 170 万个边界框。
- [VisDrone](/../datasets/detect/visdrone.md)：包含无人机拍摄图像中的对象检测和多对象跟踪数据的数据集，包含超过 1 万张图像和视频序列。
- [VOC](/../datasets/detect/voc.md)：Pascal Visual Object Classes (VOC) 对象检测和分割数据集，包含 20 个对象类别和逾 1.1 万图像。
- [xView](/../datasets/detect/xview.md)：用于航拍图像对象检测的数据集，包含 60 个对象类别和逾 100 万个注释对象。

## [实例分割数据集](/../datasets/segment/index.md)

实例分割是一种计算机视觉技术，涉及在像素级别识别和定位图像中的对象。

- [COCO](/../datasets/segment/coco.md)：一个大型数据集，专为对象检测、分割和描述任务设计，包含 20 多万带有标签的图像。
- [COCO8-seg](/../datasets/segment/coco8-seg.md)：一个用于实例分割任务的较小数据集，包含 8 张带有分割注释的 COCO 图像。

## [姿态估计](/../datasets/pose/index.md)

姿态估计是一种用于确定对象相对于相机或世界坐标系统的姿势的技术。

- [COCO](/../datasets/pose/coco.md)：一个包含人体姿态注释的大型数据集，专为姿态估计任务设计。
- [COCO8-pose](/../datasets/pose/coco8-pose.md)：一个用于姿态估计任务的较小数据集，包含 8 张带有人体姿态注释的 COCO 图像。
- [Tiger-pose](/../datasets/pose/tiger-pose.md)：一个紧凑型数据集，包含 263 张专注于老虎的图像，每只老虎注释有 12 个关键点，用于姿态估计任务。

## [分类](/../datasets/classify/index.md)

图像分类是一个计算机视觉任务，涉及基于其视觉内容将图像分类到一个或多个预定义类别中。

- [Caltech 101](/../datasets/classify/caltech101.md)：包含 101 个对象类别图像的数据集，用于图像分类任务。
- [Caltech 256](/../datasets/classify/caltech256.md)：Caltech 101 的扩展版本，具有 256 个对象类别和更具挑战性的图像。
- [CIFAR-10](/../datasets/classify/cifar10.md)：包含 60K 32x32 彩色图像的数据集，分为 10 个类别，每个类别有 6K 图像。
- [CIFAR-100](/../datasets/classify/cifar100.md)：CIFAR-10 的扩展版本，具有 100 个对象类别和每类 600 个图像。
- [Fashion-MNIST](/../datasets/classify/fashion-mnist.md)：包含 70,000 张灰度图像的数据集，图像来自 10 个时尚类别，用于图像分类任务。
- [ImageNet](/../datasets/classify/imagenet.md)：一个大型的用于对象检测和图像分类的数据集，包含超过 1400 万图像和 2 万个类别。
- [ImageNet-10](/../datasets/classify/imagenet10.md)：ImageNet 的一个较小子集，包含 10 个类别，用于更快速的实验和测试。
- [Imagenette](/../datasets/classify/imagenette.md)：ImageNet 的一个较小子集，其中包含 10 个容易区分的类别，用于更快速的训练和测试。
- [Imagewoof](/../datasets/classify/imagewoof.md)：ImageNet 的一个更具挑战性的子集，包含 10 个狗品种类别用于图像分类任务。
- [MNIST](/../datasets/classify/mnist.md)：包含 70,000 张手写数字灰度图像的数据集，用于图像分类任务。

## [定向边界框 (OBB)](/../datasets/obb/index.md)

定向边界框 (OBB) 是一种计算机视觉方法，用于使用旋转的边界框检测图像中的倾斜对象，常应用于航空和卫星图像。

- [DOTAv2](/../datasets/obb/dota-v2.md)：一个流行的 OBB 航拍图像数据集，拥有 170 万个实例和 11,268 张图像。

## [多对象跟踪](/../datasets/track/index.md)

多对象跟踪是一种计算机视觉技术，涉及在视频序列中检测和跟踪多个对象的运动。

- [Argoverse](/../datasets/detect/argoverse.md)：包含城市环境中的 3D 追踪和运动预测数据，并提供丰富的注释，适用于多对象跟踪任务。
- [VisDrone](/../datasets/detect/visdrone.md)：包含无人机拍摄图像中的对象检测和多对象跟踪数据的数据集，包含超过 1 万张图像和视频序列。

## 贡献新数据集

贡献一个新数据集需要几个步骤，来确保它与现有基础设施良好对齐。以下是必要的步骤：

### 贡献新数据集的步骤

1. **收集图像**：收集属于数据集的图像。这些可能来自公共数据库或您自己的收藏。

2. **注释图像**：根据任务对这些图像进行边界框、分段或关键点的标记。

3. **导出注释**：将这些注释转换为 Ultralytics 支持的 YOLO *.txt 文件格式。

4. **组织数据集**：按正确的文件夹结构排列您的数据集。您应该有 `train/ ` 和 `val/` 顶级目录，在每个目录内，有 `images/` 和 `labels/` 子目录。

    ```
    dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    └── val/
        ├── images/
        └── labels/
    ```

5. **创建一个 `data.yaml` 文件**：在数据集的根目录中，创建一个描述数据集的 `data.yaml` 文件，包括类别信息等必要内容。

6. **优化图像（可选）**：如果您想为了更高效的处理而减小数据集的大小，可以使用以下代码来优化图像。这不是必需的，但推荐用于减小数据集大小和加快下载速度。

7. **压缩数据集**：将整个数据集文件夹压缩成一个 zip 文件。

8. **文档和 PR**：创建描述您的数据集和它如何融入现有框架的文档页面。之后，提交一个 Pull Request (PR)。更多关于如何提交 PR 的详细信息，请参照 [Ultralytics 贡献指南](https://docs.ultralytics.com/help/contributing)。

### 优化和压缩数据集的示例代码

!!! Example "优化和压缩数据集"

    === "Python"

    ```python
    from pathlib import Path
    from ultralytics.data.utils import compress_one_image
    from ultralytics.utils.downloads import zip_directory

    # 定义数据集目录
    path = Path('path/to/dataset')

    # 优化数据集中的图像（可选）
    for f in path.rglob('*.jpg'):
        compress_one_image(f)

    # 将数据集压缩成 'path/to/dataset.zip'
    zip_directory(path)
    ```

通过遵循这些步骤，您可以贡献一个与 Ultralytics 现有结构良好融合的新数据集。
