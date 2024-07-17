---
comments: true
description: Learn how to install Ultralytics using pip, conda, or Docker. Follow our step-by-step guide for a seamless setup of YOLOv8 with thorough instructions.
keywords: Ultralytics, YOLOv8, Install Ultralytics, pip, conda, Docker, GitHub, machine learning, object detection
---

## 安装 Ultralytics

Ultralytics提供了多种安装方法，包括 pip、conda和Docker。通过`ultralytics`pip包安装YOLOv8以获得最新的稳定版本，或者通过克隆[Ultralytics GitHub 源码库](https://github.com/ultralytics/ultralytics)安装最新版本。Docker可用于在隔离的容器中执行包，避免本地安装。

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>观看:</strong> Ultralytics YOLO快速入门指南
</p>

!!! Example "安装"

    <p align="left" style="margin-bottom: -20px;">![PyPI - Python Version](https://img.shields.io/pypi/pyversions/ultralytics?logo=python&logoColor=gold)<p>

    === "Pip 安装 (推荐)"

        使用 pip 安装`ultralytics`包，或通过运行`pip install -U ultralytics`更新现有安装。访问Python Package Index（PyPI）以获取有关`ultralytics`包的更多详细信息：[https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/)。

        [![PyPI - Version](https://img.shields.io/pypi/v/ultralytics?logo=pypi&logoColor=white)](https://pypi.org/project/ultralytics/)
        [![Downloads](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # 从PyPI安装ultralytics包
        pip install ultralytics
        ```

        您也可以直接从 GitHub [源码库](https://github.com/ultralytics/ultralytics)安装`ultralytics`包。如果您想要最新的开发版本，这可能很有用。确保在系统上安装了 Git 命令行工具。`@main`命令安装`main`分支，并可以修改为另一个分支，即`@my-branch`，或完全删除以默认为`main`分支。

        ```bash
        # 从GitHub安装ultralytics包
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```

    === "Conda 安装"

        Conda 是pip的替代包管理器，也可用于安装。访问Anaconda了解更多详情，请访问[https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics)。用于更新conda包的Ultralytics仓库位于 [https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/)。

        [![Conda Version](https://img.shields.io/conda/vn/conda-forge/ultralytics?logo=condaforge)](https://anaconda.org/conda-forge/ultralytics)
        [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)
        [![Conda Recipe](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics)
        [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # 使用conda安装ultralytics包
        conda install -c conda-forge ultralytics
        ```

        !!! 备注

            如果要在 CUDA 环境中安装，最佳做法是在同一命令中安装`ultralytics`、`pytorch`和`pytorch-cuda`，以允许conda包管理器解决任何冲突，或者最后安装 `pytorch-cuda`以允许它在必要时覆盖特定于CPU的`pytorch`包。
            ```bash
            # 使用conda将所有包安装在一起
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Conda Docker 镜像

        Ultralytics Conda Docker 镜像也可从 [DockerHub](https://hub.docker.com/r/ultralytics/ultralytics)获得。这些图像基于[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)，是在Conda环境中开始使用`ultralytics`的简单方法。

        ```bash
        # 将镜像名称设置为变量
        t=ultralytics/ultralytics:latest-conda

        # 从Docker Hub拉取最新的ultralytics镜像
        sudo docker pull $t

        # 在支持GPU的容器中运行ultralytics镜像
        sudo docker run -it --ipc=host --gpus all $t  # 所有GPUs
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # 指定GPUs
        ```

    === "Git 克隆"

        如果您有兴趣为开发做出贡献或希望尝试最新的源代码，请克隆`ultralytics`存储库。克隆后，导航到目录并使用 pip 在可编辑模式“-e”下安装包。

        [![GitHub last commit](https://img.shields.io/github/last-commit/ultralytics/ultralytics?logo=github)](https://github.com/ultralytics/ultralytics)
        [![GitHub commit activity](https://img.shields.io/github/commit-activity/t/ultralytics/ultralytics)](https://github.com/ultralytics/ultralytics)

        ```bash
        # 克隆ultralytics源码库
        git clone https://github.com/ultralytics/ultralytics

        # 定位到克隆的目录
        cd ultralytics

        # 以可编辑模式安装包以进行开发
        pip install -e .
        ```

    === "Docker"

        利用Docker在隔离的容器中轻松执行`ultralytics`包，确保在各种环境中实现一致和流畅的性能。通过从[Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics)中选择一个官方的`ultralytics`映像，您不仅可以避免本地安装的复杂性，还可以从访问经过验证的工作环境中受益。Ultralytics提供5个主要支持的Docker镜像，每个镜像都旨在为不同的平台和用例提供高兼容性和效率:

        [![Docker Image Version](https://img.shields.io/docker/v/ultralytics/ultralytics?sort=semver&logo=docker)](https://hub.docker.com/r/ultralytics/ultralytics)
        [![Docker Pulls](https://img.shields.io/docker/pulls/ultralytics/ultralytics)](https://hub.docker.com/r/ultralytics/ultralytics)

        - **Dockerfile:** 建议用于训练的GPU镜像。
        - **Dockerfile-arm64:** 针对ARM64架构进行了优化，允许在Raspberry Pi和其他基于ARM64的平台上部署。
        - **Dockerfile-cpu:** 基于Ubuntu的纯CPU版本，适用于推理和没有GPU的环境。
        - **Dockerfile-jetson:** 专为NVIDIA Jetson设备量身定制，集成了针对这些平台优化的GPU支持。
        - **Dockerfile-python:** 只有Python和必要依赖项的最小镜像，非常适合轻量级应用程序和开发。
        - **Dockerfile-conda:** 基于Miniconda3和conda安装的ultralytics包。

        以下是获取最新镜像并执行它的命令:

        ```bash
        # 将镜像名称设置为变量
        t=ultralytics/ultralytics:latest

        # 从Docker Hub 拉取最新的ultralytics镜像
        sudo docker pull $t

        # 在支持GPU的容器中运行ultralytics镜像
        sudo docker run -it --ipc=host --gpus all $t  # 所有GPUs
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # 指定GPUs
        ```

        上面的命令使用最新的`ultralytics`镜像初始化Docker容器。`-it`标志分配一个伪TTY并保持stdin打开状态，使你能够与容器进行交互。`--ipc=host`标志将IPC（进程间通信）命名空间设置为主机，这对于在进程之间共享内存至关重要。`--gpus all`标志允许访问容器内所有可用的GPU，这对于需要GPU计算的任务至关重要。

        注意：要在容器中处理本地计算机上的文件，请使用Docker卷将本地目录装载到容器中:

        ```bash
        # 将本地目录装载到容器内的目录
        sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
        ```

        将`/path/on/host`更改为本地计算机上的目录路径，将`/path/in/container`更改为Docker容器中的所需路径，以便于访问。

        对于高级Docker使用，请尽情探索[Ultralytics Docker 指南](./guides/docker-quickstart.md).

有关依赖项列表，请参阅`ultralytics`[pyproject.toml](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)文件。请注意，上述所有示例都安装了所有必需的依赖项。

!!! 提示 "提示"

    PyTorch要求因操作系统和CUDA要求而异，因此建议先按照[https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)中的说明安装 PyTorch。

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch Installation Instructions" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## 将Ultralytics与CLI配合使用

Ultralytics命令行界面（CLI）允许使用简单的单行命令，而无需Python环境。CLI不需要自定义或Python代码。您可以使用`yolo`命令从终端运行所有任务。查看[CLI 指南](usage/cli.md)以了解有关从命令行使用YOLOv8的更多信息。

!!! 示例

    === "语法"

        Ultralytics `yolo`命令使用以下语法:
        ```bash
        yolo TASK MODE ARGS
        ```

        - `TASK` (可选) 是([检测](tasks/detect.md), [分割](tasks/segment.md), [分类](tasks/classify.md), [姿态估计](tasks/pose.md))之一
        - `MODE` (需要) 是([训练](modes/train.md), [验证](modes/val.md), [预测](modes/predict.md), [导出](modes/export.md), [追踪](modes/track.md))之一
        - `ARGS` (可选) 是覆盖默认值的`arg=value`对，例如`imgsz=640`。

        在完整的[配置指南](usage/cfg.md)或使用`yolo cfg` CLI命令查看所有 `ARGS`。

    === "训练"

        训练10个epoch的检测模型，初始learning_rate为0.01
        ```bash
        yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用图像大小为320的预训练分割模型预测YouTube视频:
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        验证批量大小为1、图像大小为640的预训练检测模型:
        ```bash
        yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640
        ```

    === "导出"

        将YOLOv8n分类模型导出为图像大小为224x128的ONNX格式（无需TASK）
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "特定"

        运行特定命令以查看版本、查看设置、运行检查等:
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! Warning "警告"

    参数必须作为`arg=val`对传递，由等号`=`分隔，并由对之间的空格分隔。不要在参数之间使用`--`参数前缀或逗号`，`。

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25`  ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25`  ❌ (缺失 `=`)
    - `yolo predict model=yolov8n.pt, imgsz=640, conf=0.25`  ❌ (不要使用 `,`)
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25`  ❌ (不要使用 `--`)

[CLI 指南](usage/cli.md){ .md-button }

## 将Ultralytics与Python配合使用

YOLOv8的Python接口可以无缝集成到您的Python项目中，从而轻松加载、运行和处理模型的输出。Python界面在设计时考虑到了简单性和易用性，使用户能够在其项目中快速实现对象检测、分割和分类。这使得YOLOv8的Python界面对于任何希望将这些功能整合到他们的Python项目中的人来说都是一个宝贵的工具。

例如，用户只需几行代码即可加载模型、训练模型、评估其在验证集上的性能，甚至将其导出为 ONNX 格式。查看[Python 指南](usage/python.md) 以了解有关在Python项目中使用YOLOv8的更多信息。

!!! 示例

    ```python
    from ultralytics import YOLO

    # 从头开始创建新的YOLO模型
    model = YOLO("yolov8n.yaml")

    # 加载预训练的YOLO模型（建议用于训练）
    model = YOLO("yolov8n.pt")

    # 使用`coco8.yaml`数据集训练3个epoch的模型
    results = model.train(data="coco8.yaml", epochs=3)

    # 评估模型在验证集上的性能
    results = model.val()

    # 使用模型对图像执行对象检测
    results = model("https://ultralytics.com/images/bus.jpg")

    # 将模型导出为ONNX格式
    success = model.export(format="onnx")
    ```

[Python 指南](usage/python.md){.md-button .md-button--primary}

## Ultralytics 设置

Ultralytics库提供了一个强大的设置管理系统，可以对您的实验进行细粒度的控制。通过使用`ultralytics.utils`模块中的`SettingsManager`，用户可以轻松访问和更改他们的设置。这些存储在YAML文件中，可以直接在Python环境中或通过命令行界面（CLI）查看或修改。

### 检查设置

要深入了解设置的当前配置，您可以直接查看它们:

!!! Example "查看设置"

    === "Python"

        您可以使用Python查看您的设置。首先从`ultralytics`模块导入`settings`对象。使用以下命令打印和返回设置:
        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回特定设置
        value = settings["runs_dir"]
        ```

    === "CLI"

        或者，命令行界面允许您使用简单的命令检查您的设置:
        ```bash
        yolo settings
        ```

### 修改设置

Ultralytics允许用户轻松修改其设置。可以通过以下方式执行更改:

!!! Example "更新设置"

    === "Python"

        在Python环境中，调用`settings`对象上的`update`方法来更改您的设置:
        ```python
        from ultralytics import settings

        # 更新设置
        settings.update({"runs_dir": "/path/to/runs"})

        # 更新多个设置
        settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

        # 将设置重置为默认值
        settings.reset()
        ```

    === "CLI"

        如果您更喜欢使用命令行界面，以下命令将允许您修改设置:
        ```bash
        # 更新设置
        yolo settings runs_dir='/path/to/runs'

        # 更新多个设置
        yolo settings runs_dir='/path/to/runs' tensorboard=False

        # 将设置重置为默认值
        yolo settings reset
        ```

### 了解设置

下表概述了Ultralytics中可用于调整的设置。每个设置都随示例值、数据类型和简要说明一起概述。

| 名称               | 示例值                | 数据类型 | 说明                                                                                           |
| ------------------ | --------------------- | -------- | ---------------------------------------------------------------------------------------------- |
| `settings_version` | `'0.0.4'`             | `str`    | Ultralytics _settings_ 版本(与Ultralytics[pip](https://pypi.org/project/ultralytics/)版本不同) |
| `datasets_dir`     | `'/path/to/datasets'` | `str`    | 存储数据集的目录                                                                               |
| `weights_dir`      | `'/path/to/weights'`  | `str`    | 存储模型权重的目录                                                                             |
| `runs_dir`         | `'/path/to/runs'`     | `str`    | 运行试验的目录存储                                                                             |
| `uuid`             | `'a1b2c3d4'`          | `str`    | 当前设置的唯一标识符                                                                           |
| `sync`             | `True`                | `bool`   | 是否将分析和崩溃同步到HUB                                                                      |
| `api_key`          | `''`                  | `str`    | Ultralytics HUB [API Key](https://hub.ultralytics.com/settings?tab=api+keys)                   |
| `clearml`          | `True`                | `bool`   | 是否使用ClearML日志记录                                                                        |
| `comet`            | `True`                | `bool`   | 是否使用[Comet ML](https://bit.ly/yolov8-readme-comet)进行实验跟踪和可视化                     |
| `dvc`              | `True`                | `bool`   | 是否使用[DVC 进行实验跟踪](https://dvc.org/doc/dvclive/ml-frameworks/yolo)和版本控制           |
| `hub`              | `True`                | `bool`   | 是否使用[Ultralytics HUB](https://hub.ultralytics.com)集成                                     |
| `mlflow`           | `True`                | `bool`   | 是否使用MLFlow进行实验跟踪                                                                     |
| `neptune`          | `True`                | `bool`   | 是否使用Neptune进行实验跟踪                                                                    |
| `raytune`          | `True`                | `bool`   | 是否使用光线调优进行超参数调优                                                                 |
| `tensorboard`      | `True`                | `bool`   | 是否使用TensorBoard进行可视化                                                                  |
| `wandb`            | `True`                | `bool`   | 是否使用权重和偏差记录                                                                         |

在浏览项目或实验时，请务必重新访问这些设置，以确保它们针对您的需求进行了最佳配置。

## 常见问题

### 如何使用pip安装 Ultralytics YOLOv8？

要使用pip安装 Ultralytics YOLOv8，请执行以下命令:

```bash
pip install ultralytics
```

对于最新的稳定版本，这将直接从Python Package Index（PyPI）安装`ultralytics`包。有关详细信息，请访问 [PyPI上的ultralytics包](https://pypi.org/project/ultralytics/)。

或者，您可以直接从GitHub安装最新的开发版本:

```bash
pip install git+https://github.com/ultralytics/ultralytics.git
```

确保在系统上安装了Git命令行工具。

### 我可以使用conda安装 Ultralytics YOLOv8吗？

是的，您可以通过运行conda安装 Ultralytics YOLOv8:

```bash
conda install -c conda-forge ultralytics
```

此方法是pip的绝佳替代方法，可确保与环境中的其他包兼容。对于CUDA环境，最好同时安装`ultralytics`、`pytorch`和`pytorch-cuda`以解决任何冲突:

```bash
conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
```

有关更多说明，请参考[Conda 快速入门指南](guides/conda-quickstart.md).

### 使用Docker运行Ultralytics YOLOv8有什么优势？

使用Docker运行Ultralytics YOLOv8提供了一个隔离且一致的环境，确保了不同系统的流畅性能。它还消除了本地安装的复杂性。Ultralytics的官方Docker镜像可在[Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics)上找到，并针对GPU、CPU、ARM64、NVIDIA Jetson和Conda环境提供不同的变体。以下是拉取和运行最新镜像的命令:

```bash
# 从 Docker Hub拉取最新的ultralytics镜像
sudo docker pull ultralytics/ultralytics:latest

# 在支持GPU的容器中运行ultralytics镜像
sudo docker run -it --ipc=host --gpus all ultralytics/ultralytics:latest
```

有关更详细的Docker说明，请查看[Docker 快速入门指南](guides/docker-quickstart.md).

### 如何克隆Ultralytics源码库进行开发？

要克隆Ultralytics源码库并设置开发环境，请使用以下步骤:

```bash
# 克隆ultralytics源码库
git clone https://github.com/ultralytics/ultralytics

# 定位到克隆的目录
cd ultralytics

# 以可编辑模式安装包以进行开发
pip install -e .
```

此方法允许您为项目做出贡献或使用最新的源代码进行试验。有关详细信息，请访问 [Ultralytics GitHub 源码库](https://github.com/ultralytics/ultralytics)。

### 为什么要使用 Ultralytics YOLOv8 CLI？

Ultralytics YOLOv8命令行界面（CLI）简化了运行对象检测任务，无需Python代码。您可以直接从终端执行训练、验证和预测等任务的单行命令。`yolo`命令的基本语法是:

```bash
yolo TASK MODE ARGS
```

例如，使用指定参数训练检测模型:

```bash
yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01
```

查看完整的[CLI 指南](usage/cli.md)以探索更多命令和使用示例。
