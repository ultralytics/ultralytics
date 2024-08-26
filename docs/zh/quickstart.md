---
comments: true
description: 探索使用pip、conda、git和Docker安装Ultralytics的各种方法。了解如何在命令行界面或Python项目中使用Ultralytics。
keywords: Ultralytics安装，pip安装Ultralytics，Docker安装Ultralytics，Ultralytics命令行界面，Ultralytics Python接口
---

## 安装Ultralytics

Ultralytics提供了多种安装方法，包括pip、conda和Docker。通过`ultralytics`pip包安装最新稳定版的YOLOv8，或者克隆[Ultralytics GitHub仓库](https://github.com/ultralytics/ultralytics)以获取最新版本。Docker可用于在隔离容器中执行包，避免本地安装。

!!! Example "安装"

    === "Pip安装（推荐）"
        使用pip安装`ultralytics`包，或通过运行`pip install -U ultralytics`更新现有安装。访问Python包索引(PyPI)了解更多关于`ultralytics`包的详细信息：[https://pypi.org/project/ultralytics/](https://pypi.org/project/ultralytics/)。

        [![PyPI版本](https://badge.fury.io/py/ultralytics.svg)](https://badge.fury.io/py/ultralytics) [![下载](https://static.pepy.tech/badge/ultralytics)](https://pepy.tech/project/ultralytics)

        ```bash
        # 从PyPI安装ultralytics包
        pip install ultralytics
        ```

        你也可以直接从GitHub[仓库](https://github.com/ultralytics/ultralytics)安装`ultralytics`包。如果你想要最新的开发版本，这可能会很有用。确保你的系统上安装了Git命令行工具。`@main`指令安装`main`分支，可修改为其他分支，如`@my-branch`，或完全删除，默认为`main`分支。

        ```bash
        # 从GitHub安装ultralytics包
        pip install git+https://github.com/ultralytics/ultralytics.git@main
        ```


    === "Conda安装"
        Conda是pip的一个替代包管理器，也可用于安装。访问Anaconda了解更多详情，网址为[https://anaconda.org/conda-forge/ultralytics](https://anaconda.org/conda-forge/ultralytics)。用于更新conda包的Ultralytics feedstock仓库位于[https://github.com/conda-forge/ultralytics-feedstock/](https://github.com/conda-forge/ultralytics-feedstock/)。


        [![Conda配方](https://img.shields.io/badge/recipe-ultralytics-green.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda下载](https://img.shields.io/conda/dn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda版本](https://img.shields.io/conda/vn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics) [![Conda平台](https://img.shields.io/conda/pn/conda-forge/ultralytics.svg)](https://anaconda.org/conda-forge/ultralytics)

        ```bash
        # 使用conda安装ultralytics包
        conda install -c conda-forge ultralytics
        ```

        !!! 注意

            如果你在CUDA环境中安装，最佳实践是同时安装`ultralytics`、`pytorch`和`pytorch-cuda`，以便conda包管理器解决任何冲突，或者最后安装`pytorch-cuda`，让它必要时覆盖特定于CPU的`pytorch`包。
            ```bash
            # 使用conda一起安装所有包
            conda install -c pytorch -c nvidia -c conda-forge pytorch torchvision pytorch-cuda=11.8 ultralytics
            ```

        ### Conda Docker映像

        Ultralytics Conda Docker映像也可从[DockerHub](https://hub.docker.com/r/ultralytics/ultralytics)获得。这些映像基于[Miniconda3](https://docs.conda.io/projects/miniconda/en/latest/)，是开始在Conda环境中使用`ultralytics`的简单方式。

        ```bash
        # 将映像名称设置为变量
        t=ultralytics/ultralytics:latest-conda

        # 从Docker Hub拉取最新的ultralytics映像
        sudo docker pull $t

        # 使用GPU支持运行ultralytics映像的容器
        sudo docker run -it --ipc=host --gpus all $t  # 所有GPU
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # 指定GPU
        ```

    === "Git克隆"
        如果您对参与开发感兴趣或希望尝试最新源代码，请克隆`ultralytics`仓库。克隆后，导航到目录并使用pip以可编辑模式`-e`安装包。
        ```bash
        # 克隆ultralytics仓库
        git clone https://github.com/ultralytics/ultralytics

        # 导航到克隆的目录
        cd ultralytics

        # 为开发安装可编辑模式下的包
        pip install -e .
        ```

    === "Docker"

        利用Docker轻松地在隔离的容器中执行`ultralytics`包，确保跨不同环境的一致性和流畅性能。通过选择一款官方`ultralytics`映像，从[Docker Hub](https://hub.docker.com/r/ultralytics/ultralytics)中不仅避免了本地安装的复杂性，还获得了对验证工作环境的访问。Ultralytics提供5种主要支持的Docker映像，每一种都为不同的平台和使用案例设计，以提供高兼容性和效率：

        <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker拉取次数"></a>

        - **Dockerfile：** 推荐用于训练的GPU映像。
        - **Dockerfile-arm64：** 为ARM64架构优化，允许在树莓派和其他基于ARM64的平台上部署。
        - **Dockerfile-cpu：** 基于Ubuntu的CPU版，适合无GPU环境下的推理。
        - **Dockerfile-jetson：** 为NVIDIA Jetson设备量身定制，整合了针对这些平台优化的GPU支持。
        - **Dockerfile-python：** 最小化映像，只包含Python及必要依赖，理想于轻量级应用和开发。
        - **Dockerfile-conda：** 基于Miniconda3，包含conda安装的ultralytics包。

        以下是获取最新映像并执行它的命令：

        ```bash
        # 将映像名称设置为变量
        t=ultralytics/ultralytics:latest

        # 从Docker Hub拉取最新的ultralytics映像
        sudo docker pull $t

        # 使用GPU支持运行ultralytics映像的容器
        sudo docker run -it --ipc=host --gpus all $t  # 所有GPU
        sudo docker run -it --ipc=host --gpus '"device=2,3"' $t  # 指定GPU
        ```

        上述命令初始化了一个带有最新`ultralytics`映像的Docker容器。`-it`标志分配了一个伪TTY，并保持stdin打开，使您可以与容器交互。`--ipc=host`标志将IPC（进程间通信）命名空间设置为宿主，这对于进程之间的内存共享至关重要。`--gpus all`标志使容器内可以访问所有可用的GPU，这对于需要GPU计算的任务至关重要。

        注意：要在容器中使用本地机器上的文件，请使用Docker卷将本地目录挂载到容器中：

        ```bash
        # 将本地目录挂载到容器内的目录
        sudo docker run -it --ipc=host --gpus all -v /path/on/host:/path/in/container $t
        ```

        将`/path/on/host`更改为您本地机器上的目录路径，将`/path/in/container`更改为Docker容器内希望访问的路径。

        欲了解进阶Docker使用方法，请探索[Ultralytics Docker指南](https://docs.ultralytics.com/guides/docker-quickstart/)。

有关依赖项列表，请参见`ultralytics`的[requirements.txt](https://github.com/ultralytics/ultralytics/blob/main/pyproject.toml)文件。请注意，上述所有示例均安装了所有必需的依赖项。

<p align="center">
  <br>
  <iframe width="720" height="405" src="https://www.youtube.com/embed/_a7cVL9hqnk"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Ultralytics YOLO Quick Start Guide
</p>

!!! Tip "提示"

    PyTorch的要求因操作系统和CUDA需要而异，因此建议首先根据[https://pytorch.org/get-started/locally](https://pytorch.org/get-started/locally)上的指南安装PyTorch。

    <a href="https://pytorch.org/get-started/locally/">
        <img width="800" alt="PyTorch安装指南" src="https://user-images.githubusercontent.com/26833433/228650108-ab0ec98a-b328-4f40-a40d-95355e8a84e3.png">
    </a>

## 通过CLI使用Ultralytics

Ultralytics命令行界面（CLI）允许您通过简单的单行命令使用，无需Python环境。CLI不需要自定义或Python代码。您可以直接从终端使用`yolo`命令运行所有任务。查看[CLI指南](/../usage/cli.md)，了解更多关于从命令行使用YOLOv8的信息。

!!! Example "示例"

    === "语法"

        Ultralytics `yolo`命令使用以下语法：
        ```bash
        yolo 任务 模式 参数

        其中   任务（可选）是[detect, segment, classify]中的一个
                模式（必需）是[train, val, predict, export, track]中的一个
                参数（可选）是任意数量的自定义“arg=value”对，如“imgsz=320”，可覆盖默认值。
        ```
        在完整的[配置指南](/../usage/cfg.md)中查看所有参数，或者用`yolo cfg`查看

    === "训练"

        用初始学习率0.01训练检测模型10个周期
        ```bash
        yolo train data=coco128.yaml model=yolov8n.pt epochs=10 lr0=0.01
        ```

    === "预测"

        使用预训练的分割模型以320的图像大小预测YouTube视频：
        ```bash
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320
        ```

    === "验证"

        以批量大小1和640的图像大小验证预训练的检测模型：
        ```bash
        yolo val model=yolov8n.pt data=coco128.yaml batch=1 imgsz=640
        ```

    === "导出"

        以224x128的图像大小将YOLOv8n分类模型导出到ONNX格式（无需任务）
        ```bash
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128
        ```

    === "特殊"

        运行特殊命令以查看版本、查看设置、运行检查等：
        ```bash
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg
        ```

!!! Warning "警告"

    参数必须以`arg=val`对的形式传递，用等号`=`分隔，并用空格` `分隔对。不要使用`--`参数前缀或逗号`,`分隔参数。

    - `yolo predict model=yolov8n.pt imgsz=640 conf=0.25` &nbsp; ✅
    - `yolo predict model yolov8n.pt imgsz 640 conf 0.25` &nbsp; ❌
    - `yolo predict --model yolov8n.pt --imgsz 640 --conf 0.25` &nbsp; ❌

[CLI指南](/../usage/cli.md){ .md-button }

## 通过Python使用Ultralytics

YOLOv8的Python接口允许无缝集成进您的Python项目，轻松加载、运行模型及处理输出。Python接口设计简洁易用，使用户能快速实现他们项目中的目标检测、分割和分类功能。这使YOLOv8的Python接口成为任何希望在其Python项目中纳入这些功能的人的宝贵工具。

例如，用户可以加载一个模型，训练它，在验证集上评估性能，甚至只需几行代码就可以将其导出到ONNX格式。查看[Python指南](/../usage/python.md)，了解更多关于在Python项目中使用YOLOv8的信息。

!!! Example "示例"

    ```python
    from ultralytics import YOLO

    # 从头开始创建一个新的YOLO模型
    model = YOLO("yolov8n.yaml")

    # 加载预训练的YOLO模型（推荐用于训练）
    model = YOLO("yolov8n.pt")

    # 使用“coco128.yaml”数据集训练模型3个周期
    results = model.train(data="coco128.yaml", epochs=3)

    # 评估模型在验证集上的性能
    results = model.val()

    # 使用模型对图片进行目标检测
    results = model("https://ultralytics.com/images/bus.jpg")

    # 将模型导出为ONNX格式
    success = model.export(format="onnx")
    ```

[Python指南](/../usage/python.md){.md-button .md-button--primary}

## Ultralytics设置

Ultralytics库提供了一个强大的设置管理系统，允许您精细控制实验。通过利用`ultralytics.utils`模块中的`SettingsManager`，用户可以轻松访问和修改设置。这些设置存储在YAML文件中，可以直接在Python环境中查看或修改，或者通过命令行界面(CLI)修改。

### 检查设置

若要了解当前设置的配置情况，您可以直接查看：

!!! Example "查看设置"

    === "Python"
        您可以使用Python查看设置。首先从`ultralytics`模块导入`settings`对象。使用以下命令打印和返回设置：
        ```python
        from ultralytics import settings

        # 查看所有设置
        print(settings)

        # 返回特定设置
        value = settings["runs_dir"]
        ```

    === "CLI"
        或者，命令行界面允许您用一个简单的命令检查您的设置：
        ```bash
        yolo settings
        ```

### 修改设置

Ultralytics允许用户轻松修改他们的设置。更改可以通过以下方式执行：

!!! Example "更新设置"

    === "Python"
        在Python环境中，调用`settings`对象上的`update`方法来更改您的设置：
        ```python
        from ultralytics import settings

        # 更新一个设置
        settings.update({"runs_dir": "/path/to/runs"})

        # 更新多个设置
        settings.update({"runs_dir": "/path/to/runs", "tensorboard": False})

        # 重置设置为默认值
        settings.reset()
        ```

    === "CLI"
        如果您更喜欢使用命令行界面，以下命令将允许您修改设置：
        ```bash
        # 更新一个设置
        yolo settings runs_dir='/path/to/runs'

        # 更新多个设置
        yolo settings runs_dir='/path/to/runs' tensorboard=False

        # 重置设置为默认值
        yolo settings reset
        ```

### 理解设置

下表提供了Ultralytics中可调整设置的概览。每个设置都概述了一个示例值、数据类型和简短描述。

| 名称               | 示例值                | 数据类型 | 描述                                                                                              |
| ------------------ | --------------------- | -------- | ------------------------------------------------------------------------------------------------- |
| `settings_version` | `'0.0.4'`             | `str`    | Ultralytics _settings_ 版本（不同于Ultralytics [pip](https://pypi.org/project/ultralytics/)版本） |
| `datasets_dir`     | `'/path/to/datasets'` | `str`    | 存储数据集的目录                                                                                  |
| `weights_dir`      | `'/path/to/weights'`  | `str`    | 存储模型权重的目录                                                                                |
| `runs_dir`         | `'/path/to/runs'`     | `str`    | 存储实验运行的目录                                                                                |
| `uuid`             | `'a1b2c3d4'`          | `str`    | 当前设置的唯一标识符                                                                              |
| `sync`             | `True`                | `bool`   | 是否将分析和崩溃同步到HUB                                                                         |
| `api_key`          | `''`                  | `str`    | Ultralytics HUB [API Key](https://hub.ultralytics.com/settings?tab=api+keys)                      |
| `clearml`          | `True`                | `bool`   | 是否使用ClearML记录                                                                               |
| `comet`            | `True`                | `bool`   | 是否使用[Comet ML](https://bit.ly/yolov8-readme-comet)进行实验跟踪和可视化                        |
| `dvc`              | `True`                | `bool`   | 是否使用[DVC进行实验跟踪](https://dvc.org/doc/dvclive/ml-frameworks/yolo)和版本控制               |
| `hub`              | `True`                | `bool`   | 是否使用[Ultralytics HUB](https://hub.ultralytics.com)集成                                        |
| `mlflow`           | `True`                | `bool`   | 是否使用MLFlow进行实验跟踪                                                                        |
| `neptune`          | `True`                | `bool`   | 是否使用Neptune进行实验跟踪                                                                       |
| `raytune`          | `True`                | `bool`   | 是否使用Ray Tune进行超参数调整                                                                    |
| `tensorboard`      | `True`                | `bool`   | 是否使用TensorBoard进行可视化                                                                     |
| `wandb`            | `True`                | `bool`   | 是否使用Weights & Biases记录                                                                      |

在您浏览项目或实验时，请务必重新访问这些设置，以确保它们为您的需求提供最佳配置。
