# Ultralytics YOLO 科研定制 Quickstart

本文面向具备 Python 和 PyTorch 基础、准备基于 Ultralytics YOLO 开展科研实验的读者。读完后，你应该能判断一个想法需要修改配置、模型结构、损失函数、数据管线还是训练流程，并完成一次可复现的最小实验。

如果你只想用自己的数据训练官方模型，无须修改源码，直接阅读[训练模式文档](https://docs.ultralytics.com/zh/modes/train/)。本文重点介绍“如何读源码并定制模型”。

## 1. 先建立源码地图

下面只列出科研定制最常用的目录。第一次阅读时不要从文件头逐行看起，应先沿调用链定位改动的 owner。

```text
ultralytics/
├── cfg/
│   ├── default.yaml              # 所有 train/val/predict/export 参数的统一入口
│   ├── datasets/                 # 官方数据集 YAML 示例
│   └── models/                   # 模型结构 YAML，按模型家族组织
├── data/
│   ├── build.py                  # DataLoader 和 Dataset 的创建入口
│   ├── dataset.py                # 检测、分割等数据集实现
│   └── augment.py                # 数据增强与样本变换
├── engine/
│   ├── model.py                  # 用户 API 门面：train、val、predict、export、track
│   ├── trainer.py                # 通用训练循环 BaseTrainer
│   ├── validator.py              # 通用验证生命周期
│   ├── predictor.py              # 通用推理生命周期
│   └── results.py                # 预测结果的数据结构与可视化接口
├── models/
│   ├── yolo/                     # YOLO 各任务的 Trainer、Validator、Predictor
│   │   ├── model.py              # YOLO 类及 task_map 任务分发
│   │   ├── detect/               # 检测任务 train、val、predict
│   │   ├── segment/              # 实例分割任务
│   │   ├── classify/             # 分类任务
│   │   ├── pose/                 # 姿态估计任务
│   │   └── obb/                  # 旋转框任务
│   └── rtdetr/                   # RT-DETR 家族的任务实现
├── nn/
│   ├── tasks.py                  # 模型类、权重加载和 parse_model YAML 解析器
│   ├── modules/
│   │   ├── block.py              # C2f、C3k2、SPPF、注意力等组合模块
│   │   ├── conv.py               # Conv、Concat 等基础模块
│   │   ├── head.py               # Detect、Segment、Pose、OBB 等任务头
│   │   └── __init__.py           # 模块导出入口
│   └── autobackend.py            # 多种导出格式的统一推理后端
├── utils/
│   ├── loss.py                   # 检测、分割、姿态等损失函数
│   ├── metrics.py                # 指标计算
│   ├── tal.py                    # 标签分配相关实现
│   └── callbacks/                # 日志和外部平台回调
└── trackers/                     # ByteTrack、BoT-SORT 等跟踪器
```

两个容易混淆的层次：

- `engine/` 提供任务无关的生命周期，例如 epoch 循环、保存 checkpoint 和回调。
- `models/yolo/<task>/` 提供任务相关行为，例如检测数据预处理、检测验证指标和检测预测后处理。

模型结构不在 `models/yolo/` 中定义。YAML 结构由 `nn/tasks.py` 构建，具体网络积木位于 `nn/modules/`。

## 2. 理解一次训练如何流动

以下代码虽然只有两行，但会经过完整的任务分发和模型构建过程：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.yaml")
model.train(data="coco8.yaml", epochs=10)
```

核心调用链如下：

```text
YOLO("yolo26n.yaml")
  └─ engine.Model 加载配置并识别 detect 任务
      └─ YOLO.task_map 选择 DetectionModel
          └─ DetectionModel 调用 parse_model()
              └─ 将 YAML 的 backbone + head 解析为 nn.Sequential

model.train(...)
  └─ engine.Model.train() 合并默认参数、模型参数和本次参数
      └─ YOLO.task_map 选择 DetectionTrainer
          ├─ 构建 Dataset / DataLoader
          ├─ 构建或接收 DetectionModel
          ├─ 创建损失、优化器和验证器
          └─ BaseTrainer 执行训练循环、验证和保存 checkpoint
```

记住这个定位规则：

| 研究问题                           | 首先查看的 owner                                            |
| ---------------------------------- | ----------------------------------------------------------- |
| 改 epoch、学习率、增强强度、冻结层 | `cfg/default.yaml` 和 `model.train()` 参数                  |
| 改 backbone、neck、head 的连接     | `cfg/models/` 中的模型 YAML                                 |
| 新增卷积、注意力、融合模块         | `nn/modules/` 和 `nn/tasks.py::parse_model()`               |
| 改检测损失或标签分配               | `utils/loss.py`、`utils/tal.py`                             |
| 改 batch 内容或数据增强            | `data/dataset.py`、`data/augment.py`                        |
| 改训练步骤、优化器或保存策略       | `engine/trainer.py` 或任务 Trainer                          |
| 新增验证指标或后处理               | 对应任务的 Validator 或 Predictor                           |
| 新增完整任务                       | 模型类以及 Trainer/Validator/Predictor，并注册到 `task_map` |

## 3. 选择最小的定制层级

科研代码越接近现有扩展点，越容易复现和升级。按下面顺序判断，能在前一层完成就不要进入后一层。

### 层级 A：只改训练配置

大多数基线、迁移学习和超参数实验无需修改源码：

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
metrics = model.train(
    data="path/to/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    optimizer="AdamW",
    lr0=0.001,
    seed=0,
    deterministic=True,
    project="runs/research",
    name="baseline",
)
```

使用 `.pt` 是在预训练权重上微调；使用 `.yaml` 是按结构初始化模型并从头训练。先跑预训练基线，再比较结构创新，通常更容易判断收益来自哪里。

### 层级 B：只改模型 YAML

模型 YAML 的每一层都采用统一格式：

```yaml
- [from, repeats, module, args]
```

- `from`：输入来自哪一层，`-1` 表示上一层。
- `repeats`：模块重复次数，会受深度缩放系数影响。
- `module`：模块名，例如 `Conv`、`C3k2`、`Concat`、`Detect`。
- `args`：传给模块的参数；常见模块的输入通道由解析器自动补入。

建议复制最接近目标任务的官方 YAML 到实验目录，然后一次只改一个变量。例如，将某个 backbone 块的重复次数从 2 改为 3：

```yaml
backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C3k2, [256, False, 0.25]] # 只改变这一项
```

构建后立即检查参数量、FLOPs 和前向传播：

```python
import torch

from ultralytics import YOLO

model = YOLO("path/to/my_yolo.yaml", task="detect", verbose=True)
model.info()
output = model.model(torch.zeros(1, 3, 640, 640))
print(type(output))
```

`model.info()` 显示 0 FLOPs 可能意味着前向传播失败，也可能是 THOP 未安装或不支持某个自定义算子。直接运行随机输入前向可区分这两类问题；如果前向也失败，先解决通道数、空间尺寸和 `from` 索引问题，再开始训练。

### 层级 C：新增网络模块

当 YAML 中已有模块无法表达你的结构时，再新增模块。一个接受输入/输出通道的最小模块可以写在 `nn/modules/block.py`：

```python
class ResearchBlock(nn.Module):
    """A minimal convolutional block for an ablation experiment."""

    def __init__(self, c1, c2, shortcut=True):
        """Initialize the block with input and output channels."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Apply the block and an optional residual connection."""
        y = self.cv1(x)
        return x + y if self.add else y
```

然后完成一条且只有一条注册链：

1. 在 `nn/modules/__init__.py` 导出 `ResearchBlock`。
2. 在 `nn/tasks.py` 从 `ultralytics.nn.modules` 导入它，使 YAML 名称可解析。
3. 若构造函数遵循 `(c1, c2, ...)`，将它加入 `parse_model()` 的 `base_modules` 集合，让解析器自动注入输入通道并缩放输出通道。
4. 只有当模块自己接收内部重复次数时，才把它加入 `repeat_modules`；否则使用 YAML 的外部 `repeats` 即可。

此时 YAML 可以直接引用模块名：

```yaml
- [-1, 1, ResearchBlock, [256, True]]
```

这里 YAML 的 `[256, True]` 会被 `parse_model()` 转换为构造参数 `(c1, c2, shortcut)`。如果你的模块是多输入、多输出或参数规则不同，应在 `parse_model()` 中增加一个与其真实语义对应的最小分支，不要伪装成普通 `base_modules`。

### 层级 D：定制训练、损失或数据管线

只需要在训练生命周期插入行为时，优先使用 callback；需要替换某个训练步骤时，继承任务 Trainer 并只覆盖对应方法。将自定义 Trainer 放在可导入的模块中：

```python
# research_trainer.py
from ultralytics.models.yolo.detect import DetectionTrainer


class ResearchTrainer(DetectionTrainer):
    """Detection trainer for one isolated research intervention."""

    def preprocess_batch(self, batch):
        """Reuse standard preprocessing before applying the intervention."""
        batch = super().preprocess_batch(batch)
        # 在这里加入确有必要的 batch 级处理
        return batch
```

再从训练脚本导入并传入这个类：

```python
# train.py
from research_trainer import ResearchTrainer

from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(data="path/to/dataset.yaml", trainer=ResearchTrainer)
```

独立模块对多 GPU 训练尤其重要：Ultralytics 会生成临时 DDP 启动脚本，并从 `trainer.__class__.__module__` 重新导入 Trainer。若类只定义在调用脚本的 `__main__` 中，子进程将无法导入它。

不要复制整个训练循环。继承最具体的任务 Trainer 并调用 `super()`，可以保留 AMP、DDP、EMA、断点续训、日志和验证等现有行为。

损失定制也应改在真正拥有损失计算的模型或 criterion 中，而不是在训练循环外重新计算一个无法反向传播的指标。先搜索当前任务如何创建 criterion，再替换最小方法。

## 4. 从零完成一次可复现实验

### 第一步：创建独立环境并以 editable 模式安装

```bash
git clone https://github.com/ultralytics/ultralytics.git
cd ultralytics
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e ".[dev,export-base]"
```

editable 安装保证你修改的本地源码就是 Python 实际导入的代码。用下面的命令确认没有误用全局安装：

```bash
python -c "import ultralytics; print(ultralytics.__file__)"
```

输出路径应指向当前克隆仓库中的 `ultralytics` 包。

### 第二步：准备数据 YAML

检测数据集的最小配置如下：

```yaml
path: /absolute/path/to/dataset
train: images/train
val: images/val

names:
  0: class_a
  1: class_b
```

对应目录示例：

```text
dataset/
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

先用官方小模型训练 1 个 epoch，确认标签、缓存、增强和验证都能工作，再投入完整算力。

### 第三步：建立基线

```bash
yolo detect train model=yolo26n.pt data=path/to/dataset.yaml epochs=1 imgsz=640 project=runs/research name=smoke
yolo detect train model=yolo26n.pt data=path/to/dataset.yaml epochs=100 imgsz=640 seed=0 deterministic=True project=runs/research name=baseline
```

记录代码提交、数据版本、随机种子、设备、依赖环境、训练参数以及 `best.pt` 对应指标。没有稳定基线，后续结构变化无法形成可信结论。

### 第四步：只引入一个研究变量

复制官方 YAML，替换一个模块或一条连接，保持数据划分、训练时长和超参数不变。至少先完成以下检查：

```python
import torch

from ultralytics import YOLO

model = YOLO("path/to/experiment.yaml", task="detect", verbose=True)
model.info()
model.model(torch.zeros(1, 3, 640, 640))
model.train(data="path/to/dataset.yaml", epochs=1, imgsz=640, name="experiment_smoke")
```

### 第五步：训练、验证和导出

```python
from ultralytics import YOLO

model = YOLO("path/to/experiment.yaml", task="detect")
model.train(
    data="path/to/dataset.yaml",
    epochs=100,
    seed=0,
    deterministic=True,
    project="runs/research",
    name="experiment_v1",
)

best = YOLO("runs/research/experiment_v1/weights/best.pt")
metrics = best.val(data="path/to/dataset.yaml")
best.export(format="onnx")
print(metrics.results_dict)
```

导出不是实验末尾的附属步骤。新算子可能可以在 PyTorch 中训练，却不被 ONNX、TensorRT 或目标硬件支持；如果论文目标包含部署，应尽早做一次目标格式导出。

## 5. 科研实验的最小验证清单

每次结构改动都按从便宜到昂贵的顺序验证：

1. **导入检查**：本地包能导入，新模块能从预期命名空间导入。
2. **构建检查**：YAML 能构建，verbose 输出中的通道和参数符合预期。
3. **前向检查**：随机输入可完成前向，输出结构符合任务头约定。
4. **反向检查**：一个 batch 能计算有限 loss，并完成 backward 和 optimizer step。
5. **冒烟训练**：在小数据集上训练 1 个 epoch，完成验证并生成 checkpoint。
6. **基线对照**：同一数据划分和训练配置下比较精度、参数量、FLOPs、显存与速度。
7. **部署检查**：若研究声明可部署，验证真实目标格式和硬件，而不只验证 PyTorch。

提交代码前，至少对改动文件运行格式与静态检查；仓库完整命令见 `AGENTS.md`。不要为了新模块复制测试框架或制造大量 mock，优先验证真实调用路径。

## 6. 常见问题与定位方法

| 现象                        | 常见原因                                   | 首先检查                                     |
| --------------------------- | ------------------------------------------ | -------------------------------------------- |
| `KeyError: 'ResearchBlock'` | 模块未进入 `nn/tasks.py` 的全局命名空间    | `modules/__init__.py` 导出和 `tasks.py` 导入 |
| 构造函数参数数量错误        | YAML 参数与 `parse_model()` 注入规则不一致 | 模块签名、`base_modules`、`repeat_modules`   |
| `Concat` 尺寸不一致         | 被连接特征的下采样倍率不同                 | YAML 的 `from` 索引和各层 stride             |
| 卷积通道不一致              | 模块报告的 `c2` 与真实输出不一致           | `parse_model()` 的通道推导和 forward 输出    |
| FLOPs 为 0                  | 模型前向传播异常                           | 用随机张量直接调用 `model.model(...)`        |
| 修改源码却不生效            | Python 导入了另一个安装位置                | `ultralytics.__file__` 和 editable 安装      |
| 预训练权重只加载一部分      | 新结构与 checkpoint 的层名或形状不匹配     | 加载日志以及匹配层数量                       |
| 单卡正常、多卡失败          | 自定义状态或张量未正确迁移/同步            | DDP 启动路径、device 和 rank 相关代码        |
| 训练正常、导出失败          | 新算子不受目标后端支持                     | 对应 `utils/export/` 格式实现与算子支持      |

调试时优先让错误暴露在最小前向或单 batch 中，不要用宽泛的 `try/except`、跳过首轮或额外状态标志隐藏问题。

## 7. 推荐阅读顺序

1. [Model YAML Configuration Guide](https://docs.ultralytics.com/guides/model-yaml-config/)：YAML 语法、模块解析和自定义模块注册。
2. [Train Mode](https://docs.ultralytics.com/modes/train/)：训练参数和常规训练流程。
3. [Advanced Customization](https://docs.ultralytics.com/usage/engine/)：Trainer、Validator 和 Predictor 的扩展方式。
4. [Customizing Trainer](https://docs.ultralytics.com/guides/custom-trainer/)：自定义指标、优化器、冻结策略等示例。
5. [Callbacks](https://docs.ultralytics.com/usage/callbacks/)：不改训练循环的生命周期扩展点。
6. [Configuration](https://docs.ultralytics.com/usage/cfg/)：完整配置参数说明。

最有效的源码阅读顺序是：先看一个官方模型 YAML，再看 `parse_model()` 如何构建它，然后看 `YOLO.task_map` 如何选择任务组件，最后只深入与你实验变量相关的 Trainer、loss、data 或 head。这样能在保留成熟训练基础设施的同时，把研究代码限制在一个清晰、可验证的改动面内。
