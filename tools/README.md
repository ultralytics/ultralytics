# 通用训练、验证与预测脚本

三个独立 Python 脚本覆盖当前仓库支持的七类 YOLO 任务。在各脚本顶部修改配置后运行，无需命令行参数。
脚本直接调用仓库的 `YOLO` 接口，数据读取、任务识别、指标计算和可视化均由框架处理。

## 安装与运行

在当前仓库根目录激活 Python 环境，然后安装本地代码：

```bash
python -m pip install -e .
```

分别运行需要的步骤：

```bash
python tools/train.py
python tools/val.py
python tools/predict.py
```

默认训练为 100 轮。首次检查流程时，先将 `train.py` 中 `ARGS` 的 `epochs` 改为 `1`。
默认模型和示例数据在本地缺失时会由框架下载，需要网络连接。

## 选择任务与数据

每个脚本独立配置。保留 `MODEL = None`，修改 `TASK` 即可选择仓库内置模型。
训练和验证的 `DATA = None` 使用加载后模型对应任务的示例数据：

| TASK       | 任务       | 默认模型           | 默认数据           |
| ---------- | ---------- | ------------------ | ------------------ |
| `detect`   | 目标检测   | `yolo26n.pt`       | `coco8.yaml`       |
| `segment`  | 实例分割   | `yolo26n-seg.pt`   | `coco8-seg.yaml`   |
| `semantic` | 语义分割   | `yolo26n-sem.pt`   | `cityscapes8.yaml` |
| `depth`    | 深度估计   | `yolo26n-depth.pt` | `depth8.yaml`      |
| `classify` | 图像分类   | `yolo26n-cls.pt`   | `imagenet10`       |
| `pose`     | 姿态估计   | `yolo26n-pose.pt`  | `coco8-pose.yaml`  |
| `obb`      | 旋转框检测 | `yolo26n-obb.pt`   | `dota8.yaml`       |

例如，训练实例分割只需修改 `train.py`：

```python
TASK = "segment"
MODEL = None
DATA = "/absolute/path/to/dataset.yaml"
```

`MODEL` 也可以填写已有 `.pt` 权重；训练还支持模型 `.yaml`，从模型结构开始训练。
指定模型后，任务由模型自身识别，`TASK` 仅在 `MODEL = None` 时选择默认模型。
自定义数据必须匹配模型任务；使用自己的数据时，务必在训练和验证脚本中都填写 `DATA`。

数据格式按任务区分：

- `detect`：数据集 YAML 指向图片；对应 TXT 每行是类别及归一化的中心坐标、宽、高。
- `segment`：数据集 YAML 指向图片；对应 TXT 每行是类别及归一化多边形顶点。
- `semantic`：数据集 YAML 指定图片与语义掩码配置；掩码像素值表示类别，格式参考 `cityscapes8.yaml`。
- `depth`：数据集 YAML 指定 RGB 图片及深度配置；对应深度图为 16 位 PNG，`depth_scale` 指定米制换算比例，格式参考 `depth8.yaml`。
- `classify`：`DATA` 是数据集目录，按 `train/类别名/图片` 和 `val/类别名/图片` 组织，无需数据集 YAML。
- `pose`：数据集 YAML 包含 `kpt_shape` 等配置；对应 TXT 包含类别、框及关键点，格式参考 `coco8-pose.yaml`。
- `obb`：数据集 YAML 指向图片；对应 TXT 每行是类别及旋转框的四个归一化顶点。

可复制 `ultralytics/cfg/datasets/` 内对应任务的示例 YAML，再修改路径与类别。
自定义 YAML 建议使用绝对 `path`，其中 `train`、`val` 相对此数据集根目录解析；分类直接使用绝对目录路径。

## 常用配置与输出

直接修改脚本顶部的 `ARGS`：

- `epochs`：训练轮数，默认 `100`。
- `imgsz`：输入尺寸，默认 `640`；分类可按需改为 `224`。
- `batch`：训练和验证批次大小，默认 `8`；内存不足时调小。
- `workers`：数据加载进程数，默认 `0`，可按运行环境增加。
- `device`：默认 `None`，由框架选择；可填写 `"cpu"`、`0`（CUDA）或 `"mps"`（Apple Silicon）。
- `project`：默认 `None`，沿用框架的输出目录设置；相对项目名自动放在对应任务目录下，绝对路径直接作为项目目录。
- `name`：本次输出子目录名，三个脚本分别默认为 `train`、`val`、`predict`。
- `split`：验证数据划分，默认 `"val"`；仅在数据集提供相应划分时改用 `"test"`。

默认输出通常位于 `runs/<任务>/<模式>/`，实际根目录取决于 Ultralytics 的 `runs_dir` 设置。
已有目录会自动递增，例如 `train2`；以运行日志打印的实际路径为准。
设置绝对 `project` 时，请在路径中自行区分任务，例如 `/absolute/path/to/runs/segment`。

训练结束会打印输出目录和最佳权重路径。将实际路径填入 `val.py` 和 `predict.py`：

```python
MODEL = "/absolute/path/to/runs/segment/train2/weights/best.pt"
```

同时在 `val.py` 中填写训练使用的 `DATA`，再运行验证和预测。路径可以包含空格。
验证输出 `metrics.results_dict`，由任务决定指标，例如检测 mAP、语义分割 mIoU、深度 delta1、分类准确率。

## 图片、视频与摄像头预测

修改 `predict.py` 中的 `SOURCE`，例如：

```python
SOURCE = "/absolute/path/to/image.jpg"  # 单张图片
SOURCE = "/absolute/path/to/images"  # 图片目录
SOURCE = "/absolute/path/to/video.mp4"  # 视频
SOURCE = 0  # 摄像头
```

默认输入为仓库自带示例图片，`save=True` 保存可视化，`show=False` 不弹出窗口。
需要实时查看时，将 `ARGS` 中的 `show` 改为 `True`；摄像头需要设备访问权限，按 Ctrl+C 停止。
显示和视频保存使用框架及当前环境支持的后端。

脚本逐个消费 `stream=True` 返回的 `Results`，避免将整段视频的预测结果保存在列表中。
如需后续处理，可在循环内替换 `pass`，读取当前 `result` 的 `boxes`、`masks`、`semantic_mask`、
`depth`、`probs`、`keypoints` 或 `obb`；可用字段取决于模型任务。
默认仅保存框架生成的可视化图片或视频，不额外导出原始数组。
