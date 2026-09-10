# 通用训练、验证与预测脚本

三个独立 Python 脚本覆盖当前仓库支持的七类 YOLO 任务。通过 `--参数 值` 指定配置，无需修改脚本。
脚本直接调用仓库的 `YOLO` 接口，数据读取、任务识别、指标计算和可视化均由框架处理。

## 安装与运行

在当前仓库根目录激活 Python 环境，然后安装本地代码：

```bash
python -m pip install -e .
```

查看各脚本支持的参数：

```bash
python tools/train.py --help
python tools/val.py --help
python tools/predict.py --help
```

分别运行需要的步骤，例如使用示例数据进行一轮 CPU 训练：

```bash
python tools/train.py --task detect --epochs 1 --device cpu
python tools/val.py --task detect --device cpu
python tools/predict.py --task detect --device cpu
```

省略参数时沿用默认配置，训练默认 100 轮。以上验证和预测命令使用默认预训练模型；使用训练产物的方法见下文。
默认模型和示例数据在本地缺失时会由框架下载，需要网络连接。

## 选择任务与数据

三个脚本均支持 `--task`，默认 `detect`。不传 `--model` 时，使用对应任务的仓库内置模型。
训练和验证不传 `--data` 时，使用加载后模型对应任务的示例数据：

| --task     | 任务       | 默认模型           | 默认数据           |
| ---------- | ---------- | ------------------ | ------------------ |
| `detect`   | 目标检测   | `yolo26n.pt`       | `coco8.yaml`       |
| `segment`  | 实例分割   | `yolo26n-seg.pt`   | `coco8-seg.yaml`   |
| `semantic` | 语义分割   | `yolo26n-sem.pt`   | `cityscapes8.yaml` |
| `depth`    | 深度估计   | `yolo26n-depth.pt` | `depth8.yaml`      |
| `classify` | 图像分类   | `yolo26n-cls.pt`   | `imagenet10`       |
| `pose`     | 姿态估计   | `yolo26n-pose.pt`  | `coco8-pose.yaml`  |
| `obb`      | 旋转框检测 | `yolo26n-obb.pt`   | `dota8.yaml`       |

例如，使用自己的数据训练实例分割：

```bash
python tools/train.py --task segment --data "/absolute/path/to/dataset.yaml" --epochs 100 --batch 8 --device 0
```

`--model` 可以指定已有 `.pt` 权重；训练还支持模型 `.yaml`，从模型结构开始训练。
指定模型后，任务由模型自身识别，`--task` 仅在未指定模型时选择默认模型。
自定义数据必须匹配模型任务；使用自己的数据时，务必在训练和验证命令中都传入 `--data`。

数据格式按任务区分：

- `detect`：数据集 YAML 指向图片；对应 TXT 每行是类别及归一化的中心坐标、宽、高。
- `segment`：数据集 YAML 指向图片；对应 TXT 每行是类别及归一化多边形顶点。
- `semantic`：数据集 YAML 指定图片与语义掩码配置；掩码像素值表示类别，格式参考 `cityscapes8.yaml`。
- `depth`：数据集 YAML 指定 RGB 图片及深度配置；对应深度图为 16 位 PNG，`depth_scale` 指定米制换算比例，格式参考 `depth8.yaml`。
- `classify`：`--data` 是数据集目录，按 `train/类别名/图片` 和 `val/类别名/图片` 组织，无需数据集 YAML。
- `pose`：数据集 YAML 包含 `kpt_shape` 等配置；对应 TXT 包含类别、框及关键点，格式参考 `coco8-pose.yaml`。
- `obb`：数据集 YAML 指向图片；对应 TXT 每行是类别及旋转框的四个归一化顶点。

可复制 `ultralytics/cfg/datasets/` 内对应任务的示例 YAML，再修改路径与类别。
自定义 YAML 建议使用绝对 `path`，其中 `train`、`val` 相对此数据集根目录解析；分类直接使用绝对目录路径。

## 常用配置与输出

可通过命令行覆盖以下参数，`--help` 会列出当前脚本支持的选项：

- `--epochs`：训练轮数，整数，默认 `100`。
- `--imgsz`：输入尺寸，整数，默认 `640`；分类可按需改为 `224`。
- `--batch`：训练和验证批次大小，整数，默认 `8`；内存不足时调小。
- `--workers`：训练和验证的数据加载进程数，整数，默认 `0`，可按运行环境增加。
- `--device`：省略时由框架选择；可填写 `cpu`、`0`（CUDA）或 `mps`（Apple Silicon）。
- `--project`：省略时沿用框架的输出目录设置；相对项目名自动放在对应任务目录下，绝对路径直接作为项目目录。
- `--name`：本次输出子目录名，三个脚本分别默认为 `train`、`val`、`predict`。
- `--split`：验证数据划分，默认 `val`，可选 `train`、`val`、`test`；数据集必须提供相应划分。

布尔开关不需要附加 `True` 或 `False`：预测时使用 `--show` 开启窗口显示，使用 `--no-save` 关闭保存。
未知参数、缺失参数值、非法整数和无效任务会在加载模型前由参数解析器报错。

默认输出通常位于 `runs/<任务>/<模式>/`，实际根目录取决于 Ultralytics 的 `runs_dir` 设置。
已有目录会自动递增，例如 `train-2`；以运行日志打印的实际路径为准。
设置绝对 `--project` 时，请在路径中自行区分任务，例如 `/absolute/path/to/runs/segment`。

训练结束会打印输出目录和最佳权重路径。验证和预测时，将实际路径传给 `--model`：

```bash
python tools/val.py --model "/absolute/path/to/runs/segment/train-2/weights/best.pt" --data "/absolute/path/to/dataset.yaml"
python tools/predict.py --model "/absolute/path/to/runs/segment/train-2/weights/best.pt" --source "/absolute/path/to/image.jpg"
```

验证时传入训练使用的 `--data`。路径可以包含空格，按上例使用引号包裹；其他相对路径相对于运行命令的工作目录。
验证输出 `metrics.results_dict`，由任务决定指标，例如检测 mAP、语义分割 mIoU、深度 delta1、分类准确率。

## 图片、视频与摄像头预测

通过 `--source` 指定输入，例如：

```bash
python tools/predict.py --source "/absolute/path/to/image.jpg"
python tools/predict.py --source "/absolute/path/to/images"
python tools/predict.py --source "/absolute/path/to/video.mp4"
python tools/predict.py --source 0 --show --no-save
```

默认输入为仓库自带示例图片，保存可视化且不弹出窗口。
需要实时查看时添加 `--show`；摄像头需要设备访问权限，按 Ctrl+C 停止。
显示和视频保存使用框架及当前环境支持的后端。

脚本逐个消费 `stream=True` 返回的 `Results`，避免将整段视频的预测结果保存在列表中。
如需后续处理，可在循环内替换 `pass`，读取当前 `result` 的 `boxes`、`masks`、`semantic_mask`、
`depth`、`probs`、`keypoints` 或 `obb`；可用字段取决于模型任务。
默认仅保存框架生成的可视化图片或视频，不额外导出原始数组。
