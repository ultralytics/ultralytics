# 通用训练、验证与预测脚本

三个独立 Python 脚本覆盖当前仓库支持的七类 YOLO 任务。通过 `--参数 值` 指定配置，无需修改脚本。
同目录的 `train.sh`、`val.sh` 和 `detect.sh` 分别启动训练、验证和目标检测：脚本顶部逐行列出默认配置，
运行时与命令行参数合并透传给 Python 脚本，同名参数以命令行为准，其余参数继续使用配置值。
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

也可以直接运行 shell 脚本，不带参数时完全使用脚本内的默认配置：

```bash
./tools/train.sh
./tools/val.sh
./tools/detect.sh
```

命令行参数会追加在配置之后合并透传，同名参数以命令行为准，例如只改训练轮数或临时换输入：

```bash
./tools/train.sh --epochs 50
./tools/val.sh --task segment --data "/absolute/path/to/dataset.yaml"
./tools/detect.sh --source "/absolute/path/to/video.mp4"
```

长期使用的默认值直接编辑各 sh 脚本顶部的"默认配置"区，每行一个参数；三个脚本均含 `MODEL` 配置，留空 `""` 使用 `TASK`
对应的内置预训练模型，填入 `.pt` 权重路径（如训练产出的 `best.pt`）即从该权重训练、验证或预测；均含 `PROJECT` 配置，
留空 `""` 输出到框架默认 `runs/<任务>/`，相对名追加在任务目录后，绝对路径直接作为项目目录；`NAME` 留空 `""` 使用各脚本
默认子目录名（`train`/`val`/`predict`），重名自动递增；`detect.sh` 的默认任务与输入源也在该区。
`train.sh` 的配置区已按上文「训练超参数」表列出全部超参数，取值与 `ultralytics/cfg/default.yaml` 一致，
修改后始终作为命令行覆盖项传入；设为 `None` 表示沿用框架可选默认，如 `TIME`、`FREEZE`、`CLASSES`。
配置中的 `TASK` 与 `DATA` 需保持匹配。三个 shell 脚本可从任意目录运行，参数格式与对应 Python 脚本完全一致。
若没有执行权限，也可使用 `bash tools/train.sh ...`、`bash tools/val.sh ...` 或 `bash tools/detect.sh ...`。

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
- `--batch`：训练和验证默认 `8`；训练还支持 `-1` 自动估算或 `0.7` 等显存比例，验证仍使用整数。
- `--workers`：训练和验证的数据加载进程数，整数，默认 `0`，可按运行环境增加。
- `--device`：省略时由框架选择；可填写 `cpu`、`0`（CUDA）或 `mps`（Apple Silicon）。
- `--project`：省略时沿用框架的输出目录设置；相对项目名自动放在对应任务目录下，绝对路径直接作为项目目录。
- `--name`：本次输出子目录名，三个脚本分别默认为 `train`、`val`、`predict`。
- `--split`：验证数据划分，默认 `val`，可选 `train`、`val`、`test`；数据集必须提供相应划分。

预测的布尔开关不需要附加 `True` 或 `False`：使用 `--show` 开启窗口显示，使用 `--no-save` 关闭保存。
训练的新增布尔参数需要显式值，例如 `--amp False`、`--cos_lr True`。
未知参数、缺失参数值和无效任务由参数解析器报错；训练超参数的类型和取值范围复用框架校验，在加载模型前检查。

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

## 训练超参数

`train.py --help` 列出全部支持的选项。新增参数与框架配置使用相同的下划线名称，按需传入即可：

| 类别           | 参数                                                                                                                                |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| 优化器与学习率 | `--optimizer`、`--lr0`、`--lrf`、`--momentum`、`--weight_decay`、`--cos_lr`、`--nbs`                                                |
| 预热           | `--warmup_epochs`、`--warmup_momentum`、`--warmup_bias_lr`                                                                          |
| 训练控制       | `--patience`、`--time`、`--seed`、`--deterministic`、`--amp`、`--cache`、`--freeze`、`--resume`、`--pretrained`                     |
| 数据与性能     | `--fraction`、`--rect`、`--single_cls`、`--classes`、`--multi_scale`、`--compile`、`--channels_last`、`--cls_remap`                 |
| 色彩与几何增强 | `--hsv_h`、`--hsv_s`、`--hsv_v`、`--degrees`、`--translate`、`--scale`、`--shear`、`--perspective`、`--flipud`、`--fliplr`、`--bgr` |
| 样本混合增强   | `--mosaic`、`--mixup`、`--cutmix`、`--copy_paste`、`--copy_paste_mode`、`--close_mosaic`                                            |
| 分割与分类     | `--overlap_mask`、`--mask_ratio`、`--auto_augment`、`--erasing`、`--dropout`                                                        |
| 损失权重       | `--box`、`--cls`、`--cls_pw`、`--dfl`、`--pose`、`--kobj`、`--rle`、`--angle`、`--dlog`、`--dgrad`、`--dlam`                        |
| 蒸馏           | `--distill_model`、`--dis`                                                                                                          |
| 保存与日志     | `--save`、`--save_period`、`--plots`、`--val`、`--verbose`、`--exist_ok`                                                            |

数值支持小数和科学计数法；布尔值使用 `True` / `False`，大小写均可。
列表和元组应使用引号，例如 `--freeze '[0,1,2]'`、`--classes '[0,2]'`、`--scale '(0.5,1.5)'`。
混合类型参数沿用框架语义，例如 `--cache disk`、`--amp bf16`、`--pretrained False`。
任务专属参数仅在对应任务中生效；设备相关功能也取决于框架和运行环境的支持。

新增超参数未传入时不会作为命令行覆盖项传给模型；具体取值由框架默认配置、模型设置或续训检查点决定。
帮助中的默认值来自 `ultralytics/cfg/default.yaml`；原有基础参数仍沿用脚本默认值。

例如，手动指定优化器、学习率、增强和早停：

```bash
python tools/train.py --task detect --data coco8.yaml --epochs 100 --device 0 \
  --optimizer AdamW --lr0 0.001 --lrf 0.01 --weight_decay 0.0005 \
  --warmup_epochs 3 --cos_lr True --patience 30 --seed 42 \
  --mosaic 1.0 --mixup 0.1 --close_mosaic 10 --amp True
```

希望手动控制学习率和动量时，应显式选择优化器；`--optimizer auto` 会由框架自动决定这些值。
AutoBatch 依赖支持的训练设备；CPU、MPS 等环境可能回退到固定批次。

对中断且仍保留优化器状态的训练，可使用原始 `last.pt` 续训：

```bash
python tools/train.py --model "/absolute/path/to/run/weights/last.pt" --resume True --device 0
```

续训由框架恢复原来的训练配置，部分命令行超参数不会覆盖检查点设置。
训练已完成、被裁剪或没有优化器状态的权重不能按中断训练恢复；框架会提示并按其现有规则处理。

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
