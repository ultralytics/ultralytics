# YOLOv8 代码阅读注释

这份笔记按你当前训练脚本的执行顺序来读源码。目标不是解释仓库里所有文件，而是抓住 YOLOv8 检测算法的主线：数据怎么进来，模型怎么建，前向怎么预测，loss 怎么算，mAP/Ra/Fa 怎么出来。

## 1. 你的训练入口

文件：`train_infrared.py`

重点入口：

```python
model = YOLO(args.model)
results = model.train(...)
```

第一行加载模型。你传的是 `yolov8n.pt`，所以 Ultralytics 会加载 YOLOv8n 的网络结构和预训练权重。

第二行开始训练。这里传入的数据集 yaml、epoch、batch、imgsz、device 等参数，会一路传到内部的 Trainer。

这个文件还做了三件和原始 Ultralytics 不同的事：

- 自动生成 `infrared_dataset/train.txt`、`val.txt`、`infrared.yaml`
- 注册 callback，从验证阶段缓存 TP/FP 统计
- 额外输出 `epoch_metrics_ra_fa.csv`

## 2. YOLO 高层封装

文件：`ultralytics/engine/model.py`

重点函数：

```python
def train(...)
def val(...)
def predict(...)
def add_callback(...)
```

`YOLO(...).train(...)` 最终会走到 `Model.train()`。它不是直接写训练循环，而是负责：

- 合并默认参数和你传入的参数
- 判断当前任务是 detect、segment、classify 还是 pose
- 创建对应的 Trainer
- 调用 Trainer 的训练逻辑

你在 `train_infrared.py` 里用的：

```python
model.add_callback("on_val_batch_end", collect_val_stats)
```

也是在这个类里注册的。callback 的作用是让你不改核心训练循环，也能在指定时机插入自己的代码。

## 3. 训练主循环

文件：`ultralytics/engine/trainer.py`

重点函数：

```python
def _do_train(self):
```

这个函数是训练流程的核心。可以把它看成下面的伪代码：

```text
准备模型、优化器、数据加载器

for epoch in epochs:
    设置模型为训练模式
    for batch in train_loader:
        读取一批图像和标签
        图像预处理
        前向传播，得到预测
        计算 loss
        反向传播
        优化器更新参数
        更新进度条里的 box_loss / cls_loss / dfl_loss

    运行验证集
    保存 results.csv
    保存 last.pt / best.pt
```

你看到的这段代码是关键：

```python
batch = self.preprocess_batch(batch)
loss, self.loss_items = self.model(batch)
self.loss = loss.sum()
self.scaler.scale(self.loss).backward()
self.optimizer_step()
```

含义：

- `preprocess_batch`：把图片转到 GPU/CPU，并归一化到 0 到 1
- `self.model(batch)`：模型前向，同时计算检测 loss
- `backward()`：反向传播，计算梯度
- `optimizer_step()`：更新模型参数

每个 epoch 结束后：

```python
self.metrics, self.fitness = self.validate()
self.save_metrics(...)
self.save_model()
```

含义：

- 跑验证集
- 保存 mAP、loss 等指标到 `results.csv`
- 保存模型权重到 `weights/`

## 4. 检测任务专用 Trainer

文件：`ultralytics/models/yolo/detect/train.py`

重点类：

```python
class DetectionTrainer(BaseTrainer)
```

它继承通用的 `BaseTrainer`，专门处理检测任务。

重点函数：

```python
def build_dataset(...)
def get_dataloader(...)
def preprocess_batch(...)
def get_model(...)
def get_validator(...)
def label_loss_items(...)
```

这些函数分别负责：

- `build_dataset`：构建 YOLO 检测数据集
- `get_dataloader`：把数据集封装成 PyTorch DataLoader
- `preprocess_batch`：把图像转成 float 并除以 255
- `get_model`：创建 `DetectionModel`
- `get_validator`：创建检测验证器
- `label_loss_items`：把 loss 张量命名成 `box_loss`、`cls_loss`、`dfl_loss`

你在训练日志里看到：

```text
box_loss cls_loss dfl_loss
```

就是这里命名的。

## 5. 模型结构 YAML

文件：`ultralytics/cfg/models/v8/yolov8.yaml`

这是 YOLOv8 结构的蓝图。它不是 Python 代码，但非常重要。

核心结构：

```yaml
backbone:
  Conv
  C2f
  SPPF

head:
  Upsample
  Concat
  C2f
  Detect
```

最后一行：

```yaml
- [[15, 18, 21], 1, Detect, [nc]]
```

意思是检测头使用 3 个尺度的特征图：

- 15：P3/8，小目标尺度
- 18：P4/16，中目标尺度
- 21：P5/32，大目标尺度

对红外小目标来说，P3 很关键，因为它的空间分辨率最高。默认 YOLOv8 没有 P2，所以极小目标可能在下采样过程中丢失。

## 6. YAML 如何变成模型

文件：`ultralytics/nn/tasks.py`

重点函数：

```python
def parse_model(d, ch, verbose=True):
```

这个函数会把 `yolov8.yaml` 解析成真正的 PyTorch 网络层。yaml 里的每一行，比如：

```yaml
- [-1, 1, Conv, [64, 3, 2]]
```

会被解析成一个 `Conv` 模块。

重点类：

```python
class DetectionModel(BaseModel)
```

它负责：

- 保存模型结构
- 执行前向传播
- 初始化检测 loss

关键代码：

```python
def init_criterion(self):
    return v8DetectionLoss(self)
```

这说明 YOLOv8 检测任务使用的核心 loss 是 `v8DetectionLoss`。

## 7. Backbone 和 Neck 模块

文件：`ultralytics/nn/modules/block.py`

重点模块：

```python
class C2f
class SPPF
```

`C2f` 是 YOLOv8 的核心特征提取模块。它大致做：

```text
输入特征
  ↓
卷积分成两部分
  ↓
一部分不断经过 Bottleneck
  ↓
把多路特征拼接
  ↓
再卷积融合
```

这能在计算量不太大的情况下增强特征表达。

`SPPF` 是快速空间金字塔池化。它通过多次池化让网络看到更大范围的上下文。对红外图像来说，一个亮点是不是目标，经常需要结合周围背景判断。

## 8. Detect 检测头

文件：`ultralytics/nn/modules/head.py`

重点类：

```python
class Detect(nn.Module)
```

检测头接收 P3/P4/P5 三个尺度的特征图，然后分别预测：

- 边框位置
- 类别分数

核心变量：

```python
self.nc = nc
self.nl = len(ch)
self.reg_max = reg_max
self.no = nc + self.reg_max * 4
```

含义：

- `nc`：类别数，你这里是 1
- `nl`：检测尺度数量，YOLOv8 默认是 3
- `reg_max`：DFL 边框分布的离散桶数量
- `no`：每个位置输出的通道数

关键函数：

```python
def forward_head(...)
```

这里分别走：

- `box_head`：预测边框分布
- `cls_head`：预测类别分数

训练时返回原始预测：

```python
if self.training:
    return preds
```

推理时会解码边框：

```python
y = self._inference(...)
```

这就是训练和推理输出形式不同的原因。

## 9. YOLOv8 检测损失

文件：`ultralytics/utils/loss.py`

重点类：

```python
class v8DetectionLoss
class BboxLoss
class DFLoss
```

`v8DetectionLoss` 输出三项：

```text
box_loss
cls_loss
dfl_loss
```

核心流程：

```text
模型输出预测框和类别分数
  ↓
根据真实标签生成 targets
  ↓
TaskAlignedAssigner 分配正样本
  ↓
计算分类损失
  ↓
计算边框损失
  ↓
计算 DFL 损失
```

关键代码：

```python
self.assigner = TaskAlignedAssigner(...)
self.bbox_loss = BboxLoss(...)
```

`TaskAlignedAssigner` 很重要。它决定“哪个预测点负责哪个真实目标”。如果没有这个步骤，模型不知道海量预测点里哪些应该被当成正样本。

分类损失：

```python
bce_loss = self.bce(pred_scores, target_scores)
loss[1] = bce_loss.sum() / target_scores_sum
```

边框和 DFL 损失：

```python
loss[0], loss[2] = self.bbox_loss(...)
```

最后乘上超参数权重：

```python
loss[0] *= self.hyp.box
loss[1] *= self.hyp.cls
loss[2] *= self.hyp.dfl
```

所以 `args.yaml` 里的：

```yaml
box: 7.5
cls: 0.5
dfl: 1.5
```

会直接影响三项损失在总 loss 里的权重。

## 10. 验证、NMS 和 mAP

文件：`ultralytics/models/yolo/detect/val.py`

重点类：

```python
class DetectionValidator
```

重点函数：

```python
def preprocess(...)
def postprocess(...)
def update_metrics(...)
def get_stats(...)
def _process_batch(...)
```

验证流程：

```text
读取验证图片
  ↓
模型预测
  ↓
NMS 去掉重复框
  ↓
预测框和真实框计算 IoU
  ↓
统计 TP / FP / FN
  ↓
计算 precision / recall / mAP
```

`postprocess` 里会调用 NMS：

```python
nms.non_max_suppression(...)
```

`update_metrics` 会把每张图的预测和真实标签进行匹配。

`get_stats` 会调用：

```python
self.metrics.process(...)
```

这里会进入 mAP 的统计逻辑。

## 11. mAP / Precision / Recall 计算

文件：`ultralytics/utils/metrics.py`

重点函数和类：

```python
box_iou(...)
bbox_iou(...)
ap_per_class(...)
DetMetrics
ConfusionMatrix
```

`ap_per_class` 是 mAP 的核心。它根据：

- `tp`：预测是否正确
- `conf`：预测置信度
- `pred_cls`：预测类别
- `target_cls`：真实类别

计算：

- Precision
- Recall
- AP50
- AP50-95

你在 `results.csv` 里看到的：

```text
metrics/precision(B)
metrics/recall(B)
metrics/mAP50(B)
metrics/mAP50-95(B)
```

就是这里算出来的。

## 12. 数据读取和增强

建议看这些文件：

- `ultralytics/data/build.py`
- `ultralytics/data/dataset.py`
- `ultralytics/data/augment.py`

重点理解：

- 图片如何读取
- label txt 如何匹配到 image
- mosaic、flip、scale 等增强如何作用在图像和框上
- 多张图片如何组成 batch

你当前的数据是：

```text
NUAA-SIRST/images/*.png
NUAA-SIRST/labels/*.txt
NUDT-SIRST/images/*.png
NUDT-SIRST/labels/*.txt
```

Ultralytics 会根据图片路径自动寻找对应标签。

## 13. 最推荐的阅读顺序

第一次读建议这样：

1. `train_infrared.py`
2. `ultralytics/engine/model.py`
3. `ultralytics/engine/trainer.py`
4. `ultralytics/models/yolo/detect/train.py`
5. `ultralytics/cfg/models/v8/yolov8.yaml`
6. `ultralytics/nn/tasks.py`
7. `ultralytics/nn/modules/head.py`
8. `ultralytics/utils/loss.py`
9. `ultralytics/models/yolo/detect/val.py`
10. `ultralytics/utils/metrics.py`

如果时间不多，优先看 3、5、7、8、9。它们分别对应训练循环、模型结构、检测头、损失函数和验证指标。

## 14. 对红外小目标要特别注意什么

YOLOv8 默认检测头是 P3/P4/P5。对于普通目标足够，但红外小目标经常只有几个像素，经过多次下采样后信息容易丢失。

所以你后续研究时重点关注：

- `yolov8.yaml` 里 Detect 输入层是不是只有 `[15, 18, 21]`
- 是否需要加入 P2 检测头
- `imgsz` 是否太小
- mosaic、scale 等增强是否会让小目标更难学
- `loss.py` 中正样本分配对极小框是否友好
- 验证时 `conf` 阈值如何影响 Ra/Fa

这几个点就是红外小目标检测改进的主要入口。
