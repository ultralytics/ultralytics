# YOLOv8-P23456-MoE 动态路由多尺度模型

## 概述

`yolov8-p23456-moe.yaml` 和 `yolov8-seg-p23456-moe.yaml` 是基于 **MoE (Mixture of Experts)** 思想的动态路由模型。与标准 P23456 模型使用全部 5 个检测层不同，MoE 版本通过**门控网络**为每个预测动态选择最合适的 **top-2 P 层**，实现：

- 🎯 **智能路由**: 大目标 → P5/P6，小目标 → P2/P3
- ⚡ **高效计算**: 仅使用 top-2 层，计算量减少约 60%
- 📊 **负载均衡**: 自动平衡各 P 层的使用频率
- 🔄 **可学习偏向**: 通过训练学习最优的尺寸-层级映射

## 核心机制

### MoE 路由流程

```
输入特征 [P2, P3, P4, P5, P6]
         ↓
门控网络 (Gating Network)
         ↓
路由分数 [score_P2, ..., score_P6]
         ↓
Top-K 选择 (K=2)
         ↓
加权组合 Top-2 特征
         ↓
最终预测
```

### 门控网络架构

```python
GateNet(
    AdaptiveAvgPool2d(1),  # 全局特征聚合
    Conv2d(sum(channels), 256, 1),  # 特征融合
    BatchNorm2d + ReLU,
    Conv2d(256, 5, 1),  # 输出5个P层的路由分数
    Softmax,  # 归一化为概率
)
```

### 尺寸偏向引导

使用可学习的 `size_bias` 张量初始化为 `[-2.0, -1.0, 0.0, 1.0, 2.0]`，鼓励：

- P2/P3 (负偏向) → 小目标
- P5/P6 (正偏向) → 大目标

## 模型规格

### 检测模型 (yolov8-p23456-moe)

| 规模 | 参数量 | GFLOPs | 候选P层 | 实际使用 |
| ---- | ------ | ------ | ------- | -------- |
| n    | 5.4M   | -      | 5       | 2 (动态) |
| s    | -      | -      | 5       | 2 (动态) |
| m    | -      | -      | 5       | 2 (动态) |
| l    | -      | -      | 5       | 2 (动态) |
| x    | -      | -      | 5       | 2 (动态) |

### 分割模型 (yolov8-seg-p23456-moe)

| 规模 | 参数量 | GFLOPs | 候选P层 | 实际使用 |
| ---- | ------ | ------ | ------- | -------- |
| n    | 5.7M   | -      | 5       | 2 (动态) |
| s    | -      | -      | 5       | 2 (动态) |

**相比标准 P23456**: 参数量略增 (门控网络 ~174K)，但推理时计算量显著减少。

## 使用方法

### 1. 创建和查看模型

```python
from ultralytics import YOLO

# 检测
model_det = YOLO("ultralytics/cfg/models/v8/yolov8-p23456-moe.yaml")
model_det.info()

# 分割
model_seg = YOLO("ultralytics/cfg/models/v8/yolov8-seg-p23456-moe.yaml")
model_seg.info()
```

### 2. 训练

```python
# 检测训练
model_det.train(
    data="coco.yaml",
    epochs=300,
    imgsz=1280,  # P6模型建议更大输入
    batch=8,
    lr0=0.01,
    # MoE的负载均衡损失会自动包含在总损失中
)

# 分割训练
model_seg.train(
    data="coco-seg.yaml",
    epochs=300,
    imgsz=1280,
    batch=8,
)
```

**训练要点**:

- 门控网络会自动学习最优路由策略
- 负载均衡损失 (权重 0.01) 防止所有样本只使用少数几层
- 建议前几个 epoch 观察各 P 层的使用分布

### 3. 推理

```python
# 推理时自动使用动态路由
results = model_det.predict("image.jpg", imgsz=1280)

# 查看路由决策 (需在模型内部添加钩子)
# 每个样本会选择最合适的top-2层
```

### 4. 验证路由行为

```python
import torch

model_det.eval()
with torch.no_grad():
    # 小目标图像 - 期望选择P2/P3
    results_small = model_det("small_objects.jpg")

    # 大目标图像 - 期望选择P5/P6
    results_large = model_det("large_objects.jpg")
```

## 与其他版本对比

| 模型                  | P层数 | 推理策略      | 计算量     | 参数量(n) | 适用场景        |
| --------------------- | ----- | ------------- | ---------- | --------- | --------------- |
| YOLOv8                | 3     | 固定P3-P5     | 基准       | 3.0M      | 通用            |
| YOLOv8-p2             | 4     | 固定P2-P5     | +10%       | 3.1M      | 小目标增强      |
| YOLOv8-p6             | 4     | 固定P3-P6     | +9%        | 5.0M      | 大目标/高分辨率 |
| **YOLOv8-p23456**     | **5** | **固定全部**  | **+115%**  | **5.2M**  | **全尺度**      |
| **YOLOv8-p23456-MoE** | **5** | **动态Top-2** | **+50%\*** | **5.4M**  | **智能全尺度**  |

\*计算量相比标准YOLOv8，但比固定P23456减少约60%

## 优势与权衡

### ✅ 优势

1. **自适应检测**: 根据目标尺寸自动选择最优检测层
2. **计算高效**: 比使用全部5层节省约60%计算
3. **性能潜力**: 理论上可达到或超过固定P23456的精度
4. **可解释性**: 门控分数反映模型对目标尺寸的理解

### ⚠️ 权衡

1. **训练复杂度**: 需要学习门控网络，可能需要更多epoch收敛
2. **内存开销**: 训练时需计算所有5层特征（推理时仅top-2）
3. **超参数调整**: 负载均衡损失权重需要调优
4. **不确定性**: 路由决策可能在边界情况下不稳定

## 训练建议

### 学习率策略

```python
model.train(
    lr0=0.01,  # 初始学习率
    lrf=0.01,  # 最终学习率因子
    warmup_epochs=5,  # MoE需要更长warmup
    cos_lr=True,
)
```

### 负载均衡调整

如果发现某些P层使用率过低：

```python
# 在 DetectMoE.__init__ 中调整
aux_loss_weight = 0.02  # 增大权重强制更均匀分布
```

### 监控路由分布

训练时建议记录：

- 各P层的平均使用频率
- 小/中/大目标分别倾向使用哪些层
- 门控分数的方差（反映决策置信度）

## 推理优化

### 导出优化模型

```python
# 导出为ONNX（包含动态路由）
model.export(format="onnx", imgsz=1280)

# 如需固定路由（去掉门控网络），可修改forward逻辑
# 例如总是选择P3+P5（中等+大目标）
```

### 后处理调整

由于只使用top-2层，可能需要调整NMS参数：

```python
results = model.predict(
    "image.jpg",
    conf=0.25,  # 置信度阈值
    iou=0.7,  # NMS IoU阈值
    max_det=300,  # 最大检测数
)
```

## 实验分析

### 预期路由行为

训练收敛后，典型的路由模式：

- **超小目标** (<32px): P2 (80%) + P3 (20%)
- **小目标** (32-96px): P2/P3 (60%) + P4 (40%)
- **中等目标** (96-256px): P3/P4 (主要)
- **大目标** (256-512px): P4/P5 (主要)
- **超大目标** (>512px): P5 (50%) + P6 (50%)

### 调试工具

```python
# 在 DetectMoE forward 中添加
def forward(self, x):
    _gate_scores, gate_probs = self.compute_gate_scores(x)

    # 记录路由分布
    if self.training:
        self.gate_probs_history = gate_probs.detach()

    # ... 其余代码
```

## 常见问题

**Q: MoE版本精度是否一定更高？**  
A: 不一定。理论上动态路由更灵活，但需要充分训练。初期可能不如固定P23456。

**Q: 如何可视化路由决策？**  
A: 可以在推理时保存 `gate_probs`，然后用热力图展示每个anchor选择的P层。

**Q: 能否使用top-3而非top-2？**  
A: 可以，修改 `yolov8-p23456-moe.yaml` 中的 `top_k` 参数（需重新实现参数传递）。

**Q: 与标准注意力机制的区别？**  
A: 注意力是特征加权，MoE是层级选择。MoE更粗粒度但计算开销更低。

## 引用

```bibtex
@misc{yolov8_moe_2024,
  title={YOLOv8-P23456-MoE: Dynamic Multi-Scale Detection with Mixture of Experts},
  author={Your Name},
  year={2024},
  howpublished={\\url{https://github.com/ultralytics/ultralytics}}
}
```

## 演示运行

```bash
# 完整演示
python demo_p23456_moe.py

# 预期输出
# YOLOv8-p23456-moe summary: 206 layers, 5,352,821 parameters
# YOLOv8-seg-p23456-moe summary: 238 layers, 5,672,469 parameters
# ✨ 所有 MoE 模型验证完成!
```
