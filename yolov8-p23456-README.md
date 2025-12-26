# YOLOv8-P23456 多尺度检测/分割模型

## 概述

`yolov8-p23456.yaml` 和 `yolov8-seg-p23456.yaml` 是融合了 YOLOv8-p2 和 YOLOv8-p6 的多尺度模型配置，输出 **5 个层级 (P2-P6)**，支持**目标检测**和**实例分割**两种任务，适用于需要同时处理超小目标与超大目标的场景。

## 模型架构

### 输出特征金字塔

- **P2**: 1/4 下采样 (stride=4) - 超小目标
- **P3**: 1/8 下采样 (stride=8) - 小目标
- **P4**: 1/16 下采样 (stride=16) - 中等目标
- **P5**: 1/32 下采样 (stride=32) - 大目标
- **P6**: 1/64 下采样 (stride=64) - 超大目标

### 网络结构

```
Backbone (P1-P6) → SPPF
         ↓
Top-Down FPN: P6 → P5 → P4 → P3 → P2
         ↓
Bottom-Up PAN: P2 → P3 → P4 → P5 → P6
         ↓
Detect Heads: [P2, P3, P4, P5, P6]
```

## 使用方法

### 1. 基本使用

```python
from ultralytics import YOLO

# 加载检测模型
model_det = YOLO("ultralytics/cfg/models/v8/yolov8-p23456.yaml")

# 加载分割模型
model_seg = YOLO("ultralytics/cfg/models/v8/yolov8-seg-p23456.yaml")

# 查看模型信息
model_det.info()
model_seg.info()
```

### 2. 训练

```python
# 目标检测训练
model_det = YOLO("ultralytics/cfg/models/v8/yolov8-p23456.yaml")
model_det.train(
    data="coco.yaml",
    epochs=100,
    imgsz=1280,  # P6 模型建议更大的输入尺寸
    batch=16,
)

# 实例分割训练
model_seg = YOLO("ultralytics/cfg/models/v8/yolov8-seg-p23456.yaml")
model_seg.train(data="coco-seg.yaml", epochs=100, imgsz=1280, batch=16)
```

### 3. 推理

```python
# 检测推理
results_det = model_det.predict("image.jpg", imgsz=1280)

# 分割推理
results_seg = model_seg.predict("image.jpg", imgsz=1280)

# 批量推理
results = model_seg.predict(["img1.jpg", "img2.jpg"], imgsz=1280)

# 可视化分割结果
for r in results_seg:
    r.show()  # 显示结果
    r.save(filename="result.jpg")  # 保存结果
```

### 4. 导出

```python
# 导出为 ONNX 格式
model.export(format="onnx", imgsz=1280)

# 导出为 TensorRT
model.export(format="engine", imgsz=1280, half=True)
```

### 5. 加载预训练权重

```python
# 如果你已经有训练好的权重
model = YOLO("path/to/yolov8-p23456-trained.pt")
results = model.predict("image.jpg")
```

## 模型规模

支持以下规模变体 (通过文件名或 `scale` 参数):

### 检测模型 (yolov8-p23456)

- `yolov8n-p23456`: nano - 5.2M 参数, 17.4 GFLOPs
- `yolov8s-p23456`: small
- `yolov8m-p23456`: medium
- `yolov8l-p23456`: large
- `yolov8x-p23456`: xlarge

### 分割模型 (yolov8-seg-p23456)

- `yolov8n-seg-p23456`: nano - 5.5M 参数, 28.8 GFLOPs
- `yolov8s-seg-p23456`: small
- `yolov8m-seg-p23456`: medium
- `yolov8l-seg-p23456`: large
- `yolov8x-seg-p23456`: xlarge

```python
# 指定规模
model_det_s = YOLO("ultralytics/cfg/models/v8/yolov8-p23456.yaml", scale="s")
model_seg_l = YOLO("ultralytics/cfg/models/v8/yolov8-seg-p23456.yaml", scale="l")
```

## 适用场景

✅ **推荐场景**:

- 需要同时检测/分割超小与超大物体
- 高分辨率图像 (≥1280px)
- 密集小目标检测/分割 (如人群、车辆)
- 遥感图像分析
- 工业缺陷检测 (多尺度瑕疵)
- 医学影像分割 (多尺度器官/病灶)

❌ **不推荐场景**:

- 实时性要求极高 (推理速度比标准模型慢)
- 低分辨率输入 (<640px)
- 显存受限环境

## 性能对比

### 检测模型

| 模型              | 输出层级  | 参数量(n) | GFLOPs(n) | 适用场景        |
| ----------------- | --------- | --------- | --------- | --------------- |
| YOLOv8            | P3-P5     | 3.0M      | 8.1       | 通用检测        |
| YOLOv8-p2         | P2-P5     | 3.1M      | 8.9       | 小目标增强      |
| YOLOv8-p6         | P3-P6     | 5.0M      | 8.8       | 大目标/高分辨率 |
| **YOLOv8-p23456** | **P2-P6** | **5.2M**  | **17.4**  | **全尺度覆盖**  |

### 分割模型

| 模型                  | 输出层级  | 参数量(n) | GFLOPs(n) | 适用场景       |
| --------------------- | --------- | --------- | --------- | -------------- |
| YOLOv8-seg            | P3-P5     | 3.3M      | 12.6      | 通用分割       |
| YOLOv8-seg-p6         | P3-P6     | 5.3M      | 13.5      | 大目标分割     |
| **YOLOv8-seg-p23456** | **P2-P6** | **5.5M**  | **28.8**  | **全尺度分割** |

## 训练建议

1. **图像尺寸**: 建议使用 1280×1280 或更大
2. **Batch Size**: 根据显存调整 (1280px 约需 2×标准显存)
3. **数据增强**: 启用 mosaic/mixup 以适应多尺度
4. **学习率**: 初始 lr=0.01, warmup epochs=3
5. **NMS**: 调整 `conf=0.25, iou=0.7` 平衡精度与召回

```python
model.train(data="custom.yaml", epochs=300, imgsz=1280, batch=8, lr0=0.01, warmup_epochs=3, mosaic=1.0, mixup=0.1)
```

## 验证

运行演示脚本:

```bash
# 检测模型演示
python demo_p23456.py

# 检测+分割完整演示
python demo_p23456_all.py
```

预期输出:

```
YOLOv8-p23456 summary: 201 layers, 5,178,736 parameters, 5,178,720 gradients, 17.4 GFLOPs
YOLOv8-seg-p23456 summary: 233 layers, 5,498,384 parameters, 5,498,368 gradients, 28.8 GFLOPs
✅ 模型成功创建!
📌 输出层级: P2(1/4), P3(1/8), P4(1/16), P5(1/32), P6(1/64)
```

## 常见问题

**Q: 为什么推理速度比标准 YOLOv8 慢?**  
A: P23456 包含 5 个检测头 (vs 标准 3 个),计算量约增加 2 倍。可使用更小规模 (n/s) 或导出 TensorRT 加速。

**Q: 如何在已有权重上微调?**  
A: 暂不支持直接加载标准 YOLOv8 权重,需从头训练或手动迁移 backbone 权重。

**Q: 检测和分割模型可以共享权重吗?**  
A: backbone 部分可以共享,但 head 不同。可以先训练检测模型,然后迁移 backbone 到分割模型。

**Q: 是否支持其他任务 (姿态)?**  
A: 可以参考 yolov8-seg-p23456.yaml 修改最后的 head 为 `Pose`,创建 yolov8-pose-p23456.yaml。

## 引用

基于 Ultralytics YOLOv8 实现:

```
@software{yolov8_ultralytics,
  author = {Glenn Jocher and others},
  title = {Ultralytics YOLOv8},
  year = {2023},
  url = {https://github.com/ultralytics/ultralytics}
}
```
