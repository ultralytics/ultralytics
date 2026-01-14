# 学习指南：理解 trajectory_demo 前的准备

这个指南帮你理解：

1. 现有的类似实现（从 examples 里学习）
2. trajectory_demo 在哪些方面是"新的"或"更进阶"的
3. 如何一步步开始学习和修改

## 第一步：学习现有实现

### 1.1 object_tracking.ipynb —— 最基础的追踪示例

**位置**：`examples/object_tracking.ipynb`

**这个示例展示了什么**：

- 如何加载 YOLO 模型（带 segmentation）
- 如何逐帧读取视频并调用 `model.track()`
- 如何从 results 中提取 `track_id`（关键！）
- 如何把追踪结果可视化并保存成视频

**关键代码**：

```python
from collections import defaultdict

import cv2

from ultralytics import YOLO

track_history = defaultdict(lambda: [])  # 按 ID 存储历史
model = YOLO("yolo11n-seg.pt")
cap = cv2.VideoCapture("path/to/video.mp4")

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    # 核心：调用 model.track() 获得追踪结果
    results = model.track(im0, persist=True)

    if results[0].boxes.is_track:
        track_ids = results[0].boxes.id.int().cpu().tolist()
        # ... 处理 track_ids
```

**学会什么**：

- `model.track()` 返回的 `results` 对象结构
- 如何提取 `boxes.id`（track ID）、`boxes.xy`（坐标）等信息
- 用 `defaultdict(list)` 管理简单的历史数据

---

### 1.2 YOLOv8-Region-Counter —— 更复杂的数据管理

**位置**：`examples/YOLOv8-Region-Counter/yolov8_region_counter.py`

**这个示例展示了什么**：

- 如何用 `model.track()` 逐帧处理
- 如何计算 bbox 的中心点（anchor point）
- 如何管理每个 ID 的轨迹历史（用 `defaultdict`）
- 如何基于轨迹做简单计算（例如判断目标是否进入某个区域）

**关键代码**：

```python
track_history = defaultdict(list)  # 全局变量存轨迹

for box, track_id, cls in zip(boxes, track_ids, clss):
    bbox_center = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2  # 计算中心点

    track = track_history[track_id]  # 获取该 ID 的历史
    track.append((float(bbox_center[0]), float(bbox_center[1])))  # 添加新点
    if len(track) > 30:
        track.pop(0)  # 限制历史长度

    # 用历史轨迹画线
    points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [points], isClosed=False, color=colors(cls, True), thickness=2)
```

**学会什么**：

- 如何计算 bbox 的中心点（这就是我们说的 "anchor point"）
- 如何用 `defaultdict(list)` + ID 管理每个目标的历史
- 如何基于历史轨迹做后续计算

---

## 第二步：理解 trajectory_demo 在这基础上做了什么

现有的两个例子都是"轻量级"的数据管理——只用一个全局字典 `track_history` 存储。

**trajectory_demo 的改进**：

1. **模块化**：把不同的功能分到不同文件（adapter、coord_transform、osm 等）
2. **完整数据结构**：不仅保存位置，还保存 class、confidence、bbox、时间戳
3. **专用类**：用 `ObjectStateManager` 类来管理，而不是全局变量
4. **高级功能**：提供 `distance_between()`、`approximate_velocity()`、`ttc_between()` 等工具函数
5. **坐标转换**：内置像素→世界坐标的变换（虽然现在是占位符）
6. **轨迹预测**：提供 CV/CA 模型的未来位置预测

---

## 第三步：学习路线（推荐顺序）

### 方案 A：先从现有例子改造（更安全）

1. 打开 `examples/object_tracking.ipynb`，运行一遍，理解基础流程
2. 打开 `examples/YOLOv8-Region-Counter/yolov8_region_counter.py`，理解数据管理方式
3. 把 trajectory_demo 的各个模块当作"你想要实现的增强版"来看待
4. 逐个修改 trajectory_demo 以满足你的需求

### 方案 B：学习 trajectory_demo 的设计逻辑（更快）

1. 先看 `ObjectStateManager` 的 `update()` 和 `get_trajectory()` 方法，对比 `track_history` 的用法
2. 理解 `detection_adapter` 把原始 results 转成什么格式
3. 理解 `coord_transform` 怎么做坐标转换
4. 运行 `yolo_runner.py`，看输出的 `tracks.json` 文件格式

---

## 第四步：最简单的开始方式

**不用我之前写的复杂代码，先试试最简单的修改**：

### 步骤 1：复制并简化 object_tracking.ipynb 的思路

新建文件 `examples/trajectory_demo/simple_tracker.py`：

```python
import json
from collections import defaultdict

import cv2

from ultralytics import YOLO

# 简单版本：只做最基础的追踪
track_history = defaultdict(list)
all_detections = {}  # 按帧保存所有检测

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture("your_video.mp4")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True)

    if results[0].boxes.is_track:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()

        frame_detections = []
        for box, tid, cls_id in zip(boxes, track_ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            # 保存到历史
            track_history[tid].append({"x": cx, "y": cy, "t": frame_idx})

            frame_detections.append(
                {
                    "id": tid,
                    "cx": float(cx),
                    "cy": float(cy),
                    "cls": int(cls_id),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        all_detections[frame_idx] = frame_detections

    frame_idx += 1

# 保存结果
with open("tracks_simple.json", "w") as f:
    json.dump(dict(track_history), f)

with open("detections_simple.json", "w") as f:
    json.dump(all_detections, f)

cap.release()
print("Done!")
```

**这个版本**：

- 只有 ~50 行代码
- 不用任何新的模块，直接基于 YOLO API
- 输出两个 JSON：轨迹历史 + 检测结果
- 你可以在此基础上逐步添加坐标转换、预测等功能

---

## 第五步：理解 trajectory_demo 各文件的"为什么"

| 文件                       | 对标现有代码中的哪部分                                | 为什么需要？                                                    |
| -------------------------- | ----------------------------------------------------- | --------------------------------------------------------------- |
| `detection_adapter.py`     | object_tracking 中的 `results[0].boxes` 解析          | 把解析逻辑独立出来，方便复用和测试                              |
| `object_state_manager.py`  | YOLOv8-Region-Counter 中的 `track_history` + 计数逻辑 | 把简单字典升级为专门的类，提供更多方法（distance、velocity 等） |
| `coord_transform.py`       | YOLOv8-Region-Counter 中的"判断是否在区域内"的逻辑    | 把像素坐标转成世界坐标，这是后续近似碰撞检测的基础              |
| `trajectory_prediction.py` | 不存在于现有代码                                      | 这是你的"新增功能"——基于历史预测未来                            |
| `yolo_runner.py`           | object_tracking.ipynb 的主循环                        | 把所有模块整合到一个脚本，统一数据流                            |

---

## 第六步：建议的学习动作

选择一个：

### 选项 1（学习优先）

1. 打开 `examples/object_tracking.ipynb`，逐行运行
2. 打开 `examples/YOLOv8-Region-Counter/yolov8_region_counter.py`，阅读并理解
3. 然后看我给你的 `trajectory_demo` 文件，对比理解

### 选项 2（快速上手）

1. 复制上面的 `simple_tracker.py`，在你的视频上运行一遍
2. 看看输出的 JSON 文件格式
3. 然后逐步替换为 `trajectory_demo` 的各个模块

### 选项 3（深度理解）

我现在可以为你：

- 用一个完全的示例（包含样本数据）运行 `object_tracking.ipynb`，生成可视化
- 用实际视频运行 `simple_tracker.py`，展示输出
- 逐个讲解 `trajectory_demo` 中每个函数的作用

---

## 总结

现有的代码 + trajectory_demo 的关系：

```
object_tracking.ipynb（最基础）
    ↓
YOLOv8-Region-Counter（加入数据管理）
    ↓
trajectory_demo（模块化 + 增强功能）
    ↓
你的最终项目（添加坐标转换 + 近似碰撞检测）
```

**你现在的位置**：我直接跳到了最后一层，可能跨度太大。建议：

1. 先从 object_tracking 或 simple_tracker.py 开始
2. 逐步理解每一层的改进
3. 最后再使用完整的 trajectory_demo

需要我帮你做哪个？
