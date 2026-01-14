# 代码对比：现有实现 vs trajectory_demo

这个表格帮你理解 trajectory_demo 中的每个部分来自哪里，为什么需要，以及它如何改进了现有代码。

---

## 核心概念对比

| 概念                  | object_tracking.ipynb          | YOLOv8-Region-Counter            | simple_tracker.py                | yolo_runner.py + 其他                      |
| --------------------- | ------------------------------ | -------------------------------- | -------------------------------- | ------------------------------------------ |
| **怎么读视频**        | cv2.VideoCapture               | cv2.VideoCapture                 | cv2.VideoCapture                 | cv2.VideoCapture                           |
| **怎么调用 YOLO**     | model.track(im0, persist=True) | model.track(frame, persist=True) | model.track(frame, persist=True) | model.track(source, stream=True)           |
| **怎么提取 track_id** | results[0].boxes.id            | results[0].boxes.id              | results[0].boxes.id              | detection_adapter.parse_result()           |
| **怎么保存轨迹**      | 直接用 track_history dict      | track_history defaultdict        | track_history defaultdict        | ObjectStateManager 类                      |
| **轨迹数据结构**      | list of (x, y) tuples          | list of (x, y) tuples            | list of dicts {x,y,t,cls,...}    | list of dicts {x,y,t,cls,...}              |
| **提供的方法**        | 无（自己用数据）               | 无（自己用数据）                 | 基本的 JSON 保存                 | distance_between, ttc_between, velocity 等 |
| **代码行数**          | ~50 行                         | ~250 行                          | ~200 行                          | ~500+ 行（包含所有模块）                   |
| **模块化程度**        | 无（一个 notebook）            | 低（一个文件）                   | 低（一个文件）                   | 高（多个文件）                             |
| **易于扩展**          | 困难（改 notebook）            | 中等（改一个文件）               | 容易（改一个文件）               | 容易（加新模块）                           |

---

## 具体代码对比

### 读视频并调用 YOLO

**object_tracking.ipynb**：

```python
cap = cv2.VideoCapture("path/to/video.mp4")
while True:
    ret, im0 = cap.read()
    if not ret:
        break
    results = model.track(im0, persist=True)
    # 处理 results...
```

**simple_tracker.py**：

```python
cap = cv2.VideoCapture(source)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model.track(frame, persist=True)
    # 同上
```

**yolo_runner.py**：

```python
for result in model.track(source=source, stream=True, persist=True):
    # 直接迭代，不用自己管理 VideoCapture
```

⭐ **差别**：yolo_runner 用 `stream=True`，让 YOLO 自己管理视频读取，更简洁。

---

### 提取 track_id 和坐标

**object_tracking.ipynb**：

```python
if results[0].boxes.is_track:
    track_ids = results[0].boxes.id.int().cpu().tolist()
    # ... boxes 还要自己处理
```

**simple_tracker.py**：

```python
if results[0].boxes.is_track:
    boxes = results[0].boxes.xyxy.cpu().numpy()
    track_ids = results[0].boxes.id.int().cpu().tolist()
    clss = results[0].boxes.cls.cpu().tolist()
    confs = results[0].boxes.conf.cpu().tolist()
```

**yolo_runner.py（用 detection_adapter）**：

```python
dets = detection_adapter.parse_result(result, timestamp)
# dets 是一个 list of dicts：
# [{'id': 42, 'cx': 640, 'cy': 360, 't': 0, 'cls': 0, 'conf': 0.98}, ...]
```

⭐ **差别**：detection_adapter 把所有解析逻辑封装起来，返回统一格式，不用重复写转换代码。

---

### 保存轨迹历史

**object_tracking.ipynb / YOLOv8-Region-Counter**：

```python
track_history = defaultdict(list)  # 全局变量

for box, track_id, cls in zip(boxes, track_ids, clss):
    bbox_center = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
    track = track_history[track_id]
    track.append(bbox_center)  # 只保存 (x, y)
```

**simple_tracker.py**：

```python
track_history = defaultdict(list)  # 同上

for box, tid, cls_id, conf in zip(boxes, track_ids, clss, confs):
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    sample = {"x": cx, "y": cy, "t": frame_idx, "cls": cls_id, "conf": conf, "bbox": [x1, y1, x2, y2]}
    track_history[tid].append(sample)  # 保存更多信息
```

**yolo_runner.py（用 ObjectStateManager）**：

```python
osm = ObjectStateManager()
osm.update(dets, timestamp)  # 内部管理轨迹
# 可以调用：
osm.get_trajectory(id)  # 获取轨迹
osm.distance_between(id1, id2)  # 计算距离
osm.approximate_velocity(id)  # 计算速度
osm.ttc_between(id1, id2)  # 计算碰撞时间
```

⭐ **差别**：

- simple_tracker 保存更多字段（cls, conf, bbox）
- yolo_runner 把管理逻辑放进类，提供高级查询接口

---

### 坐标转换（新增）

**现有代码**：完全没有

**trajectory_demo**：

```python
# coord_transform.py
from coord_transform import pixel_to_world

X, Y = pixel_to_world(cx, cy)  # 把像素坐标转成世界坐标
```

⭐ **差别**：这是你的**新功能**。后续可以替换为相机标定矩阵。

---

### 轨迹预测（新增）

**现有代码**：完全没有

**trajectory_demo**：

```python
# trajectory_prediction.py
from trajectory_prediction import predict_future

future = predict_future(traj, horizon_sec=1.0, dt=0.1, model="cv")
# 返回未来位置列表
```

⭐ **差别**：这是你的**新功能**。简单的 CV/CA 模型。

---

## 学习建议

### 如果你已经熟悉 object_tracking.ipynb

→ 直接看 `simple_tracker.py`，理解：

1. 如何添加更多字段（cls, conf, bbox）
2. 如何创建 JSON 输出
3. 代码如何组织（从 notebook 到 .py 脚本）

### 如果你已经看过 YOLOv8-Region-Counter

→ 理解：

1. `simple_tracker.py` 和它的相似之处（都用 track_history）
2. `ObjectStateManager` 如何改进了 track_history 的用法
3. 为什么把逻辑分成多个模块（detection_adapter, osm 等）

### 如果你是新手

→ **按这个顺序学**：

1. 先看 `examples/object_tracking.ipynb`（最基础）
2. 再看 `simple_tracker.py`（添加了什么新东西？）
3. 再看 `yolo_runner.py + ObjectStateManager`（为什么这样设计？）
4. 最后看 `coord_transform` 和 `trajectory_prediction`（如何扩展？）

---

## 快速对标表

| 你需要的功能         | 现有哪个文件有参考    | trajectory_demo 的改进                    |
| -------------------- | --------------------- | ----------------------------------------- |
| 用 YOLO 读视频并追踪 | object_tracking.ipynb | simple_tracker.py（同，只是更清晰）       |
| 保存轨迹为 JSON      | 无                    | simple_tracker.py（新功能）               |
| 管理多个 ID 的轨迹   | YOLOv8-Region-Counter | ObjectStateManager（升级版）              |
| 计算两目标距离       | 无                    | ObjectStateManager.distance_between()     |
| 计算目标速度         | 无                    | ObjectStateManager.approximate_velocity() |
| 计算碰撞时间         | 无                    | ObjectStateManager.ttc_between()          |
| 坐标转换             | 无                    | coord_transform.py（新模块）              |
| 轨迹预测             | 无                    | trajectory_prediction.py（新模块）        |

---

## 关键洞察

1. **simple_tracker.py 不是革命性的**
   - 它基本上是把 object_tracking.ipynb 的逻辑改成 .py 脚本
   - 加了 JSON 输出和更多字段
   - 但核心思想完全一样（track_history dict）

2. **yolo_runner.py + ObjectStateManager 才是新的**
   - 把 defaultdict 变成专门的类
   - 提供 distance、velocity、ttc 等方法
   - 支持坐标转换、轨迹预测等扩展

3. **coord_transform 和 trajectory_prediction 是你的需求**
   - 这两个完全是新功能
   - 不存在于现有代码中
   - 这才是你项目的核心

---

## 总结

```
现有代码 + 我的改进 = 完整的解决方案

    object_tracking.ipynb
          ↓
    simple_tracker.py （更清晰的版本 + JSON 输出）
          ↓
    yolo_runner.py + ObjectStateManager （模块化 + 高级功能）
          ↓
    + coord_transform.py （坐标转换，你的需求）
    + trajectory_prediction.py （轨迹预测，你的需求）
          ↓
    完整的 A/B/C 模块实现
```

你现在有了全部的积木块。现在的任务是**理解**这些积木块，然后**组装**成你的项目。
