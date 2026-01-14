# YOLO-First Pipeline 实现指南

**分支**: `approach-yolo-first`  
**文件**: `collision_detection_pipeline_yolo_first.py`  
**创建时间**: 2026-01-06  
**状态**: ✅ 完整实现（基础功能）

---

## 📋 实现概览

YOLO-First Pipeline 将碰撞检测分为 5 个独立的步骤，每个步骤都可以独立运行：

```
Step 1: YOLO 检测 (原始视频)
   ↓
   → detections.json (所有物体检测)

Step 2: 轨迹构建 (像素空间)
   ↓
   → tracks.json (轨迹和速度)

Step 3: 关键帧提取 (接近事件)
   ↓
   → proximity_events.json (接近事件)

Step 4: Homography 变换 (世界坐标)
   ↓
   → events_world_coords.json (世界坐标事件)

Step 5: 风险分析 (TTC + 分级)
   ↓
   → collision_events.json (分级事件)
   → analysis_report.txt (报告)
```

---

## 🏗️ 核心类和方法

### `YOLOFirstPipeline` 类

#### 初始化
```python
pipeline = YOLOFirstPipeline(
    video_path="videos/test.mp4",
    homography_path="calibration/H.json",  # 可选
    output_base="results",
)
```

#### 方法

1. **`run_yolo_detection(conf_threshold=0.45)`**
   - 在原始视频上运行 YOLO 检测
   - 所有帧都处理，保存所有检测框
   - 输出: `detections.json`, `detection_stats.json`
   - 特点: **不需要预处理，直接在原始分辨率上检测**

2. **`build_trajectories(all_detections)`**
   - 关联 Track ID，构建轨迹
   - 计算每个点的速度 (px/s)
   - 输出: `tracks.json`, `track_stats.json`
   - 特点: **轨迹在像素空间，速度为 px/s**

3. **`extract_key_frames(all_detections, tracks, pixel_distance_threshold=150)`**
   - 检测距离 < 150px 的物体对
   - 标记为关键帧
   - 输出: `proximity_events.json`
   - 特点: **仅检测关键帧，减少后续处理量**

4. **`transform_to_world_coords(proximity_events, all_detections)`**
   - 仅对关键帧进行 Homography 变换
   - 转换距离: px → 米
   - 转换速度: px/s → m/s
   - 输出: `events_world_coords.json`
   - 特点: **仅变换关键帧，计算量小**

5. **`analyze_collision_risk(proximity_events, transformed_events=None)`**
   - 计算 TTC
   - 分级事件 (L1/L2/L3)
   - 输出: `collision_events.json`
   - 特点: **支持像素空间和世界坐标两种分析**

6. **`generate_report(proximity_events, analyzed_events, level_counts)`**
   - 生成最终报告
   - 输出: `analysis_report.txt`

---

## 📊 输出数据结构

### 1. detections.json
```json
[
  {
    "frame": 1,
    "time": 0.033,
    "objects": [
      {
        "track_id": 42,
        "class": 2,           // 0=person, 2=car, ...
        "conf": 0.95,
        "bbox_xywh": [640, 360, 100, 200]
      }
    ]
  }
]
```

### 2. tracks.json
```json
{
  "42": [
    {
      "frame": 1,
      "time": 0.033,
      "class": 2,
      "conf": 0.95,
      "center_x": 640,
      "center_y": 360,
      "vx": 0.0,       // px/s
      "vy": 0.0,       // px/s
      "speed": 0.0     // px/s
    }
  ]
}
```

### 3. proximity_events.json
```json
[
  {
    "frame": 1000,
    "time": 33.3,
    "object_ids": [42, 15],
    "distance_pixel": 120.5,
    "object_classes": [2, 0]  // car, person
  }
]
```

### 4. events_world_coords.json
```json
[
  {
    "frame": 1000,
    "time": 33.3,
    "object_ids": [42, 15],
    "distance_pixel": 120.5,
    "distance_meters": 0.85,
    "pixel_per_meter": 141.76
  }
]
```

### 5. collision_events.json
```json
[
  {
    "frame": 1000,
    "time": 33.3,
    "object_ids": [42, 15],
    "distance_pixel": 120.5,
    "distance_meters": 0.85,
    "level": 2,
    "level_name": "Near Miss"
  }
]
```

---

## 💻 使用方式

### 方式 1: Python 脚本直接运行
```bash
cd /workspace/ultralytics/examples/trajectory_demo

python collision_detection_pipeline_yolo_first.py \
  --video ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \
  --conf 0.45
```

### 方式 2: 使用 Shell 脚本
```bash
cd /workspace/ultralytics/examples/trajectory_demo

# 运行 YOLO-First pipeline
bash run_yolo_first_pipeline.sh

# 运行两个方案对比
bash compare_both_approaches.sh
```

### 方式 3: Python 代码中导入使用
```python
from collision_detection_pipeline_yolo_first import YOLOFirstPipeline

pipeline = YOLOFirstPipeline(video_path="videos/test.mp4", homography_path="calibration/H.json")
pipeline.run(conf_threshold=0.45)
```

---

## 🔧 参数配置

### 命令行参数
```bash
--video           : 输入视频路径 (必须)
--homography      : Homography JSON 路径 (可选)
--output          : 输出基础目录 (默认: ../../results)
--conf            : YOLO 置信度阈值 (默认: 0.45)
```

### 关键参数
```python
# 在 Step 3 (extract_key_frames) 中
pixel_distance_threshold = 150  # 像素空间的接近距离阈值

# 在 Step 5 (analyze_collision_risk) 中
threshold_collision = 50  # 碰撞阈值（像素空间）
threshold_near_miss = 150  # 近距离阈值（像素空间）

# 或在世界坐标空间中
threshold_collision = 0.5  # 碰撞阈值（米）
threshold_near_miss = 1.5  # 近距离阈值（米）
```

---

## 🔄 与 Homography-First 的对比

| 特性 | Homography-First | YOLO-First |
|------|-----------------|-----------|
| **整体流程** | H → 变换 → YOLO → 分析 | YOLO → 轨迹 → H(关键) → 分析 |
| **预处理** | warped video 生成 | 无 |
| **检测位置** | warped 视频 | 原始视频 |
| **Homography 应用** | 全帧 | 仅关键帧 |
| **坐标空间** | 世界坐标 | 像素 → 世界 |
| **性能** | ~40-60s | ~15-30s |
| **内存** | 高 (warped video) | 低 (缓存) |
| **灵活性** | 低 | 高 |

---

## 📈 预期性能

基于 Homograph_Teset_FullScreen.mp4 (267秒, 30fps, 8000帧):

| 步骤 | 预计时间 | 备注 |
|------|---------|------|
| YOLO 检测 | 8-10秒 | 所有帧 |
| 轨迹构建 | 1-2秒 | 轨迹关联 |
| 关键帧提取 | <1秒 | 只是距离计算 |
| Homography 变换 | 1-2秒 | 仅关键帧 (通常<100帧) |
| 风险分析 | <1秒 | 简单分级 |
| **总计** | **12-16秒** | **比 Homography-First 快 3-4倍** |

---

## 🎯 下一步改进

### 即将实现的部分
- [ ] 完整 TTC 计算（当前是简单分级）
- [ ] PET 计算
- [ ] 动态视频绘制（带标注）
- [ ] 详细的事件统计
- [ ] 轨迹可视化

### 可选优化
- [ ] GPU 推理加速
- [ ] 跳帧策略 (skip_frame parameter)
- [ ] 多线程处理
- [ ] 实时处理流式输入

---

## ⚠️ 已知限制

1. **距离阈值** (150px) 是固定的，不考虑物体大小
   - 解决: 可根据物体尺寸动态调整

2. **速度计算** 仅用相邻两帧
   - 解决: 可用最小二乘法或Kalman滤波

3. **Homography 转换** 是线性近似
   - 解决: 可使用完整的H矩阵进行透视变换

4. **TTC 计算** 当前不完整
   - 解决: 需要从轨迹中提取速度进行完整计算

---

## 📝 代码更改记录

**2026-01-06 初始版本**
- 创建 YOLOFirstPipeline 类
- 实现 5 个核心步骤
- 支持 Homography 可选变换
- 生成详细输出和报告

---

## 🔗 相关文件

- **Pipeline 代码**: `collision_detection_pipeline_yolo_first.py`
- **快速启动**: `run_yolo_first_pipeline.sh`
- **对比测试**: `compare_both_approaches.sh`
- **设计文档**: `YOLO_FIRST_APPROACH.md`
- **分支对比**: `BRANCH_COMPARISON.md`

---

**状态**: ✅ **实现完成，准备测试**

下一步：在同一个视频上运行两个 pipeline，生成对比结果。
