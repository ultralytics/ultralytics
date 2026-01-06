# 实现路线图 & 代码清单

**目标**: 在导师要求的时间框架内 (2月中旬) 完成关键功能  
**当前日期**: 2026-01-06  
**可用时间**: ~6周  
**关键提交期限**: 2026-01-25 (PPT 审核)

---

## Phase 1: TTC + Event分级 (第1-2周，最关键)

### 1.1 完整 TTC 计算实现

**文件**: `examples/trajectory_demo/ttc_calculator.py` (新建)

```python
# 伪代码框架
class TTCCalculator:
    def estimate_velocity(track_data):
        """
        从轨迹数据估计速度
        输入: track_data = [(x, y, t), (x, y, t), ...]
        输出: vx, vy (像素/秒 或 米/秒)
        """
        # 使用最近两帧或最小二乘法
        
    def calculate_ttc(obj1_pos, obj1_vel, obj2_pos, obj2_vel, distance):
        """
        TTC = distance / |relative_velocity_along_collision_axis|
        """
        # 计算相对速度
        # 计算沿碰撞轴的分量
        # 返回 TTC (秒)
        
    def calculate_pet(obj1_trajectory, obj2_trajectory):
        """
        Post Encroachment Time
        测量一个物体离开碰撞点的时间到另一个物体到达的时间差
        """
        # 找到碰撞点
        # 计算时间差
```

**输出数据结构**:
```json
{
  "frame": 8000,
  "time": 267.3,
  "object_ids": [42, 15],
  "distance": 0.8,
  "velocities": {
    "obj_42": {"vx": -0.15, "vy": 0.05},
    "obj_15": {"vx": 0.2, "vy": 0}
  },
  "ttc": 2.3,
  "pet": null,
  "risk_level": 2
}
```

### 1.2 Event 分级逻辑实现

**文件**: `examples/trajectory_demo/event_classifier.py` (新建)

```python
class EventClassifier:
    def classify(distance, ttc, pet=None):
        """
        Level 1 (Collision): distance < 0.5m 或 TTC < 1.0s
        Level 2 (Near Miss):  0.5m ≤ distance < 1.5m 且 TTC < 3.0s
        Level 3 (Avoidance):  distance ≥ 1.5m 但有交集迹象
        """
        if distance < 0.5 or (ttc and ttc < 1.0):
            return 1, "Collision"
        elif distance < 1.5 and (not ttc or ttc < 3.0):
            return 2, "Near Miss"
        else:
            return 3, "Avoidance"
```

### 1.3 修改 collision_detection_pipeline.py

**更新内容**:
- 在检测循环中添加 TTC 计算
- 添加事件分级逻辑
- 修改输出 JSON 格式
- 修改报告生成逻辑

**修改点**:
```python
# 原来的 collision_events 数据结构
# {frame, time, object_ids, distance, distance_str, frame_image}

# 改为
# {frame, time, object_ids, distance, distance_str, 
#  velocity_1, velocity_2, ttc, pet, risk_level, level_name, frame_image}
```

---

## Phase 2: 动态视频绘制 (第2-3周，高优先级)

### 2.1 实现视频标注模块

**文件**: `examples/trajectory_demo/video_annotator.py` (新建)

```python
class VideoAnnotator:
    def __init__(self, video_path, output_path, homography_path):
        """初始化视频标注器"""
        
    def draw_detection_frame(frame, detections, event_info=None):
        """
        在单帧上绘制:
        - 检测边框 (绿色边界框)
        - Track ID (例如 "ID:42")
        - 距离标注 (例如 "Dist: 0.8m")
        - TTC 标注 (例如 "TTC: 2.3s")
        - Level 标记 (L1=红, L2=黄, L3=绿)
        - 速度向量 (箭头)
        """
        
    def process_video(collision_events, fps, total_frames):
        """
        遍历整个视频，对每一帧:
        1. 读取帧
        2. 查找该帧的检测/事件信息
        3. 绘制标注
        4. 写入输出视频
        输出: annotated_video.mp4
        """
        
    def create_level_color(level):
        """Level 颜色编码: 1=红(255,0,0), 2=黄(0,255,255), 3=绿(0,255,0)"""
```

**关键细节**:
```python
# 边框绘制
cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=2)

# Track ID
cv2.putText(frame, f"ID:{track_id}", (x1, y1-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# 距离和 TTC
cv2.putText(frame, f"Dist:{distance:.1f}m TTC:{ttc:.1f}s", 
            (x1, y2+30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# Level 标记
level_text = f"L{level}"
cv2.putText(frame, level_text, (x2-50, y1-10), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
```

### 2.2 集成到 Pipeline

**修改 collision_detection_pipeline.py**:
```python
def generate_annotated_video(self, collision_events):
    """新增方法: 生成标注视频"""
    annotator = VideoAnnotator(self.warped_video_path, 
                                self.collision_dir / 'annotated_video.mp4',
                                self.homography_path)
    annotator.process_video(collision_events, fps, total_frames)
```

---

## Phase 3: 报告增强 (第2周)

### 3.1 修改报告生成

**文件**: `examples/trajectory_demo/collision_detection_pipeline.py` 的 `generate_report()` 方法

**当前报告**:
```
检测统计:
  - 检测到物体的帧数: 57
  - 碰撞事件数: 0
```

**改进为**:
```
碰撞风险分析报告
================================================

【基本信息】
生成时间: 2026-01-06 12:34:56
输入视频: Homograph_Teset_FullScreen.mp4
分析时长: 267.3 秒 (8000帧)
帧率: 30 fps

【事件统计】
总检测帧数: 57
- Level 1 (Collision):  0 events
- Level 2 (Near Miss):  3 events  
- Level 3 (Avoidance):  8 events

【高风险事件详情】

事件 #1 - LEVEL 2 (Near Miss)
  时间戳: 00:05:23 (Frame 8000)
  物体对: Vehicle_42 ↔ Pedestrian_15
  最小距离: 0.8m
  相对速度: 0.35 m/s
  TTC: 2.3s
  PET: N/A
  截图: event_level2_001.jpg

事件 #2 - LEVEL 2 (Near Miss)
  时间戳: 00:07:15 (Frame 8700)
  物体对: Vehicle_28 ↔ Vehicle_42
  最小距离: 1.2m
  相对速度: 0.28 m/s
  TTC: 4.3s
  PET: N/A
  截图: event_level2_002.jpg

【可视化输出】
- 标注视频: collision_events_annotated.mp4
  包含: 边框、ID、距离、TTC、Level颜色标记
- 事件截图: 按 Level 分类保存
  event_level1_*.jpg
  event_level2_*.jpg
  event_level3_*.jpg

【技术信息】
Homography 校准点: 4
Homography 误差: < 0.1%
世界坐标范围: X=[-3.75, 3.75]m, Y=[0, 50]m
距离阈值: 1.5m
```

---

## Phase 4: 性能优化 (第3周，可选)

### 4.1 跳帧策略

**文件**: `examples/trajectory_demo/collision_detection_pipeline.py`

```python
def detect_collisions(self, conf_threshold=0.45, skip_frame=2):
    """
    skip_frame=2: 处理每2帧，检测速度提升2倍
    skip_frame=0: 处理所有帧（精度最高）
    """
    frame_count = 0
    for result in model.track(...):
        if skip_frame > 0 and frame_count % skip_frame != 0:
            frame_count += 1
            continue
        # 处理这一帧
```

### 4.2 GPU 推理 (如有GPU)

```python
# 在 Pipeline 初始化时
model = YOLO('yolo11n.pt')
model.to('cuda')  # 如果有 GPU

# 或在 track 时指定
results = model.track(source=video, device=0)  # device=0 表示 GPU
```

---

## Phase 5: 检测精度改进 (可选，若时间允许)

### 5.1 调整参数

```python
# 尝试更大的输入分辨率
model = YOLO('yolo11n.pt')
results = model.predict(source=frame, imgsz=640)  # 而不是 384

# 或使用更大的模型
model = YOLO('yolo11s.pt')  # small 而不是 nano
```

### 5.2 考虑 Segmentation

```python
# 如果需要更好的精度，可考虑
model = YOLO('yolo11n-seg.pt')  # Segmentation 模型
```

---

## 📁 最终输出目录结构

```
results/20260106_XXXXXX/
├── 1_homography/
│   ├── homography.json
│   ├── verify_original.jpg
│   └── verify_grid_warp.jpg
├── 2_warped_video/
│   ├── warped_video.mp4
│   └── warped_video_stats.json
├── 3_yolo_detection/
│   ├── detection_results.json
│   └── detection_stats.json
├── 4_tracking/
│   ├── trajectories.json
│   └── track_stats.json
├── 5_collision_analysis/
│   ├── collision_events.json        [新增: 完整数据]
│   ├── collision_events_annotated.mp4  [新增: 视频]
│   ├── event_frames/
│   │   ├── level1_collision_001.jpg
│   │   ├── level2_nearmiss_001.jpg
│   │   └── level3_avoidance_001.jpg
│   └── analysis_report.txt          [升级: 详细版]
└── analysis_summary.json            [新增: 元数据]
```

---

## 🎯 具体代码修改清单

### 新建文件 (3个)
1. **ttc_calculator.py** - TTC 和速度计算
2. **event_classifier.py** - Event 分级
3. **video_annotator.py** - 视频标注

### 修改文件 (1个)
1. **collision_detection_pipeline.py**
   - 导入新模块
   - 在 detect_collisions() 中添加 TTC 计算
   - 在 detect_collisions() 中添加事件分级
   - 新增 generate_annotated_video() 方法
   - 修改 generate_report() 方法
   - 修改输出 JSON 格式

### 修改行数估计
- ttc_calculator.py: ~150 行
- event_classifier.py: ~50 行
- video_annotator.py: ~250 行
- collision_detection_pipeline.py: +200 行修改

**总计**: 新增/修改 ~650 行代码

---

## ⏱️ 时间估计

| 任务 | 估计时间 | 完成期限 |
|------|---------|---------|
| TTC 计算实现 | 1.5 天 | 2026-01-08 |
| Event 分级实现 | 0.5 天 | 2026-01-08 |
| Pipeline 集成 | 1 天 | 2026-01-09 |
| 视频标注实现 | 2 天 | 2026-01-11 |
| 报告增强 | 0.5 天 | 2026-01-12 |
| 测试和调试 | 1.5 天 | 2026-01-14 |
| **总计** | **~7 天** | **2026-01-14** |

剩余时间: 11 天用于性能优化、精度改进、视频验证、PPT 准备

---

## 🚀 快速启动

如果你现在想立即开始，建议的顺序:

1. **今天 (1月6日)**: 
   - 创建 ttc_calculator.py 和 event_classifier.py 的框架
   - 编写 TTC 计算逻辑

2. **明天 (1月7日)**: 
   - 完成 TTC + 分级集成到 pipeline
   - 运行测试，验证输出数据格式

3. **后天 (1月8日)**: 
   - 开始视频标注实现

4. **1月9日**: 
   - 完成报告升级
   - 生成演示输出

这样你可以在 1 月 14 日前完成所有关键功能，为 PPT 审核和演示准备留出充足时间。
