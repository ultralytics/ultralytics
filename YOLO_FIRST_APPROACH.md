# YOLO-First Pipeline (Approach 2)

**分支**: `approach-yolo-first`  
**创建日期**: 2026-01-06  
**对比分支**: `approach-homography-first`

---

## 📋 设计思路

与 `approach-homography-first` 不同的执行顺序：

```
YOLO-First Approach:
1. 直接在原始视频上运行 YOLO 检测
2. 识别所有目标并建立轨迹
3. 计算轨迹中的关键帧（接近/碰撞事件）
4. 仅对关键帧进行 Homography 变换和坐标转换
5. 计算 TTC/PET 等参数（仅关键帧）

优势:
- 避免对整个视频进行透视变换（节省计算）
- Homography 仅用于关键帧（精度关键部分）
- 可以从不同帧率/分辨率的原始视频直接检测
- 更灵活的流程设计

劣势:
- 需要在像素空间进行轨迹关联
- 距离阈值需要动态调整（因为缺少世界坐标）
```

---

## 🏗️ 实现框架

### Phase 1: YOLO Detection on Original Video

```
pipeline_yolo_first.py
├── YOLODetector
│   └── detect(video_path) → raw_detections
├── TrajectoryBuilder
│   └── build_tracks(detections) → tracks
└── KeyFrameExtractor
    └── extract_proximity_frames(tracks) → key_frames
```

### Phase 2: Homography Transform on Key Frames Only

```
KeyFrameProcessor
├── load_homography(H_path)
├── transform_key_frames(frames, H) → world_coords
└── save_transformed_frames()
```

### Phase 3: Risk Analysis on Processed Frames

```
RiskAnalyzer
├── estimate_velocity_from_tracks(track)
├── calculate_ttc(obj1, obj2) → ttc
├── classify_event(distance, ttc) → level
└── generate_report()
```

---

## 📁 需要创建的新文件

1. **pipeline_yolo_first.py** (主 pipeline)
   - YOLODetector: 在原始视频上检测
   - TrajectoryBuilder: 构建轨迹（像素空间）
   - KeyFrameExtractor: 提取接近事件帧

2. **proximity_detector_pixel_space.py**
   - 在像素空间中检测接近事件
   - 使用动态阈值（距离 < 150px 或类似）

3. **risk_analyzer_yolo_first.py**
   - 针对 YOLO-first 流程的风险分析
   - TTC/PET 计算
   - Event 分级

---

## 🎯 关键差异点

### Distance Thresholding

**Homography-First**:
```python
distance < 1.5m  # 世界坐标，米制
```

**YOLO-First** (关键帧前):
```python
distance < 150px  # 像素空间，动态调整
# 考虑物体大小和深度信息
```

### TTC Calculation

**相同**: 都使用速度估计和相对速度

**不同**: 
- Homography-First: 速度单位是 m/s
- YOLO-First: 速度单位是 px/s，后续需转换

---

## 📊 预期输出结构

```
results/20260106_XXXXXX_yolo_first/
├── 1_raw_detections/
│   ├── detections.json (所有帧的检测结果)
│   └── detection_stats.json
├── 2_trajectories/
│   ├── tracks.json (完整轨迹，像素空间)
│   └── track_stats.json
├── 3_key_frames/
│   ├── proximity_events.json (接近事件列表)
│   ├── key_frames/ (接近事件的原始帧)
│   └── event_analysis.json
├── 4_homography_transform/
│   ├── key_frames_world.json (转换后的坐标)
│   ├── homography_matrix.json
│   └── transformed_frames/
├── 5_risk_analysis/
│   ├── collision_events.json (完整的事件信息，带 TTC)
│   ├── events_by_level.json (按 Level 分类)
│   └── analysis_report.txt
└── comparison_with_homography_first.md
```

---

## 🔧 实现步骤

### Step 1: Raw YOLO Detection (第1-2天)
- [ ] 创建 YOLODetector (检测所有帧)
- [ ] 保存原始检测结果
- [ ] 生成检测统计

### Step 2: Trajectory Building (第2天)
- [ ] 创建 TrajectoryBuilder (ID 关联和轨迹管理)
- [ ] 在像素空间中计算轨迹
- [ ] 估计速度 (px/s)

### Step 3: Key Frame Extraction (第2-3天)
- [ ] 创建 KeyFrameExtractor
- [ ] 实现像素空间的接近事件检测 (distance < 150px)
- [ ] 保存关键帧

### Step 4: Homography Transform on Key Frames (第3天)
- [ ] 加载 Homography 矩阵
- [ ] 变换关键帧坐标
- [ ] 转换速度单位 (px/s → m/s)

### Step 5: Risk Analysis & Comparison (第3-4天)
- [ ] 计算 TTC/PET
- [ ] Event 分级
- [ ] 生成报告
- [ ] 与 Homography-First 方案对比

---

## 🧪 测试对比

创建一个对比脚本来评估两个方案:

```python
# compare_approaches.py


def compare_performance():
    """对比两个 pipeline 的性能."""
    metrics = {
        "detection_time": {},
        "trajectory_time": {},
        "homography_time": {},
        "total_time": {},
        "memory_usage": {},
        "accuracy": {},
    }

    # 运行两个 pipeline
    # 记录时间和内存
    # 对比结果一致性


def compare_outputs():
    """对比两个 pipeline 的输出."""
    # 检测结果是否一致
    # TTC 值是否接近
    # Event 分级是否相同
```

---

## 📝 关键实现细节

### 像素空间距离计算

```python
def pixel_distance(obj1_bbox, obj2_bbox):
    """计算两个物体的最小距离."""
    # 使用接触点而非中心点
    contact_points_1 = get_contact_points(obj1_bbox)
    contact_points_2 = get_contact_points(obj2_bbox)

    min_distance = float("inf")
    for p1 in contact_points_1:
        for p2 in contact_points_2:
            dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
            min_distance = min(min_distance, dist)

    return min_distance
```

### 动态接近阈值

```python
def is_proximity_event(dist_pixels, obj1_size, obj2_size):
    """判断是否为接近事件，考虑物体大小."""
    # 基础阈值: 150px
    # 调整因子: 根据物体大小
    base_threshold = 150

    # 较大的物体可能需要更大的阈值
    size_factor = (obj1_size + obj2_size) / 2 / 100
    threshold = base_threshold * size_factor

    return dist_pixels < threshold
```

---

## 🔄 切换分支命令

```bash
# 查看所有分支
git branch -a

# 切换到 Homography-First 分支
git checkout approach-homography-first

# 切换到 YOLO-First 分支
git checkout approach-yolo-first

# 比较两个分支的差异
git diff approach-homography-first approach-yolo-first
```

---

## 📊 对比矩阵

| 特征 | Homography-First | YOLO-First |
|------|-----------------|-----------|
| 全帧处理 | 是 (所有帧透视变换) | 否 (仅关键帧) |
| 坐标空间 | 世界坐标 (米) | 像素空间 → 世界坐标 |
| 计算量 | 高（视频处理） | 低（仅关键帧） |
| 距离阈值 | 固定 (1.5m) | 动态 (基于像素) |
| Homography用途 | 视频变换 + 坐标转换 | 仅坐标转换 |
| 速度单位 | m/s (直接) | px/s (需转换) |
| 输出灵活性 | 固定格式 | 高度灵活 |
| 适用场景 | 需要全视图分析 | 仅关注关键事件 |

---

## 📌 下一步

1. **实现 Phase 1-3** (YOLO Detection + Trajectory + Key Frame Extraction)
2. **运行初步测试** 确保关键帧提取正常
3. **等待导师反馈** 关于两个方案的选择
4. **基于反馈调整** 或继续完善当前方案

---

**状态**: 准备就绪，等待实现  
**预计时间**: 4-5 天完成 YOLO-First 完整 pipeline  
**对比期限**: 等导师选择方向后进行最终优化
