# 项目与导师要求的对标分析

**日期**: 2026-01-06  
**分析目的**: 检查当前项目实现是否符合导师要求，识别功能缺口

---

## 📋 导师提出的技术架构（3大模块）

### 模块1: 目标检测与追踪（Segmentation/Detection）
**导师要求**: 
- 使用segmentation获取路边轨迹
- 检测车辆、行人等对象

**当前实现**: ✓ 部分满足
- ✅ 使用 YOLO11-nano 进行目标检测
- ✅ 追踪 track_id 并存储轨迹
- ❌ **未实现**: Segmentation（只做了detection）
- ❌ **问题**: 检测精度问题 - 图像缩放导致车辆被误检为行人

**建议改进**:
```
1. 考虑添加 segmentation 模型（YOLOv8/v11 支持 segmentation）
   或至少用更大的输入尺寸（current: 384×640）
2. 或者用 YOLOv8-seg / YOLOv11-seg
3. 调整输入分辨率，避免信息丢失导致误分类
```

---

### 模块2: 坐标系变换（Image → World Coordinates）
**导师要求**: 
- 将图像坐标转换为世界坐标系
- 支持米制单位

**当前实现**: ✅ 已完成
- ✅ Homography 矩阵标定
- ✅ 像素坐标 → 世界坐标变换
- ✅ 输出米制距离
- ❌ **性能问题**: 手动像素计算导致处理速度慢

**建议改进**:
```
1. 使用向量化 NumPy 操作替代逐像素计算
2. 预计算变换矩阵以加快速度
3. 考虑 GPU 加速（OpenCV CUDA 支持）
```

---

### 模块3: 碰撞风险计算（距离 + 时间）
**导师要求**: 
- 距离检测 ✓
- 时间计算 (TTC/PET)
- 多级别结果分类 (level 分级)

**当前实现**: ⚠️ 部分满足
- ✅ 距离检测（1.5m 阈值）
- ❌ **缺失**: 完整 TTC 计算（只有公式，未实现速度估计）
- ❌ **缺失**: PET 计算
- ❌ **缺失**: Level 分级（Collision/Near Miss/Avoidance）
- ❌ **缺失**: 时间戳记录（帧号、秒数）

**建议改进**:
```
1. 实现速度估计：v = Δposition / Δtime
2. 完整 TTC 公式实现：TTC = distance / relative_velocity
3. 增加事件分类：
   - Level 1: Collision (距离 < 0.5m)
   - Level 2: Near Miss (0.5m ≤ 距离 < 1.5m)  
   - Level 3: Avoidance (距离 ≥ 1.5m 但有交集迹象)
4. 输出包含时间戳、帧号等详细信息
```

---

## 📊 导师强调的关键功能需求

### 1. 可视化输出 ❌ 不足
**要求**: 动态绘制、边框、距离标注、level标记

**当前**: 
- ✅ 静态检测帧截图
- ❌ 无动态视频绘制
- ❌ 无距离/TTC 标注
- ❌ 无 level 标记

**需要实现**:
```python
# 伪代码
for frame in video:
    results = yolo.detect(frame)
    for det in results:
        # 绘制边框
        cv2.rectangle(frame, ...)
        # 标注 track_id
        cv2.putText(frame, f"ID:{track_id}", ...)
        # 标注距离/TTC
        cv2.putText(frame, f"Dist:{dist:.2f}m", ...)
        # 标注 level
        cv2.putText(frame, f"Level:{level}", ...)
    cv2.imwrite(frame)
```

---

### 2. 性能优化 ❌ 未实现
**导师建议**:
- 跳帧策略（frame skip）
- GPU 推理
- 分离式架构（推理与逻辑分离）

**当前状态**:
- ❌ 无跳帧策略
- ❌ 无 GPU 推理选项
- ❌ 未实现分离式架构

**需要实现**:
```python
# 跳帧策略
skip_frame = 3
for frame_idx, frame in enumerate(video):
    if frame_idx % skip_frame != 0:
        continue
    # 处理这一帧

# GPU 推理
model = YOLO("yolo11n.pt")
model.to("cuda")  # 或使用 device=0 参数

# 分离式推理
# 选项1: 本地 GPU 推理
# 选项2: 调用远程 Triton 服务
```

---

### 3. 检测精度 ⚠️ 需要改进
**问题**: 图像缩放导致车辆被误检为行人

**根本原因**:
- 输入分辨率过小（384×640）
- warped 视频分辨率变化（180×1200）
- 物体大小变化导致特征丢失

**解决方案**:
```
1. 调整输入分辨率（640×640 或更大）
2. 使用 segmentation 而非仅 detection
3. 尝试 YOLOv11-small/medium（更好的精度）
4. 对 person 类做特殊处理（检查高度比例）
```

---

### 4. 验证视频多样性 ❌ 未获取
**导师要求**: 找 5 种不同类型视频验证模型
- 晴天/阴天/雨天/夜间
- 不同的视角和交叉类型
- 明确会发生碰撞的视频

**当前**: 仅用了部分测试视频

**建议行动**:
```
1. 整理已有视频的分类
2. 标记出"明显碰撞"的视频片段
3. 补充缺失类型的视频
4. 为每个视频生成对比结果
```

---

### 5. 输出报告格式 ⚠️ 需要增强
**导师要求**:
- 文本信息 ✓
- 标注截图 ✓ 但不够丰富
- Level 分级 ❌ 无
- 时间戳 ⚠️ 部分有
- 帧数信息 ✓
- 丰富的可视化效果 ❌

**当前报告示例**:
```
检测统计:
  - 检测到物体的帧数: 57
  - 碰撞事件数: 0
```

**需要改进为**:
```
碰撞风险分析报告
===================

事件 #1
--------
时间: 00:05:23 (Frame 8000)
类型: Near Miss (Level 2)
物体对: Vehicle_42 ↔ Pedestrian_15
距离: 0.8m
TTC: 2.3s
相对速度: 0.35 m/s
截图: event_001_frame_8000.jpg

[ 可视化截图，带标注 ]

Level 统计:
- Level 1 (Collision): 0 events
- Level 2 (Near Miss): 3 events  
- Level 3 (Avoidance): 8 events
```

---

## 🔄 流程改进建议

### 当前流程:
```
1_homography/        → Homography标定 + 验证图
2_warped_video/      → 透视变换视频
3_collision_events/  → YOLO检测 + 距离计算
    ├── frames/      → 检测帧截图
    ├── collision_events.json
    └── analysis_report.txt
```

### 改进后流程:
```
1_homography/
    ├── homography.json
    ├── verify_original.jpg
    └── verify_grid_warp.jpg (新增：网格验证)

2_warped_video/
    ├── warped_video.mp4
    └── warped_video_stats.json (新增：尺寸/fps信息)

3_yolo_detection/
    ├── detection_results.json (检测框原始数据)
    └── detection_stats.json (检测统计)

4_tracking/
    ├── trajectories.json (完整轨迹)
    └── track_stats.json (追踪统计)

5_collision_analysis/
    ├── proximity_events.json (接近事件)
    ├── collision_events.json (碰撞事件，按level分类)
    ├── annotated_video.mp4 (新增：绘制标注的视频)
    ├── event_frames/
    │   ├── level1_collision_001.jpg
    │   ├── level2_nearmiss_001.jpg
    │   └── level3_avoidance_001.jpg
    └── analysis_report.txt (增强的报告)
```

---

## 📈 功能完成度评估

| 模块 | 功能 | 完成度 | 优先级 |
|------|------|--------|--------|
| 检测 | YOLO 检测 | 90% | 中 |
| 检测 | 精度改进 | 40% | 高 |
| 追踪 | Track ID 管理 | 80% | 中 |
| 坐标转换 | Homography 变换 | 95% | 中 |
| 坐标转换 | 性能优化 | 30% | 高 |
| 碰撞计算 | 距离检测 | 100% | 低 |
| 碰撞计算 | TTC 计算 | 20% | 高 |
| 碰撞计算 | PET 计算 | 0% | 高 |
| 碰撞计算 | Event 分级 | 0% | 高 |
| 可视化 | 静态截图 | 80% | 中 |
| 可视化 | 动态视频绘制 | 0% | 高 |
| 可视化 | 距离标注 | 0% | 高 |
| 性能 | 跳帧策略 | 0% | 中 |
| 性能 | GPU 推理 | 0% | 中 |
| 报告 | 文本报告 | 70% | 中 |
| 报告 | Level 分级统计 | 0% | 高 |
| 报告 | 详细时间戳 | 50% | 中 |

---

## ✅ 建议的优先级排序

### 第一阶段（关键）
1. ✅ 实现完整 TTC 计算
2. ✅ 实现 Event 分级（Collision/Near Miss/Avoidance）
3. ✅ 动态视频绘制（带标注）
4. ✅ 增强报告格式（包含 Level 统计）

### 第二阶段（重要）
5. ✅ 性能优化（跳帧、向量化）
6. ✅ 检测精度改进（调整输入分辨率或使用更大模型）
7. ✅ GPU 推理选项

### 第三阶段（补充）
8. ✅ Segmentation 集成
9. ✅ PET 计算
10. ✅ 验证视频多样性测试

---

## 📝 总体结论

**当前状态**: 框架完整，基础功能 60-70% 完成

**主要缺口**: 
- TTC/PET 完整实现
- Event 分级和分类
- 动态可视化输出
- 性能优化

**预计补完时间**: 7-10 天（按照优先级）

**建议**: 专注第一阶段（TTC + 分级 + 可视化），这些是导师和演示最关注的部分。
