# Pipeline使用指南

## 📌 快速开始

### 基本用法

```bash
cd /workspace/ultralytics/examples/trajectory_demo

python run_pipeline.py \
  --video ../../videos/YOUR_VIDEO.mp4 \
  --homography ../../calibration/YOUR_HOMOGRAPHY.json
```

### 已有示例

```bash
cd /workspace/ultralytics/examples/trajectory_demo

python run_pipeline.py \
  --video ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json
```

## 🎛️ 参数说明

| 参数 | 必需 | 说明 | 默认值 |
|------|------|------|--------|
| `--video` | ✓ | 输入视频路径 | - |
| `--homography` | ✓ | Homography JSON文件路径 | - |
| `--output` | ✗ | 结果基础目录 | `../../results` |
| `--conf` | ✗ | YOLO置信度阈值 | 0.45 |

## 📂 输出目录结构

每次运行会生成一个**时间戳文件夹**（YYYYMMDD_HHMMSS格式），内部结构如下：

```
results/
└── 20251218_225957/              # 时间戳（年月日_时分秒）
    ├── 1_homography/             # Homography标定结果
    │   ├── homography.json       # 矩阵和参考点
    │   └── verify_original.jpg   # 原始视频验证图
    │
    ├── 2_warped_video/           # 透视变换结果
    │   └── warped.mp4            # 变换后的视频
    │
    ├── 3_collision_events/       # 碰撞检测结果
    │   ├── collision_events.json  # 事件数据（JSON）
    │   ├── analysis_report.txt    # 分析报告
    │   ├── event_frame_0001.jpg   # 碰撞事件帧（如有）
    │   ├── event_frame_0002.jpg
    │   └── ...
    │
    └── README.md                 # 本次运行说明
```

## 🔄 Pipeline流程

### 步骤1: Homography标定 (1_homography/)

**输入**: Homography JSON文件
**输出**:
- `homography.json` - 矩阵 + 参考点备份
- `verify_original.jpg` - 原始视频第一帧，标注4个参考点

**作用**: 
- 验证Homography矩阵正确性
- 确保参考点标注准确

### 步骤2: 视频透视变换 (2_warped_video/)

**输入**: 原始视频 + Homography矩阵
**输出**: `warped.mp4` - 鸟瞰图视频

**转换细节**:
- 输出分辨率: 180×1200像素
- 世界坐标范围: X∈[-3.75, 3.75]m, Y∈[0, 50]m
- 帧率: 与原始视频相同

**作用**:
- 将倾斜视角转为俯视（鸟瞰）视角
- 为后续的碰撞检测提供规范化的坐标系统

### 步骤3: YOLO检测 + 碰撞分析 (3_collision_events/)

**输入**: Warped视频 + YOLO模型（yolo11n.pt）
**输出**:
- `collision_events.json` - 检测到的所有碰撞事件
- `analysis_report.txt` - 分析摘要
- `event_frame_*.jpg` - 每个事件的帧截图

**碰撞定义**:
- 任何两个物体之间的距离 < 0.5m
- 基于世界坐标（通过Homography矩阵转换）

**JSON格式**:
```json
[
  {
    "frame": 15,           # 帧号
    "time": 0.5,          # 时间戳（秒）
    "object_ids": [1, 2], # 涉及的物体ID
    "distance": 0.45,     # 距离（米）
    "distance_str": "0.45m",
    "frame_image": "event_frame_0015.jpg"
  }
]
```

## 📊 调试与问题排查

### 问题1: 未检测到物体

**原因**:
1. Warped视频中物体太小
2. YOLO置信度阈值过高

**解决方案**:
```bash
# 降低置信度阈值
python run_pipeline.py \
  --video ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \
  --conf 0.3
```

### 问题2: Warped视频质量差

**检查事项**:
1. 查看 `verify_original.jpg` 是否正确标注了参考点
2. 确认Homography矩阵的标定精度（查看JSON中的calibration_error）
3. 检查原始视频是否清晰

### 问题3: 碰撞事件过多或过少

**调整方法**:
1. 修改碰撞距离阈值（当前：0.5m）
   - 在 `collision_detection_pipeline.py` 中找到 `if distance < 0.5` 
   - 改为所需的距离值

2. 修改YOLO置信度（影响检测敏感性）

## 🛠️ 自定义修改

### 修改碰撞距离阈值

编辑 `collision_detection_pipeline.py`，找到：
```python
if distance < 0.5 or (H is None and distance < 50):
    # 保存碰撞事件
```

改为所需的距离（单位：米）。

### 修改输出视频分辨率

编辑 `collision_detection_pipeline.py`，找到：
```python
output_size = (180, 1200)
```

改为所需的分辨率（宽度, 高度）。

### 修改Homography参考坐标

编辑 Homography JSON文件：
```json
{
  "pixel_points": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "world_points": [[wx1,wy1], [wx2,wy2], [wx3,wy3], [wx4,wy4]],
  ...
}
```

## 📈 性能优化

### 加快处理速度

1. **降低视频分辨率** (在source编码时):
   ```bash
   # 使用-scale参数在推理前缩放
   ```

2. **减少追踪帧数**:
   - 修改 `detect_collisions()` 中的采样间隔

3. **使用更轻量的模型**:
   ```python
   model = YOLO('yolo11s.pt')  # 改用s版本
   ```

### 保存更多信息

当前只保存了碰撞事件帧。若需保存所有检测帧：

编辑 `collision_detection_pipeline.py`，在 `detect_collisions()` 中添加：
```python
# 保存每一帧（会很慢，占用大量磁盘）
frame_path = self.collision_dir / f"frame_{frame_count:04d}.jpg"
cv2.imwrite(str(frame_path), frame_img)
```

## 📝 常用命令

```bash
# 进入目录
cd /workspace/ultralytics/examples/trajectory_demo

# 查看帮助
python run_pipeline.py --help

# 基础运行
python run_pipeline.py --video ../../videos/Homograph_Teset_FullScreen.mp4 --homography ../../calibration/Homograph_Teset_FullScreen_homography.json

# 低置信度运行（更敏感）
python run_pipeline.py --video ../../videos/Homograph_Teset_FullScreen.mp4 --homography ../../calibration/Homograph_Teset_FullScreen_homography.json --conf 0.3

# 查看最新运行结果
ls -lh /workspace/ultralytics/results/ | tail -5

# 查看特定运行的详细报告
cat /workspace/ultralytics/results/20251218_225957/3_collision_events/analysis_report.txt

# 列出所有碰撞事件
cat /workspace/ultralytics/results/20251218_225957/3_collision_events/collision_events.json | python -m json.tool
```

---

**上次更新**: 2025-12-18
