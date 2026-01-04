# Trajectory Demo - 轨迹追踪和碰撞检测

## 核心文件说明

### Homography标定
- **calibration.py** - 从像素坐标和世界坐标计算Homography矩阵
  - 输入：像素坐标（4个点）和世界坐标（4个点）
  - 输出：`{video_name}_homography.json`
  
示例用法：
```bash
python calibration.py \
  --pixel-points "100,50 1800,80 1850,1000 120,1050" \
  --world-points "0,0 12,0 12,8 0,8" \
  --video-name MyVideo \
  --output calibration/
```

### 坐标转换
- **coord_transform.py** - 使用Homography矩阵进行坐标转换
  - 加载Homography矩阵
  - 将像素坐标转换为世界坐标（米）
  - 计算世界坐标中的距离

### 检测和追踪
- **detection_adapter.py** - 适配YOLO检测结果
- **object_state_manager.py** - 管理物体状态、轨迹和速度
- **trajectory_prediction.py** - 轨迹预测

### 可视化
- **visualize_collision_events.py** - 可视化碰撞事件
- **visualize_contact_points.py** - 可视化接触点

### 主程序
- **yolo_runner.py** - 完整的目标追踪和碰撞分析流程

## 工作流程

1. **标定阶段** - 获取Homography矩阵
   ```bash
   python calibration.py --pixel-points "..." --world-points "..." ...
   ```

2. **分析阶段** - 运行完整的追踪和碰撞检测
   ```bash
   python yolo_runner.py \
     --source videos/MyVideo.mp4 \
     --homography calibration/MyVideo_homography.json \
     --output runs/trajectory_demo
   ```

## 输出文件

在输出目录下生成：
- `tracks.json` - 所有物体的轨迹数据
- `near_misses.json` - 碰撞接近事件
- `analysis_report.txt` - 统计分析报告
- `visualization.mp4` - 可视化视频（可选）
