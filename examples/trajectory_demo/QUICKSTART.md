# 快速开始

## 你现在的数据

等待你提供：
- 4个像素坐标 (从视频截图中测量)
- 4个对应的世界坐标 (实际距离, 单位：米)

## 步骤1：标定Homography

当你准备好数据后，运行：

```bash
cd /workspace/ultralytics/examples/trajectory_demo
python calibration.py \
  --pixel-points "px1,py1 px2,py2 px3,py3 px4,py4" \
  --world-points "x1,y1 x2,y2 x3,y3 x4,y4" \
  --video-name YourVideoName \
  --output ../../calibration/
```

例如（假设你的数据）：
```bash
python calibration.py \
  --pixel-points "100,50 1800,80 1850,1000 120,1050" \
  --world-points "0,0 12,0 12,8 0,8" \
  --video-name Homograph_Teset_FullScreen \
  --output ../../calibration/
```

输出：`calibration/Homograph_Teset_FullScreen_homography.json`

## 步骤2：运行完整分析

```bash
python yolo_runner.py \
  --source ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json \
  --output ../../runs/trajectory_demo/
```

输出目录包含：
- `tracks.json` - 轨迹数据
- `near_misses.json` - 碰撞接近事件
- `analysis_report.txt` - 分析报告

准备好你的坐标数据时告诉我！
