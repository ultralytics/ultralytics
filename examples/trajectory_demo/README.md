# 📖 碰撞检测Pipeline - 文档索引

欢迎使用完整的碰撞检测Pipeline！本文档帮助您快速定位所需的信息。

---

## 🚀 我想立即开始

**→ 阅读**: [QUICK_START.md](QUICK_START.md) (5分钟)

快速命令：

```bash
cd /workspace/ultralytics/examples/trajectory_demo
python run_pipeline.py \
  --video ../../videos/Homograph_Teset_FullScreen.mp4 \
  --homography ../../calibration/Homograph_Teset_FullScreen_homography.json
```

---

## 📚 我想了解详细用法

**→ 阅读**: [PIPELINE_USAGE.md](PIPELINE_USAGE.md) (15分钟)

涵盖内容：

- 完整的参数说明
- 输出目录结构详解
- Pipeline三个阶段的工作流
- 调试和问题排查
- 性能优化建议

---

## 🏗️ 我想了解项目结构

**→ 阅读**: [STRUCTURE.txt](STRUCTURE.txt)

包含：

- 完整的文件夹结构图
- 快速命令参考
- 各个阶段的输入输出
- 可修改的代码参数
- JSON格式说明

---

## ✅ 我想看实现总结

**→ 阅读**: [SUMMARY.md](SUMMARY.md)

包含：

- 实现目标和完成情况
- 运行示例和输出
- 关键技术说明
- 常见问题解答
- 自定义配置指南

---

## 🎯 我想查看运行结果

**→ 查看**: `/workspace/ultralytics/results/20251218_225957/`

结构：

```
20251218_225957/
├── 1_homography/          # Homography标定结果
├── 2_warped_video/        # 变换后的视频
├── 3_collision_events/    # 检测结果
└── README.md              # 本次运行说明
```

**查看报告**：

```bash
cat /workspace/ultralytics/results/20251218_225957/3_collision_events/analysis_report.txt
```

---

## 🔍 我想了解特定功能

### Homography标定

- 什么是Homography矩阵？ → [PIPELINE_USAGE.md#关键矩阵](PIPELINE_USAGE.md)
- 如何验证标定？ → [PIPELINE_USAGE.md#步骤1](PIPELINE_USAGE.md)
- 如何创建新的标定？ → [calibration.py](calibration.py)

### 视频透视变换

- 如何实现鸟瞰图转换？ → [PIPELINE_USAGE.md#步骤2](PIPELINE_USAGE.md)
- 变换矩阵如何计算？ → [collision_detection_pipeline.py#L80-L100](collision_detection_pipeline.py)
- 如何调整输出分辨率？ → [SUMMARY.md#修改输出分辨率](SUMMARY.md)

### 碰撞检测

- 如何定义碰撞事件？ → [PIPELINE_USAGE.md#碰撞定义](PIPELINE_USAGE.md)
- 如何修改距离阈值？ → [SUMMARY.md#修改碰撞距离阈值](SUMMARY.md)
- YOLO如何运行？ → [collision_detection_pipeline.py#L115](collision_detection_pipeline.py)

---

## 🛠️ 我想自定义参数

### 运行时参数（命令行）

```bash
python run_pipeline.py --help
```

主要参数：

- `--video`: 输入视频
- `--homography`: Homography JSON
- `--output`: 结果目录
- `--conf`: YOLO置信度

### 代码参数（修改源文件）

| 参数       | 位置                                | 说明         |
| ---------- | ----------------------------------- | ------------ |
| 碰撞距离   | collision_detection_pipeline.py:150 | 默认0.5m     |
| 输出分辨率 | collision_detection_pipeline.py:90  | 默认180×1200 |
| YOLO模型   | collision_detection_pipeline.py:115 | 默认yolo11n  |

→ 详见: [SUMMARY.md#自定义配置](SUMMARY.md#自定义配置)

---

## ❓ 我遇到了问题

### 问题排查流程

1. **检查文件存在**

   ```bash
   ls -la ../../videos/Homograph_Teset_FullScreen.mp4
   ls -la ../../calibration/Homograph_Teset_FullScreen_homography.json
   ```

2. **查看错误消息**
   - Pipeline会显示详细的错误信息
   - 查看 `analysis_report.txt` 中的摘要

3. **查看常见问题**
   → [SUMMARY.md#常见问题与解决方案](SUMMARY.md#常见问题与解决方案)
   → [PIPELINE_USAGE.md#问题排查](PIPELINE_USAGE.md#问题排查)

### 常见问题快速查找

| 问题             | 原因               | 解决                    |
| ---------------- | ------------------ | ----------------------- |
| 未检测到物体     | 物体太小或置信度高 | `--conf 0.3`            |
| 检测太灵敏       | 置信度太低         | `--conf 0.6`            |
| Warped视频质量差 | 标定不准确         | 检查verify_original.jpg |
| 处理速度慢       | 模型太大           | 使用yolo11n.pt          |

---

## 📊 我想处理新视频

### 第1步：创建Homography标定

```bash
cd /workspace/ultralytics/examples/trajectory_demo

python calibration.py \
  --pixel-points "x1,y1 x2,y2 x3,y3 x4,y4" \
  --world-points "wx1,wy1 wx2,wy2 wx3,wy3 wx4,wy4"
```

→ 详见: `calibration.py` 或 [PIPELINE_USAGE.md#修改Homography参考坐标](PIPELINE_USAGE.md)

### 第2步：运行Pipeline

```bash
python run_pipeline.py \
  --video path/to/your_video.mp4 \
  --homography path/to/your_homography.json
```

### 第3步：查看结果

```bash
ls -lh /workspace/ultralytics/results/
cat /workspace/ultralytics/results/[timestamp]/3_collision_events/analysis_report.txt
```

---

## 🚀 我想集成到自动化工作流

### Python脚本示例

```python
import subprocess

# 定义输入
videos = ["video1.mp4", "video2.mp4"]
homographies = ["h1.json", "h2.json"]

# 批量处理
for video, h in zip(videos, homographies):
    result = subprocess.run(["python", "run_pipeline.py", "--video", video, "--homography", h, "--conf", "0.4"])
    if result.returncode == 0:
        print(f"✓ {video} 处理完成")
    else:
        print(f"✗ {video} 处理失败")
```

---

## 📋 核心文件一览

| 文件                              | 说明         | 用途           |
| --------------------------------- | ------------ | -------------- |
| `run_pipeline.py`                 | 启动脚本     | 运行Pipeline   |
| `collision_detection_pipeline.py` | Pipeline核心 | 3个阶段实现    |
| `calibration.py`                  | 标定工具     | 生成Homography |
| `yolo_runner.py`                  | 检测器       | YOLO推理       |
| `coord_transform.py`              | 坐标变换     | 工具函数       |
| `QUICK_START.md`                  | 快速开始     | 5分钟上手      |
| `PIPELINE_USAGE.md`               | 详细说明     | 完整文档       |
| `SUMMARY.md`                      | 完成总结     | 技术细节       |

---

## 🎓 我想深入理解原理

### Homography矩阵

- **定义**: 3×3矩阵，将像素坐标映射到世界坐标
- **计算**: 使用4个参考点对求解
- **验证**: 通过`verify_original.jpg`中的参考点检查

→ 详见: `coord_transform.py` 中的 `load_homography()`

### 透视变换

- **原理**: 应用变换矩阵 M = H_inv @ A
  - H_inv: 世界→像素（Homography的逆矩阵）
  - A: 输出坐标→世界坐标

→ 详见: `collision_detection_pipeline.py` 中的 `transform_video()`

### 碰撞检测

- **算法**: 计算所有物体对之间的距离
- **坐标系**: 使用世界坐标（通过Homography变换）
- **阈值**: 距离 < 0.5m 标记为碰撞

→ 详见: `collision_detection_pipeline.py` 中的 `detect_collisions()`

---

## 💡 快速参考

### 常用命令

```bash
# 查看帮助
python run_pipeline.py --help

# 运行示例
python run_pipeline.py --video ../../videos/test.mp4 --homography ../../calibration/h.json

# 查看最新结果
ls -lh /workspace/ultralytics/results/ | head -3

# 查看报告
cat /workspace/ultralytics/results/[latest]/3_collision_events/analysis_report.txt

# 查看事件
cat /workspace/ultralytics/results/[latest]/3_collision_events/collision_events.json

# 列出所有事件帧
ls /workspace/ultralytics/results/[latest]/3_collision_events/event_frame_*.jpg
```

### 文件路径

```
Pipeline代码: /workspace/ultralytics/examples/trajectory_demo/
输入视频: /workspace/ultralytics/videos/
输入标定: /workspace/ultralytics/calibration/
输出结果: /workspace/ultralytics/results/[timestamp]/
```

---

## 🎯 学习路径

**初级用户**

1. 阅读 [QUICK_START.md](QUICK_START.md)
2. 运行示例Pipeline
3. 查看结果目录结构

**中级用户**

1. 学习 [PIPELINE_USAGE.md](PIPELINE_USAGE.md)
2. 调整运行参数
3. 处理自己的视频

**高级用户**

1. 研究 [SUMMARY.md](SUMMARY.md) 的技术细节
2. 修改源代码参数
3. 集成到自动化工作流
4. 开发自定义扩展

---

## ✅ 验证安装

检查所有必要文件：

```bash
cd /workspace/ultralytics/examples/trajectory_demo

# 检查脚本
ls -la run_pipeline.py collision_detection_pipeline.py

# 检查文档
ls -la QUICK_START.md PIPELINE_USAGE.md SUMMARY.md

# 检查示例数据
ls -la ../../videos/Homograph_Teset_FullScreen.mp4
ls -la ../../calibration/Homograph_Teset_FullScreen_homography.json
```

所有文件都存在 → ✓ 安装完成！

---

## 🎉 现在就开始！

```bash
cd /workspace/ultralytics/examples/trajectory_demo
python run_pipeline.py --video ../../videos/Homograph_Teset_FullScreen.mp4 --homography ../../calibration/Homograph_Teset_FullScreen_homography.json
```

祝你使用愉快！

---

**最后更新**: 2025-12-18  
**状态**: ✅ 完成并测试  
**文档版本**: 1.0
