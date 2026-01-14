# 🔄 两个分支的快速对比

**分支1**: `approach-homography-first` ← 当前已完成框架  
**分支2**: `approach-yolo-first` ← 新分支，待实现  

---

## 📊 架构对比

### Approach 1: Homography-First (已实现)

```
原始视频 
    ↓
[1] Homography 标定
    ↓ (验证H矩阵)
[2] 视频透视变换
    ↓ (所有帧 → warped video)
[3] YOLO 检测 (在 warped 视频上)
    ↓ (检测框架、轨迹)
[4] 距离计算 (已是世界坐标)
    ↓ (1.5m 阈值)
[5] TTC 计算 (未完成)
    ↓ (需要实现速度估计)
[6] Event 分级 (未完成)
    ↓ (L1/L2/L3 分类)
[7] 视频绘制 (未完成)
    ↓ (标注输出视频)
[8] 报告生成
```

**优点**:
- ✅ 所有处理都在世界坐标空间
- ✅ 距离阈值固定且可理解 (1.5m = ~2车道)
- ✅ 坐标变换一次性完成
- ✅ 便于全帧的整体分析

**缺点**:
- ❌ 计算量大 (整个视频透视变换)
- ❌ 需要 Homography 标定好才能开始
- ❌ 如果 H 矩阵有问题会影响后续所有步骤

---

### Approach 2: YOLO-First (待实现)

```
原始视频
    ↓
[1] YOLO 检测 (原始分辨率)
    ↓ (检测所有物体)
[2] 轨迹构建 (像素空间)
    ↓ (Track ID + 速度估计)
[3] 关键帧提取
    ↓ (接近事件帧只有几十帧)
[4] Homography 变换 (仅关键帧)
    ↓ (坐标转换)
[5] 坐标和速度转换 (px → 米)
    ↓ (应用 H 矩阵)
[6] TTC 计算 (世界坐标)
    ↓ (速度已转换为 m/s)
[7] Event 分级
    ↓ (L1/L2/L3)
[8] 视频绘制 + 报告
```

**优点**:
- ✅ 计算量小 (仅处理关键帧)
- ✅ 不依赖 H 矩阵就能检测
- ✅ 更灵活的流程控制
- ✅ 可以处理任意分辨率视频
- ✅ 关键帧的坐标变换精度高

**缺点**:
- ❌ 需要在像素空间判断接近 (阈值动态调整)
- ❌ 速度估计在像素空间 (需要单位转换)
- ❌ 轨迹关联更复杂 (无世界坐标参考)

---

## 📈 性能对比预测

| 指标 | Homography-First | YOLO-First |
|------|-----------------|-----------|
| 视频处理时间 | ~30-50秒 (整个视频) | ~10-20秒 (仅检测) |
| Homography耗时 | 5-10秒 (全帧转换) | <1秒 (仅关键帧) |
| 内存峰值 | 高 (整个warped video) | 低 (关键帧缓存) |
| 总时间 | ~40-60秒 | ~15-30秒 |
| 速度提升 | - | ~2-3倍 |

---

## 💡 应用场景

### 选择 Homography-First 的情况
- 需要完整的视频分析
- 需要生成带标注的完整视频
- 已有高质量的 Homography 标定
- 计算资源充足

### 选择 YOLO-First 的情况
- 仅关心关键事件 (碰撞/近miss)
- 需要快速处理
- 计算资源有限
- 想要灵活的流程控制
- 不确定 Homography 标定质量

---

## 🎯 导师可能的选择

根据导师的会议摘要，可能的倾向:

> "轨迹获取后的计算都是有限的计算，基本上都可以实现"

这意味着:
1. **不强制要求全帧处理** → YOLO-First 可接受
2. **重点在关键事件分析** → YOLO-First 更合适
3. **可视化重要** → 两个方案都需要

**导师可能认同 YOLO-First** 因为:
- ✅ 性能更好 (2-3倍提升)
- ✅ 灵活性高
- ✅ 仍能达到演示效果

---

## 📝 两个分支的共同点

### 都需要实现的部分

1. **TTC 完整计算**
   ```python
   v = estimate_velocity(track)
   v_rel = v1 - v2
   ttc = distance / |v_rel|
   ```

2. **Event 分级**
   ```python
   if distance < 0.5m or ttc < 1.0s:
       level = 1  # Collision
   elif distance < 1.5m and ttc < 3.0s:
       level = 2  # Near Miss
   else:
       level = 3  # Avoidance
   ```

3. **动态视频绘制**
   ```python
   draw_bboxes()  # 边框
   draw_track_id()  # ID
   draw_distance()  # 距离
   draw_ttc()  # TTC
   draw_level_color()  # 颜色标记
   ```

4. **报告生成**
   - Level 统计
   - 事件详情
   - 时间戳信息

---

## 🔄 如何同时开发两个分支

### Git 命令

```bash
# 查看当前在哪个分支
git branch

# 切换分支
git checkout approach-homography-first
git checkout approach-yolo-first

# 查看分支差异
git diff approach-homography-first approach-yolo-first

# 合并分支 (最后)
git checkout main
git merge approach-homography-first
git merge approach-yolo-first
```

### 推荐工作流

```
Day 1-2:  在 approach-yolo-first 上实现 Phase 1-3
          (YOLO detection + trajectory + key frame extraction)

Day 3:    测试 approach-yolo-first
          收集一些初步结果

Day 4:    根据结果评估两个方案的优缺点

Day 5-6:  等待导师反馈
          - 如果选择 approach-homography-first: 
            切回该分支，完成 TTC + 分级 + 视频绘制
          - 如果选择 approach-yolo-first:
            继续完善该分支

Day 7:    性能优化、精度改进、PPT 准备
```

---

## 📊 当前状态

| 项目 | Homography-First | YOLO-First |
|------|-----------------|-----------|
| 分支 | ✅ 已创建 | ✅ 已创建 |
| 框架 | ✅ 90% 完成 | ⏳ 待实现 |
| YOLO 检测 | ✅ 完成 | ⏳ 待实现 |
| Homography变换 | ✅ 完成 | ⏳ 待实现 |
| 距离计算 | ✅ 完成 | ⏳ 待实现 |
| TTC 计算 | ⏳ 20% | ⏳ 0% |
| Event 分级 | ⏳ 0% | ⏳ 0% |
| 视频绘制 | ⏳ 0% | ⏳ 0% |
| 报告生成 | ✅ 70% | ⏳ 0% |

---

## 🎯 立即行动

### 保存当前分支 ✅ (已完成)
```bash
git branch -m main approach-homography-first
git commit -m "Approach 1: Homography-first Pipeline"
```

### 创建新分支 ✅ (已完成)
```bash
git branch approach-yolo-first
git checkout approach-yolo-first
```

### 接下来
1. 在 `approach-yolo-first` 上实现 YOLO-first pipeline
2. 生成对比结果
3. 等导师回复后选择继续方向
4. 基于选择完成相应分支

---

## 📚 相关文档位置

- **Approach 1 详细信息**: 无独立文档 (当前已完成的 pipeline)
- **Approach 2 详细信息**: [YOLO_FIRST_APPROACH.md](YOLO_FIRST_APPROACH.md)
- **分支管理**: 本文档

---

## ✅ 检查清单

- [x] 保存 approach-homography-first 分支
- [x] 创建 approach-yolo-first 分支
- [x] 当前切换到 approach-yolo-first
- [x] 创建 YOLO_FIRST_APPROACH.md 说明文档
- [x] 创建两个分支的对比文档 (本文档)
- [ ] 在 approach-yolo-first 上实现 Phase 1-3
- [ ] 测试并生成对比结果
- [ ] 等待导师反馈
- [ ] 根据反馈选择最终方案

---

**当前分支**: `approach-yolo-first`  
**当前状态**: 准备就绪，可以开始实现  
**下一步**: 开始实现 YOLODetector 和 TrajectoryBuilder

💡 **提示**: 两个分支完全独立，可以同时开发，最后再根据导师反馈选择最优方案！
