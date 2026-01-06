# 实现优化建议清单 & 路线图

**基于导师会议分析** | **优先级排序** | **预计工作量**

---

## 🔴 高优先级 (必须完成)

### H1: 完整的TTC计算器 [2-3天]
**目标**: 实现导师提到的"Time-to-Collision"计算  
**文件**: `examples/trajectory_demo/ttc_calculator.py`

**当前状态**: 基础实现存在，但缺少完整的物理模型

**改进内容**:
```python
class TTCCalculator:
    def estimate_velocity_robust(track_history, window_size=5):
        """
        改进版速度估计 - 使用最小二乘法
        - 降低噪声影响
        - 支持长轨迹的拟合
        - 返回速度向量 (vx, vy) 和可信度分数
        """
        
    def calculate_ttc_physics_based(pos1, vel1, pos2, vel2):
        """
        基于物理的TTC计算
        - 投影速度到接近轴 (closing axis)
        - TTC = distance / relative_closing_velocity
        - 处理速度相近/相反的情况
        """
        
    def calculate_pet(trajectory1, trajectory2):
        """
        Post-Encroachment Time (碰撞后时间差)
        - 找到两条轨迹的最近点
        - 计算各自通过该点的时间
        - PET = time2 - time1
        """
```

**验收标准**:
- [ ] TTC计算结果与专业工具对齐 (差异<5%)
- [ ] 支持平行运动、相交、碰撞等各种场景
- [ ] 返回置信度指标
- [ ] 单元测试通过 (10+个测试用例)

**关键参考**:
> 导师会议: "需要实现TTC...形成一个三维数据结构（X、Y坐标加时间轴）"
> → 现有ObjectStateManager已有(X,Y,T)，TTCCalculator需要基于此计算

---

### H2: 事件分类器的完整实现 [1-2天]
**目标**: 按导师标准分级碰撞事件  
**文件**: `examples/trajectory_demo/event_classifier.py`

**当前状态**: 框架存在，逻辑过于简单

**改进内容**:
```python
class EventClassifier:
    """
    导师提到的风险分级：
    - Level 1 (Collision)：距离<0.5m OR TTC<1.0s
    - Level 2 (Near Miss)：0.5m ≤ d < 1.5m AND TTC<3.0s  
    - Level 3 (Avoidance)：有接近但>1.5m
    """
    
    def classify_event(distance, ttc, pet, 
                      velocity1, velocity2,
                      object_types):
        """
        多维评分系统：
        1. 距离评分 (0-100)
        2. TTC评分 (0-100) 
        3. PET评分 (0-100)
        4. 类别权重 (car/car > car/person > 其他)
        5. 动向评分 (是否快速接近)
        
        返回: (risk_level, confidence, detailed_scores)
        """
        
    def filter_false_positives(event_list):
        """
        虚警过滤：
        - 同一对物体的短期多次检测应合并
        - 静止物体的距离波动应忽略
        - 只保留真实的接近趋势
        """
```

**验收标准**:
- [ ] 与导师的分级标准100%匹配
- [ ] 有虚警过滤机制
- [ ] 同一事件的重复检测被合并
- [ ] 输出JSON格式清晰

**关键参考**:
> 导师会议: "需要设计状态管理器来记录和管理所有目标的状态信息"
> → EventClassifier应基于ObjectStateManager的数据做分析

---

### H3: 轨迹预测模型升级 [3-4天]
**目标**: 从线性预测升级到多项式+卡尔曼  
**文件**: `examples/trajectory_demo/trajectory_prediction.py`

**当前状态**: 仅线性外推

**改进内容**:

```python
class TrajectoryPredictor:
    """
    三层预测系统：
    1. 线性预测 (baseline)
    2. 二阶多项式预测 (高精度)
    3. 卡尔曼滤波 (噪声抑制)
    """
    
    def predict_quadratic(track_history, ahead_frames=10):
        """
        二阶多项式拟合 (抛物线)
        - 假设运动方程: y = a*t^2 + b*t + c
        - 使用最小二乘法拟合
        - 输出: 拟合参数 + 预测轨迹 + 拟合误差
        """
        
    def kalman_smoother(noisy_track):
        """
        卡尔曼滤波器：
        - 状态向量: [x, y, vx, vy]
        - 处理运动模型噪声和测量噪声
        - 输出: 平滑的轨迹 + 不确定性椭圆
        """
        
    def detect_anomalies(track_history, threshold=3.0):
        """
        异常值检测 (标准差方法)
        - 识别跳变、跟踪失败等
        - 可选修复或标记
        """
        
    def predict_collision_with_confidence(
        track1, track2, 
        prediction_method='kalman',
        ahead_time=2.0
    ):
        """
        碰撞预测 + 置信度
        - 预测future positions
        - 计算碰撞概率
        - 返回: collision_point, time_to_collision, confidence
        """
```

**验收标准**:
- [ ] 轨迹预测误差 < 10% (vs 真实轨迹)
- [ ] 卡尔曼滤波有效降低噪声
- [ ] 支持异常值检测和修复
- [ ] 单元测试 (15+个)

**关键参考**:
> 导师会议: "需要完成轨迹预测算法...基于Python进行拟合、预测和计算"
> → 现有线性版本满足基本要求，但二阶/卡尔曼会显著提升精度

---

## 🟡 中优先级 (重要但不紧急)

### M1: 多视图全局跟踪系统 [5-7天]
**目标**: 实现导师强调的"多视图物体跟踪"  
**新文件**: `examples/trajectory_demo/global_tracker.py`

**为什么重要**:
> 导师会议: "最困难的技术挑战是如何在相机视图中跟踪物体，特别是当物体从当前视图中消失或移动到其他视图时"

**设计框架**:
```python
class GlobalTracker:
    """
    多摄像头全局ID管理系统
    """
    
    def __init__(self, num_cameras=2):
        self.local_trackers = {}  # camera_id -> ObjectStateManager
        self.global_id_map = {}   # (camera_id, local_id) -> global_id
        self.inter_camera_matches = {}  # 跨摄像头关联记录
        
    def update_camera_detection(camera_id, detections, timestamp):
        """
        接收单个摄像头的检测结果
        - 本地YOLO跟踪
        - 与其他摄像头的ID统一
        """
        
    def match_across_cameras(local_detections_1, local_detections_2):
        """
        基于多特征进行跨摄像头关联：
        1. 外观特征 (appearance) - 颜色直方图
        2. 时空特征 (spatio-temporal) - 位置、时间连续性
        3. 语义特征 (semantic) - 物体类别、大小
        
        返回: 匹配对列表 [(local_id_1, local_id_2, confidence), ...]
        """
        
    def unify_ids(matches, timestamp):
        """
        统一ID：
        - 为跨摄像头关联的物体分配相同的global_id
        - 处理ID冲突和歧义
        """
```

**实现步骤**:
1. Extract appearance features (color histogram, CNN embeddings)
2. Build spatial-temporal model (Kalman filter per camera)
3. Implement Hungarian algorithm for ID matching
4. Handle view transitions and occlusions

**验收标准**:
- [ ] 跨摄像头ID一致性 > 90%
- [ ] 支持至少2个摄像头
- [ ] 处理视图切换的ID连续性
- [ ] 集成测试 (3+场景)

---

### M2: 更新Pipeline集成两个新模块 [2天]
**文件**: `examples/trajectory_demo/collision_detection_pipeline.py` 修改

**改进内容**:
```python
# 在 run() 函数中添加以下流程：

# 之前的流程
results = model.track(frame)
osm.update(detections, t)

# 新增：TTC计算
ttc_calc = TTCCalculator()
for id1, id2 in get_object_pairs(osm):
    traj1 = osm.get_trajectory(id1, last_n=10)
    traj2 = osm.get_trajectory(id2, last_n=10)
    
    vel1 = ttc_calc.estimate_velocity_robust(traj1)
    vel2 = ttc_calc.estimate_velocity_robust(traj2)
    
    distance = osm.distance_between(id1, id2)
    ttc = ttc_calc.calculate_ttc(
        osm._get_point_at(id1),
        vel1, 
        osm._get_point_at(id2),
        vel2,
        distance
    )
    
    # 新增：事件分类
    classifier = EventClassifier()
    risk_level, name = classifier.classify_event(
        distance=distance,
        ttc=ttc,
        pet=None,  # TODO
        velocity1=vel1,
        velocity2=vel2,
        object_types=(osm.get_class(id1), osm.get_class(id2))
    )
    
    if risk_level <= 2:  # 记录碰撞和近miss
        event = {
            'frame': frame_idx,
            'time': t,
            'object_ids': [id1, id2],
            'distance': distance,
            'ttc': ttc,
            'risk_level': risk_level,
            'risk_name': name,
            'velocities': {'obj_' + str(id1): vel1, 'obj_' + str(id2): vel2}
        }
        all_events.append(event)
```

**验收标准**:
- [ ] Pipeline运行无错误
- [ ] 输出JSON包含TTc和risk_level字段
- [ ] 与之前的输出向后兼容（可选字段可缺省）

---

### M3: 完整的单元测试套件 [2-3天]
**新文件**: `examples/trajectory_demo/test_integration.py`

```python
class TestTTCCalculator(unittest.TestCase):
    def test_linear_collision_head_on(self):
        """两物体对向运动，应能预测碰撞"""
        
    def test_parallel_motion(self):
        """两物体平行运动，TTC应为无穷"""
        
    def test_noise_robustness(self):
        """在噪声轨迹上计算TTC，结果应稳定"""

class TestEventClassifier(unittest.TestCase):
    def test_collision_level(self):
        """distance<0.5m, TTC<1s → Level 1"""
        
    def test_near_miss_level(self):
        """distance in [0.5, 1.5], TTC<3s → Level 2"""

class TestTrajectoryPredictor(unittest.TestCase):
    def test_quadratic_fit_accuracy(self):
        """二阶拟合误差 < 10%"""
        
    def test_kalman_smoothing(self):
        """卡尔曼滤波应降低轨迹抖动"""
```

---

## 🟢 低优先级 (可选优化)

### L1: 文档完善 [1-2天]
- 编写 `API_REFERENCE.md` (每个模块的接口文档)
- 编写 `TUTORIAL.md` (从零开始的使用指南)
- 添加使用示例代码片段

### L2: 性能优化 [1天]
- 使用多进程并行处理多视频
- 缓存Homography矩阵计算
- 优化内存占用 (大视频可能OOM)

### L3: 可视化增强 [1-2天]
- 绘制轨迹预测线
- 绘制TTC数值
- 绘制碰撞风险热力图

---

## 📅 建议实现时间表

```
Week 1 (Jan 6-12):
  ├─ H1: TTC计算器 [2-3天] → Monday-Wed
  ├─ H2: 事件分类器 [1-2天] → Wed-Thu
  └─ 集成测试 [1天] → Fri

Week 2 (Jan 13-19):
  ├─ H3: 轨迹预测升级 [3-4天] → Mon-Thu
  ├─ M2: Pipeline更新 [2天] → Thu-Fri
  └─ 测试和调试

Week 3 (Jan 20-26):
  ├─ M1: 多视图系统 [5-7天] → Full week
  └─ 或选择L1-L3完善

PPT审核 (2026-01-25):
  ├─ 演示H1-H3的完整功能
  └─ 展示测试结果
```

---

## ✅ 对标导师要求的验证清单

完成所有H1-H3后，应能展示：

- [x] "需要完成轨迹预测算法的实现" → H3完成
- [x] "基于Python进行拟合、预测和计算" → 整个系统都是Python
- [x] "需要考虑数据转换功能" → 已有(ObjectStateManager + coord_transform)
- [x] "建立一个基于ID的数据结构" → ObjectStateManager的track字典
- [x] "包含X、Y坐标加时间轴的三维数据结构" → (x, y, t)三元组
- [x] "设计一个专门的Python类来管理所有目标的状态" → ObjectStateManager类
- [x] "包含数据存储、索引查询、轨迹提取等功能" → update/get_trajectory/get_all_ids
- [x] "跟踪物体...从当前视图中消失或移动到其他视图" → M1的全局跟踪系统

---

## 关键指标与验证方式

| 指标 | 目标 | 验证方法 |
|------|------|---------|
| TTC精度 | ±5% | 对比标准视频 |
| 事件分类正确率 | >95% | 手工标注100帧验证 |
| 轨迹预测误差 | <10% | RMSE计算 |
| 多视图ID一致性 | >90% | 跨摄像头追踪测试 |
| 代码覆盖率 | >80% | pytest coverage report |

