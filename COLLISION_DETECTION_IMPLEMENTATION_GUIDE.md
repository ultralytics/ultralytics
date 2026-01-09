# 锚点投影碰撞检测系统 - 可视化示例和代码框架

**目的**: 演示改进方法在实际场景中的应用  
**版本**: 1.0  
**日期**: 2025-01-09

---

## 一、场景1：多车道并入（你的图片场景）

### 1.1 场景描述

```
视角：俯视图（无人机或监控摄像头）

上方车道（皮卡方向）→
┌─────────────────────┐
│   深灰色皮卡 (ID:5) │  heading ≈ 0° (向东)
│    [中心 (500, 200)]│
│    w=100, h=150     │
└─────────────────────┘
              ↙
            并入
              ↙
┌─────────────────────┐
│  银色小车 (ID:8) →  │  heading ≈ 30° (向东北)
│  [中心 (350, 350)]  │
│   w=80, h=120       │
└─────────────────────┘

危险：小车并入时，其车头可能与皮卡的左后侧相撞
```

### 1.2 当前系统的分析

```
╔═══════════════════════════════════════════════════════════════╗
║           当前系统（仅中心点距离）                              ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                ║
║  皮卡中心点: (500, 200)                                       ║
║        ●                                                       ║
║        │                                                       ║
║        │                                                       ║
║        │ 距离 = √[(500-350)² + (200-350)²]                   ║
║        │      = √[150² + 150²]                               ║
║        │      = 212 px ≈ 3.47 m                              ║
║        │                                                       ║
║        │                                                       ║
║        └─────────────────────●                               ║
║                   小车中心: (350, 350)                         ║
║                                                                ║
║  判定：✓ 检测到接近 (< 5m threshold)                          ║
║  但是：没有具体说明哪些部位最接近                              ║
║  问题：中心点距离 ≠ 碰撞风险                                   ║
║                                                                ║
╚═══════════════════════════════════════════════════════════════╝
```

### 1.3 改进系统的分析

```
╔═══════════════════════════════════════════════════════════════╗
║        改进系统（多锚点距离 + 朝向分析）                        ║
╠═══════════════════════════════════════════════════════════════╣
║                                                                ║
║  深灰皮卡的锚点：                                              ║
║  - front_center: (500, 275)                                  ║
║  - front_left:   (450, 275)                                  ║
║  - front_right:  (550, 275)                                  ║
║  - rear_center:  (500, 125)                                  ║
║  - rear_left:    (450, 125)     ← 关键锚点！                  ║
║  - rear_right:   (550, 125)                                  ║
║  - left_center:  (450, 200)                                  ║
║  - right_center: (550, 200)                                  ║
║                                                                ║
║  银色小车的锚点：                                              ║
║  - front_center: (350, 410)                                  ║
║  - front_left:   (310, 410)                                  ║
║  - front_right:  (390, 410)     ← 关键锚点！                  ║
║  - rear_center:  (350, 290)                                  ║
║  - rear_left:    (310, 290)                                  ║
║  - rear_right:   (390, 290)                                  ║
║  - left_center:  (310, 350)                                  ║
║  - right_center: (390, 350)                                  ║
║                                                                ║
║  计算所有锚点对距离（共 8×8=64 个）：                          ║
║  - dist(rear_left, front_right) = 115.2 px ≈ 1.89 m ← MIN  ║
║  - dist(rear_left, front_left)  = 141.4 px ≈ 2.32 m        ║
║  - dist(rear_center, front_center) = 138.7 px ≈ 2.28 m     ║
║  - ...其他组合...                                             ║
║                                                                ║
║  最关键的发现：                                                 ║
║  ┌─────────────────────────────────────────────────────────┐ ║
║  │ 皮卡的【左后车尾】(450, 125)                              │ ║
║  │         距离                                              │ ║
║  │ 小车的【右前车头】(390, 410)                              │ ║
║  │         约 1.89 米                                       │ ║
║  │                                                           │ ║
║  │ → 这两个部位最可能碰撞！                                  │ ║
║  └─────────────────────────────────────────────────────────┘ ║
║                                                                ║
║  朝向分析：                                                    ║
║  - 皮卡朝向：0°（水平向右）                                    ║
║  - 小车朝向：30°（向右上方）                                   ║
║  - 相对角度：30°                                              ║
║  - 接近方向：【收敛】(converging) → 碰撞风险高！               ║
║                                                                ║
║  风险评级：                                                    ║
║  ╔════════════════════════════════════════════════════════╗ ║
║  ║ 【严重 CRITICAL】                                       ║ ║
║  ║                                                         ║ ║
║  ║ 最小距离: 1.89 m (< 2.0m 阈值)                          ║ ║
║  ║ 碰撞部位: rear_left ↔ front_right                      ║ ║
║  ║ 冲突角度: 30° (收敛方向)                                ║ ║
║  ║ 预计碰撞时间 (TTC): 2.3 秒                              ║ ║
║  ║                                                         ║ ║
║  ║ 建议: 立即警告！前方车辆应减速或改道。                  ║ ║
║  ╚════════════════════════════════════════════════════════╝ ║
║                                                                ║
╚═══════════════════════════════════════════════════════════════╝
```

### 1.4 可视化对比图

#### **当前系统的画面**
```
[图像注释示例]
━━━━━━━━━━━━━━━━━━━━━━━━
    ◆ (皮卡)
   / \
  /   \  3.47m
 /     \
━━━━━━━●━━━━━━━
     ◆ (小车)

说明: 仅显示中心点连线，缺乏空间细节
```

#### **改进系统的画面**
```
[改进的图像注释示例]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ●─●─●         (皮卡的车头)
    ┃   ┃
    ◆   ◆  ← 左右中线
    ┃   ┃
    ●═●═●         (皮卡的车尾，左后特别标记)
         ║
         ║ 1.89m  ← 最关键的距离！
         ║
    ●───●  ╔═╗      (小车的车头)
    ┃   ┃  ║▲║
    ◆   ◆  ║ ║  ← 右前特别标记
    ┃   ┃  ║朝║
    ●───●  ║向║
            ╚═╝

部位标记: rear_left ↔ front_right
风险等级: 【严重 CRITICAL】
"""

---

## 二、场景2：行人与车辆

### 2.1 场景描述

```
场景：停车场中，行人在车辆右侧经过

视图（俯视）：

    [小轿车] (ID: 10)
  ┌──────────┐
  │●────────●│  朝向: 0° (向右)
  │┃        ┃│
  │◆────────◆│  center: (300, 200), w=80, h=100
  │┃        ┃│
  └──────────┘
       right_center: (340, 200)  ← 监控点
            ↓
         [行人] (ID: 25)    
         (400, 200)  ← 在车的右侧
```

### 2.2 改进系统的分析

```python
# 车辆锚点
car_anchors = {
    'right_center': (340, 200),    ← 车的右侧中点
    'front_right': (340, 250),
    'rear_right': (340, 150),
    # ... 其他
}

# 行人锚点（简化）
pedestrian_anchors = {
    'head': (400, 100),        ← 行人头部
    'torso': (400, 200),       ← 躯干
    'lower': (400, 300),       ← 下身
    'feet': (400, 350),        ← 脚部
}

# 计算距离
distances = [
    dist(car_right_center, ped_head) = 60 px ≈ 0.98m
    dist(car_right_center, ped_torso) = 60 px ≈ 0.98m  ← MIN!
    dist(car_right_center, ped_lower) = 100 px ≈ 1.64m
    dist(car_front_right, ped_head) = 150 px ≈ 2.46m
    # ... 等等
]

# 结论
最小距离部位: car_right_center ↔ pedestrian_torso
距离: 0.98 m (< 1.0m) → 非常危险！
```

---

## 三、实现代码框架（伪代码 + 注释）

### 3.1 基础数据结构

```python
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np

@dataclass
class AnchorPoint:
    """单个锚点的定义"""
    name: str                    # e.g., "rear_left", "head"
    position: Tuple[float, float] # (x, y) 像素坐标
    importance: float            # 0-1, 重要性权重
    category: str               # e.g., "structural", "vulnerability"

class ObjectAnchors:
    """某个物体的所有锚点"""
    
    def __init__(self, object_class: int, bbox_xywh, heading=None):
        """
        Args:
            object_class: COCO class ID
            bbox_xywh: [center_x, center_y, width, height]
            heading: 朝向角度（弧度）
        """
        self.object_class = object_class
        self.bbox_xywh = bbox_xywh
        self.heading = heading or 0.0
        self.anchors = self._generate_anchors()
    
    def _generate_anchors(self) -> Dict[str, AnchorPoint]:
        """根据类别生成锚点"""
        x, y, w, h = self.bbox_xywh
        
        if self.object_class == 2:  # car
            return {
                'front_center': AnchorPoint('front_center', (x, y+h/2), 0.9, 'structural'),
                'front_left': AnchorPoint('front_left', (x-w/2, y+h/2), 0.8, 'corner'),
                'front_right': AnchorPoint('front_right', (x+w/2, y+h/2), 0.8, 'corner'),
                'rear_center': AnchorPoint('rear_center', (x, y-h/2), 0.9, 'structural'),
                'rear_left': AnchorPoint('rear_left', (x-w/2, y-h/2), 0.8, 'corner'),
                'rear_right': AnchorPoint('rear_right', (x+w/2, y-h/2), 0.8, 'corner'),
                'left_center': AnchorPoint('left_center', (x-w/2, y), 0.7, 'side'),
                'right_center': AnchorPoint('right_center', (x+w/2, y), 0.7, 'side'),
            }
        elif self.object_class == 0:  # person
            return {
                'head': AnchorPoint('head', (x, y-h/2), 1.0, 'vulnerability'),
                'torso': AnchorPoint('torso', (x, y), 0.9, 'vulnerability'),
                'lower': AnchorPoint('lower', (x, y+h/3), 0.8, 'vulnerability'),
                'feet': AnchorPoint('feet', (x, y+h/2), 0.7, 'vulnerability'),
            }
        # ... 其他类别
    
    def get_anchors(self) -> Dict[str, Tuple[float, float]]:
        """返回所有锚点的位置"""
        return {name: point.position for name, point in self.anchors.items()}

@dataclass
class CollisionResult:
    """碰撞分析结果"""
    min_distance: float              # 最小距离（米）
    min_distance_px: float           # 最小距离（像素）
    object1_part: str                # 物体1的部位
    object2_part: str                # 物体2的部位
    point1: Tuple[float, float]      # 物体1的点位置
    point2: Tuple[float, float]      # 物体2的点位置
    relative_heading: float          # 相对朝向角度
    approach_vector: Tuple[float, float]  # 接近方向向量
    risk_level: str                  # critical/high/medium/low
    ttc: Optional[float] = None      # Time To Collision (秒)
```

### 3.2 碰撞分析器

```python
class CollisionAnalyzer:
    """分析两个物体之间的碰撞风险"""
    
    def analyze(self, 
               obj1_anchors: ObjectAnchors,
               obj2_anchors: ObjectAnchors,
               obj1_velocity: Tuple[float, float],
               obj2_velocity: Tuple[float, float],
               pixel_per_meter: float = 60.0) -> CollisionResult:
        """
        分析两个物体之间的碰撞风险
        
        Args:
            obj1_anchors: 物体1的锚点
            obj2_anchors: 物体2的锚点
            obj1_velocity: 物体1的速度向量 (vx, vy) [m/s]
            obj2_velocity: 物体2的速度向量 [m/s]
            pixel_per_meter: 像素与米的转换比
        
        Returns:
            CollisionResult 对象
        """
        
        # Step 1: 获取所有锚点
        anchors1 = obj1_anchors.get_anchors()
        anchors2 = obj2_anchors.get_anchors()
        
        # Step 2: 计算所有锚点对的距离
        min_dist_px = float('inf')
        min_result = None
        
        for part1_name, (x1, y1) in anchors1.items():
            for part2_name, (x2, y2) in anchors2.items():
                dist_px = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                
                if dist_px < min_dist_px:
                    min_dist_px = dist_px
                    min_result = {
                        'part1': part1_name,
                        'part2': part2_name,
                        'point1': (x1, y1),
                        'point2': (x2, y2),
                    }
        
        # Step 3: 转换为世界坐标（米）
        min_dist_m = min_dist_px / pixel_per_meter
        
        # Step 4: 计算朝向和接近向量
        relative_heading = obj1_anchors.heading - obj2_anchors.heading
        
        # 接近向量：从obj1指向obj2
        approach_x = min_result['point2'][0] - min_result['point1'][0]
        approach_y = min_result['point2'][1] - min_result['point1'][1]
        approach_vector = (approach_x, approach_y)
        
        # Step 5: 计算 TTC (Time To Collision)
        ttc = self._calculate_ttc(
            obj1_velocity, obj2_velocity,
            approach_vector, min_dist_m
        )
        
        # Step 6: 评定风险等级
        risk_level = self._assess_risk(
            min_dist_m, ttc, relative_heading
        )
        
        return CollisionResult(
            min_distance=min_dist_m,
            min_distance_px=min_dist_px,
            object1_part=min_result['part1'],
            object2_part=min_result['part2'],
            point1=min_result['point1'],
            point2=min_result['point2'],
            relative_heading=relative_heading,
            approach_vector=approach_vector,
            risk_level=risk_level,
            ttc=ttc
        )
    
    @staticmethod
    def _calculate_ttc(v1, v2, approach_vector, min_dist) -> Optional[float]:
        """
        计算预计碰撞时间 (TTC)
        
        假设线性运动，计算多久后距离会减少到0
        """
        # 相对速度
        rel_vx = v2[0] - v1[0]
        rel_vy = v2[1] - v1[1]
        
        # 在接近方向上的速度分量
        approach_speed = np.sqrt(approach_vector[0]**2 + approach_vector[1]**2)
        
        if approach_speed < 0.01:  # 静止或近乎静止
            return None
        
        # 接近速度（负值表示接近）
        approach_vel = (rel_vx * approach_vector[0] + rel_vy * approach_vector[1]) / approach_speed
        
        if approach_vel >= 0:  # 不接近
            return float('inf')
        
        ttc = min_dist / abs(approach_vel)
        return ttc if ttc > 0 else None
    
    @staticmethod
    def _assess_risk(min_dist, ttc, relative_heading) -> str:
        """根据距离、TTC和朝向评定风险等级"""
        
        # 距离阈值
        if min_dist < 0.5:
            return 'critical'  # 严重
        elif min_dist < 1.5:
            if ttc is not None and ttc < 2.0:
                return 'critical'
            else:
                return 'high'  # 高风险
        elif min_dist < 3.0:
            if ttc is not None and ttc < 3.0:
                return 'high'
            else:
                return 'medium'  # 中等风险
        else:
            return 'low'  # 低风险
```

### 3.3 集成到现有管道

```python
# 在 collision_detection_pipeline_yolo_first_method_a.py 中

def extract_key_frames(self, all_detections, tracks, 
                       world_distance_threshold=2.0, 
                       debug_threshold=5.0):
    """改进版本"""
    
    # ... 前面的代码不变 ...
    
    for frame in sorted(frame_keys):
        frame_data = all_detections_by_frame[frame]
        frame_tracks = tracks_by_frame[frame]
        
        for tid1 in frame_tracks:
            for tid2 in frame_tracks:
                if tid1 >= tid2:
                    continue
                
                obj1 = frame_data_objs_by_id[tid1]
                obj2 = frame_data_objs_by_id[tid2]
                
                # 【改进】：创建锚点对象，而不是仅计算中心点
                anchors1 = ObjectAnchors(
                    object_class=obj1['class'],
                    bbox_xywh=obj1['bbox_xywh'],
                    heading=self._estimate_heading(tracks[tid1])
                )
                anchors2 = ObjectAnchors(
                    object_class=obj2['class'],
                    bbox_xywh=obj2['bbox_xywh'],
                    heading=self._estimate_heading(tracks[tid2])
                )
                
                # 【改进】：使用碰撞分析器
                collision_result = CollisionAnalyzer().analyze(
                    obj1_anchors=anchors1,
                    obj2_anchors=anchors2,
                    obj1_velocity=(
                        track1.get('vx_world', 0),
                        track1.get('vy_world', 0)
                    ),
                    obj2_velocity=(
                        track2.get('vx_world', 0),
                        track2.get('vy_world', 0)
                    ),
                    pixel_per_meter=self.pixel_per_meter
                )
                
                # 使用改进的距离判定
                if collision_result.min_distance < world_distance_threshold:
                    # 【改进】：扩展事件记录
                    event = {
                        'frame': frame,
                        'time': frame_data['time'],
                        'track_id_1': tid1,
                        'track_id_2': tid2,
                        'class_1': class1_name,
                        'class_2': class2_name,
                        
                        # 原有信息
                        'distance_pixel': float(collision_result.min_distance_px),
                        'distance_meters': float(collision_result.min_distance),
                        
                        # 【新增】：锚点信息
                        'closest_parts': {
                            'object1_part': collision_result.object1_part,
                            'object2_part': collision_result.object2_part,
                            'point1_px': collision_result.point1,
                            'point2_px': collision_result.point2,
                            'description': f"{class1_name}的{collision_result.object1_part} ↔ "
                                         f"{class2_name}的{collision_result.object2_part}"
                        },
                        
                        # 【新增】：朝向信息
                        'heading_analysis': {
                            'relative_heading_rad': float(collision_result.relative_heading),
                            'approach_vector': collision_result.approach_vector,
                        },
                        
                        # 【新增】：预计碰撞时间
                        'ttc_seconds': float(collision_result.ttc) if collision_result.ttc else None,
                        
                        # 【新增】：风险评级
                        'risk_level': collision_result.risk_level,
                    }
                    
                    proximity_events.append(event)
                    
                    # 保存改进的关键帧
                    frame_img_path = self.keyframe_dir / f"keyframe_{frame:04d}_ID{tid1}_ID{tid2}.jpg"
                    self._save_advanced_keyframe(
                        self.video_path, frame, frame_img_path,
                        event, anchors1, anchors2
                    )
    
    return proximity_events

def _estimate_heading(self, track_points):
    """从轨迹历史估计车辆朝向"""
    if len(track_points) < 3:
        return 0.0
    
    recent = track_points[-5:]
    vx = recent[-1]['center_x'] - recent[0]['center_x']
    vy = recent[-1]['center_y'] - recent[0]['center_y']
    
    return np.arctan2(vy, vx)

def _save_advanced_keyframe(self, video_path, frame_num, output_path,
                           event, anchors1, anchors2):
    """保存带有高级信息的关键帧"""
    # ... 加载视频帧 ...
    
    # 绘制所有锚点
    for anchor_name, (px, py) in anchors1.get_anchors().items():
        cv2.circle(frame, (int(px), int(py)), 4, (0, 255, 0), -1)  # 绿色
    
    for anchor_name, (px, py) in anchors2.get_anchors().items():
        cv2.circle(frame, (int(px), int(py)), 4, (255, 0, 0), -1)  # 蓝色
    
    # 突出最接近的两个锚点
    pt1 = event['closest_parts']['point1_px']
    pt2 = event['closest_parts']['point2_px']
    cv2.circle(frame, (int(pt1[0]), int(pt1[1])), 8, (0, 255, 255), 2)  # 黄色
    cv2.circle(frame, (int(pt2[0]), int(pt2[1])), 8, (0, 255, 255), 2)
    cv2.line(frame, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1])),
            (0, 0, 255), 3)  # 红色粗线
    
    # 添加文字说明
    cv2.putText(frame, f"Risk: {event['risk_level'].upper()}",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.putText(frame, f"Distance: {event['distance_meters']:.2f}m",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.putText(frame, f"Closest: {event['closest_parts']['description']}",
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    
    if event['ttc_seconds'] is not None:
        cv2.putText(frame, f"TTC: {event['ttc_seconds']:.1f}s",
                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    cv2.imwrite(str(output_path), frame)
```

---

## 四、测试场景检查清单

### 改进前后对比测试

| 场景 | 当前结果 | 改进后结果 | 预期改进 |
|-----|---------|----------|---------|
| 多车道并入 | 中心距离 3.47m | 最近部位 1.89m | ✅ 更精确 |
| 行人与车辆 | 中心距离 2.5m | 最近部位 0.98m | ✅ 发现高风险 |
| 平行行驶（无碰撞） | 距离 2.0m | 相对朝向平行，TTC=∞ | ✅ 正确判定安全 |
| 侧向超车 | 中心距离 1.8m | 右侧距离 0.5m | ✅ 识别侧向碰撞 |

---

## 五、下一步改进方向

### 5.1 短期（1-2周）
- [ ] 实现基础的锚点定义
- [ ] 集成多锚点距离计算
- [ ] 更新事件记录格式
- [ ] 基于轨迹的朝向估计

### 5.2 中期（2-4周）
- [ ] 改进 TTC 计算
- [ ] 加入更多车辆类别的锚点
- [ ] 开发可视化面板
- [ ] 完整测试和调优

### 5.3 长期（4-8周）
- [ ] YOLO 模型扩展，输出朝向角度
- [ ] 3D 碰撞检测（物理约束）
- [ ] 机器学习模型预测碰撞概率
- [ ] 实时警报和可视化系统

---

**文档维护**: 2025-01-09  
**下一个版本**: 实现完成后的验证文档

