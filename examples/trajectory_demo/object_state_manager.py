"""
object_state_manager.py

实现 ObjectStateManager: 按 track_id 管理目标轨迹、时间戳等信息。

支持两种坐标系：
  1. 像素坐标：直接从YOLO检测得到
  2. 世界坐标：使用Homography矩阵变换（需要标定）

【新增功能】
- 计算每个物体的三个接触点（前、中、后）
- 检测两个物体的9对接触点中距离最近的一对
- near-miss事件记录哪个点发生了碰撞接近

保存结构示例：
{
    track_id: [ {x, y, t, cls, conf, bbox, x_world, y_world, contact_points_pixel, contact_points_world}, ... ],
    ...
}

提供查询接口用于后续的轨迹预测 / TTC / near-miss 计算。
"""
from __future__ import annotations
from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np


def get_contact_points_from_bbox(bbox: Tuple[float, float, float, float], object_class: str = 'default') -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """
    从bbox计算物体的三个接触点（前、中、后）。
    
    接触点定义：贴近地面的底部边界上的三个点
    - 前点（front）：沿物体宽度方向的前侧（+x方向）
    - 中点（center）：物体宽度的中点
    - 后点（back）：沿物体宽度方向的后侧（-x方向）
    
    参数：
    - bbox: (x1, y1, x2, y2) YOLO格式的边界框
    - object_class: 物体类别 ('car', 'person', 'default')
              不同类别可能需要不同的偏移比例
    
    返回：
    - (front_point, center_point, back_point)
      每个点都是 (x, y) 的元组
    """
    x1, y1, x2, y2 = bbox
    bottom_y = y2  # bbox下边界（贴近地面处）
    center_x = (x1 + x2) / 2  # 水平中线
    
    # 物体宽度
    width = x2 - x1
    
    # 根据物体类别调整前后点的间隔
    if object_class in [2, 'car']:  # YOLO COCO中汽车的id是2
        # 汽车：前后点间隔 = 宽度的30%
        offset = width * 0.3
    elif object_class in [0, 'person']:  # YOLO COCO中人的id是0
        # 人：前后点间隔 = 宽度的20%（人比较窄）
        offset = width * 0.2
    else:
        # 默认：间隔 = 宽度的25%
        offset = width * 0.25
    
    front_point = (center_x + offset, bottom_y)
    center_point = (center_x, bottom_y)
    back_point = (center_x - offset, bottom_y)
    
    return (front_point, center_point, back_point)


class ObjectStateManager:
    def __init__(self, H: Optional[np.ndarray] = None):
        """初始化轨迹管理器
        
        参数：
        - H: Homography矩阵（可选），如果提供则进行像素->世界坐标变换
        """
        # key: track_id, value: list of samples ordered by time
        self.tracks: Dict[int, List[Dict[str, Any]]] = {}
        # optional: store last-seen frame index to detect disappearance
        self.last_seen: Dict[int, float] = {}
        # Homography矩阵（用于坐标变换）
        self.H = H
        self.use_world_coords = H is not None

    def update(self, detections: List[Dict[str, Any]], timestamp: float) -> None:
        """按帧更新轨迹数据。

        detections: 列表，每项包含至少 {id, cls, cx, cy, bbox, t}
        timestamp: 帧时间或帧号
        """
        for det in detections:
            tid = det.get('id')
            if tid is None:
                # 如果没有 id（predict 模式），可以选择跳过或为其分配临时 id
                continue
            
            cx = float(det.get('cx', det.get('x', 0.0)))
            cy = float(det.get('cy', det.get('y', 0.0)))
            bbox = det.get('bbox')  # 应该是 (x1, y1, x2, y2) 格式
            cls_id = det.get('cls')
            
            # 计算三个接触点（像素坐标）
            contact_points_pixel = None
            if bbox is not None:
                try:
                    contact_points_pixel = get_contact_points_from_bbox(bbox, cls_id)
                except Exception as e:
                    print(f"警告：计算接触点失败 {e}")
                    contact_points_pixel = None
            
            # 如果有Homography矩阵，进行坐标变换
            cx_world, cy_world = None, None
            contact_points_world = None
            if self.use_world_coords and self.H is not None:
                try:
                    import cv2
                    # perspectiveTransform需要 (N, 1, 2) 的格式
                    pixel_point = np.array([[[cx, cy]]], dtype=np.float32)
                    world_point = cv2.perspectiveTransform(pixel_point, self.H)
                    cx_world = float(world_point[0][0][0])
                    cy_world = float(world_point[0][0][1])
                    
                    # 也变换三个接触点
                    if contact_points_pixel is not None:
                        contact_points_pixel_arr = np.array(contact_points_pixel, dtype=np.float32).reshape(-1, 1, 2)
                        contact_points_world_arr = cv2.perspectiveTransform(contact_points_pixel_arr, self.H)
                        contact_points_world = [
                            (float(contact_points_world_arr[0][0][0]), float(contact_points_world_arr[0][0][1])),
                            (float(contact_points_world_arr[1][0][0]), float(contact_points_world_arr[1][0][1])),
                            (float(contact_points_world_arr[2][0][0]), float(contact_points_world_arr[2][0][1])),
                        ]
                except Exception as e:
                    print(f"警告：坐标变换失败 {e}，使用像素坐标")
                    cx_world, cy_world = cx, cy
                    contact_points_world = None
            
            sample = {
                'x': cx,  # 像素坐标
                'y': cy,
                'x_world': cx_world,  # 世界坐标（如果有Homography）
                'y_world': cy_world,
                't': float(timestamp),
                'cls': cls_id,
                'conf': det.get('conf'),
                'bbox': bbox,
                'contact_points_pixel': contact_points_pixel,  # 【新增】三个接触点（像素坐标）
                'contact_points_world': contact_points_world,  # 【新增】三个接触点（世界坐标）
            }
            if tid not in self.tracks:
                self.tracks[tid] = []
            self.tracks[tid].append(sample)
            self.last_seen[tid] = float(timestamp)

    def get_trajectory(self, track_id: int, last_n: Optional[int] = None) -> List[Dict[str, Any]]:
        """返回指定 id 的轨迹（按时间升序）。可选只返回最近 N 个点。"""
        traj = self.tracks.get(track_id, [])
        if last_n is not None:
            return traj[-last_n:]
        return list(traj)

    def get_all_ids(self) -> List[int]:
        return list(self.tracks.keys())

    def get_current_objects(self, since_time: Optional[float] = None) -> List[int]:
        """返回当前场景中被认为仍存在的对象 id。若指定 since_time,则仅返回最近一次出现时间 >= since_time 的对象。"""
        if since_time is None:
            return list(self.tracks.keys())
        return [tid for tid, t in self.last_seen.items() if t >= since_time]

    def distance_between(self, id1: int, id2: int, at_time: Optional[float] = None) -> Optional[float]:
        """计算 id1 与 id2 在给定时间的欧氏距离。如果 at_time 为 None，使用各自最新位置。
        
        如果有Homography矩阵，返回世界坐标距离（米）
        否则返回像素坐标距离（像素）
        """
        p1 = self._get_point_at(id1, at_time)
        p2 = self._get_point_at(id2, at_time)
        if p1 is None or p2 is None:
            return None
        
        # 优先使用世界坐标（如果有的话）
        if self.use_world_coords and p1.get('x_world') is not None and p2.get('x_world') is not None:
            dx = p1['x_world'] - p2['x_world']
            dy = p1['y_world'] - p2['y_world']
        else:
            # 降级到像素坐标
            dx = p1['x'] - p2['x']
            dy = p1['y'] - p2['y']
        
        return math.hypot(dx, dy)

    def distance_between_contact_points(self, id1: int, id2: int, at_time: Optional[float] = None) -> Optional[Tuple[float, Tuple[str, str], Tuple[float, float], Tuple[float, float]]]:
        """【新增】计算两个物体接触点间的距离，返回最近的一对。
        
        比较两个物体各自的三个接触点（前、中、后），共9对组合。
        返回距离最小的那一对。
        
        参数：
        - id1, id2: 物体ID
        - at_time: 时间戳（默认使用最新位置）
        
        返回：
        - (min_distance, (point_type1, point_type2), (x1, y1), (x2, y2))
          min_distance: 最近接触点的距离
          point_type1, point_type2: 点的类型（'front', 'center', 'back'）
          (x1, y1), (x2, y2): 两个点的坐标
        - 如果无法计算，返回 None
        """
        p1 = self._get_point_at(id1, at_time)
        p2 = self._get_point_at(id2, at_time)
        
        if p1 is None or p2 is None:
            return None
        
        # 获取接触点
        use_world = self.use_world_coords and p1.get('contact_points_world') is not None and p2.get('contact_points_world') is not None
        
        if use_world:
            points1 = p1['contact_points_world']
            points2 = p2['contact_points_world']
        else:
            points1 = p1['contact_points_pixel']
            points2 = p2['contact_points_pixel']
        
        if points1 is None or points2 is None:
            # 降级：如果没有接触点，用中心点
            return self._fallback_distance(id1, id2, at_time)
        
        # 点类型标签
        point_types = ['front', 'center', 'back']
        
        min_dist = float('inf')
        closest_pair = None
        
        # 遍历所有9对组合
        for i, pt1 in enumerate(points1):
            for j, pt2 in enumerate(points2):
                dist = math.hypot(pt1[0] - pt2[0], pt1[1] - pt2[1])
                if dist < min_dist:
                    min_dist = dist
                    closest_pair = (
                        dist,
                        (point_types[i], point_types[j]),  # 点的类型
                        pt1,  # 第一个物体的点坐标
                        pt2   # 第二个物体的点坐标
                    )
        
        return closest_pair

    def _fallback_distance(self, id1: int, id2: int, at_time: Optional[float] = None) -> Optional[float]:
        """【辅助函数】当无法计算接触点距离时，降级到中心点距离。"""
        p1 = self._get_point_at(id1, at_time)
        p2 = self._get_point_at(id2, at_time)
        if p1 is None or p2 is None:
            return None
        
        if self.use_world_coords and p1.get('x_world') is not None and p2.get('x_world') is not None:
            dx = p1['x_world'] - p2['x_world']
            dy = p1['y_world'] - p2['y_world']
        else:
            dx = p1['x'] - p2['x']
            dy = p1['y'] - p2['y']
        
        return math.hypot(dx, dy)

    def _get_point_at(self, track_id: int, at_time: Optional[float]) -> Optional[Dict[str, Any]]:
        traj = self.tracks.get(track_id)
        if not traj:
            return None
        if at_time is None:
            return traj[-1]
        # 简单策略：返回最后一个时间 <= at_time
        for sample in reversed(traj):
            if sample['t'] <= at_time:
                return sample
        return None

    def approximate_velocity(self, track_id: int, last_n: int = 2) -> Optional[Dict[str, float]]:
        """用最后 last_n 个点估计速度(单位: units / time)。返回 {'vx','vy'} 或 None。"""
        traj = self.tracks.get(track_id, [])
        if len(traj) < 2:
            return None
        pts = traj[-last_n:]
        p0 = pts[0]
        p1 = pts[-1]
        dt = p1['t'] - p0['t']
        if dt == 0:
            return None
        vx = (p1['x'] - p0['x']) / dt
        vy = (p1['y'] - p0['y']) / dt
        return {'vx': vx, 'vy': vy}

    def ttc_between(self, id1: int, id2: int) -> Optional[float]:
        """估算两个目标的 Time-To-Collision(非常粗略的近似)。

        方法：取两个目标的最新位置和速度，解沿连线方向的相对速度。
        若相对速度朝向彼此且距离 / 相对速度 > 0,则返回 ttc(秒)，否则返回 None。
        """
        p1 = self._get_point_at(id1, None)
        p2 = self._get_point_at(id2, None)
        if p1 is None or p2 is None:
            return None
        v1 = self.approximate_velocity(id1)
        v2 = self.approximate_velocity(id2)
        if v1 is None or v2 is None:
            return None
        dx = p2['x'] - p1['x']
        dy = p2['y'] - p1['y']
        dist = math.hypot(dx, dy)
        # 相对速度向量 v_rel = v2 - v1
        vx_rel = v2['vx'] - v1['vx']
        vy_rel = v2['vy'] - v1['vy']
        # 相对速度在连线方向的分量（正值表示距离在增大）
        if dist == 0:
            return 0.0
        # 投影
        rel_along = (vx_rel * dx + vy_rel * dy) / dist
        # 如果 rel_along < 0 表示彼此靠近（距离在减小）
        if rel_along >= 0:
            return None
        ttc = dist / (-rel_along)
        return float(ttc)

    def detect_near_miss(self, distance_threshold: float = 100.0, 
                         ttc_threshold: float = 2.0) -> List[Dict[str, Any]]:
        """【改进】检测可能的 near-miss 事件（两个目标靠得很近）。
        
        现在使用接触点距离而不是中心距离。
        
        参数
        - distance_threshold: 距离阈值（像素/米），小于此值认为靠近
        - ttc_threshold: TTC 阈值（秒），小于此值认为将发生碰撞
        
        返回
        - List[{id1, id2, distance, closest_point_pair, ttc, timestamp, is_collision_risk}]
        """
        near_misses = []
        ids = self.get_all_ids()
        
        # 两两比对
        for i, id1 in enumerate(ids):
            for id2 in ids[i+1:]:
                # 【改进】使用接触点距离而不是中心距离
                contact_result = self.distance_between_contact_points(id1, id2)
                
                if contact_result is None:
                    continue
                
                dist, point_types, pt1, pt2 = contact_result
                
                if dist is not None and dist < distance_threshold:
                    ttc = self.ttc_between(id1, id2)
                    # 记录靠近或有碰撞风险的事件
                    p1 = self._get_point_at(id1, None)
                    timestamp = p1['t'] if p1 else 0
                    event = {
                        'id1': id1,
                        'id2': id2,
                        'distance': float(dist),
                        'closest_point_pair': {  # 【新增】记录接触点信息
                            'obj1_point_type': point_types[0],  # 'front', 'center', 或 'back'
                            'obj2_point_type': point_types[1],
                            'obj1_coords': [float(pt1[0]), float(pt1[1])],  # 坐标
                            'obj2_coords': [float(pt2[0]), float(pt2[1])],
                        },
                        'ttc': float(ttc) if ttc is not None else None,
                        'timestamp': float(timestamp),
                        'is_collision_risk': ttc is not None and ttc < ttc_threshold
                    }
                    near_misses.append(event)
        
        return near_misses

    def save_tracks(self, path: str) -> None:
        import json, os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.tracks, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    # 简单自测
    osm = ObjectStateManager()
    osm.update([{'id': 1, 'cx': 0, 'cy': 0, 't': 0}, {'id': 2, 'cx': 10, 'cy': 0, 't': 0}], 0)
    osm.update([{'id': 1, 'cx': 1, 'cy': 0, 't': 1}, {'id': 2, 'cx': 9, 'cy': 0, 't': 1}], 1)
    print('ids', osm.get_all_ids())
    print('dist', osm.distance_between(1,2))
    print('ttc', osm.ttc_between(1,2))
