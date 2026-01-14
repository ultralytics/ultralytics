"""
detection_adapter.py.

把 Ultralytics 的 `result` 解析成统一格式的检测列表:
[ {"id":..., "cls":..., "x":..., "y":..., "t":..., "conf":..., "bbox": [...]}, ... ]

字段命名与 ObjectStateManager 一致。
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 延迟导入 coord_transform，以避免循环依赖


def parse_result(result, timestamp: float) -> list[dict[str, Any]]:
    """从一个 Ultralytics `result` 对象解析出检测项列表。.

    参数
    - result: 单帧的 Results 对象（来自 model.predict/track 的迭代项）
    - timestamp: 帧的时间戳或帧号

    返回
    - detections: 列表，每项为 dict: {id, cls, x, y, t, conf, bbox, mask}
    """
    # Boxes 对象字段可能在不同版本中略有差异，使用 getattr 兜底
    boxes = getattr(result, "boxes", None)
    masks = getattr(result, "masks", None)  # 分割掩码（如果有）

    if boxes is None:
        return []

    xyxy = getattr(boxes, "xyxy", None)
    cls = getattr(boxes, "cls", None)
    conf = getattr(boxes, "conf", None)
    ids = getattr(boxes, "id", None)
    if ids is None:
        ids = getattr(boxes, "ids", None)

    # 转为 numpy（若是 tensor）
    if xyxy is None:
        return []

    try:
        xyxy_np = xyxy.cpu().numpy()
    except Exception:
        xyxy_np = np.asarray(xyxy)

    try:
        cls_np = cls.cpu().numpy() if cls is not None else None
    except Exception:
        cls_np = np.asarray(cls) if cls is not None else None

    try:
        conf_np = conf.cpu().numpy() if conf is not None else None
    except Exception:
        conf_np = np.asarray(conf) if conf is not None else None

    try:
        ids_np = ids.cpu().numpy() if ids is not None else None
    except Exception:
        ids_np = np.asarray(ids) if ids is not None else None

    # 尝试获取掩码数据
    masks_data = None
    if masks is not None:
        try:
            masks_data = masks.masks.cpu().numpy()  # shape: (N, H, W)
        except Exception:
            try:
                masks_data = masks.cpu().numpy()
            except Exception:
                masks_data = None

    dets = []
    for i, box in enumerate(xyxy_np):
        x1, y1, x2, y2 = box.tolist()
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)

        # 如果有分割掩码，计算掩码中心而不是bbox中心
        if masks_data is not None and i < len(masks_data):
            mask = masks_data[i]
            # 找到掩码中的前景像素
            y_coords, x_coords = np.where(mask > 0)
            if len(x_coords) > 0:
                cx = float(np.mean(x_coords))
                cy = float(np.mean(y_coords))
                # 计算掩码的最小外接矩形
                _x_min, _x_max = int(np.min(x_coords)), int(np.max(x_coords))
                _y_min, _y_max = int(np.min(y_coords)), int(np.max(y_coords))

        det = {
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "cx": cx,
            "cy": cy,
            "t": timestamp,
            "cls": int(cls_np[i]) if cls_np is not None else None,
            "conf": float(conf_np[i]) if conf_np is not None else None,
            "id": int(ids_np[i]) if ids_np is not None else None,
            "has_mask": masks_data is not None and i < len(masks_data),  # 标记是否用了掩码
        }
        dets.append(det)

    return dets
