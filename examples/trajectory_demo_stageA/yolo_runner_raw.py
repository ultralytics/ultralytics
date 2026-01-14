"""
yolo_runner_raw.py.

阶段 A：只运行 YOLO 并保存原始检测输出（JSONL），不做 adapter / state manager / 坐标转换。

用途：
- 快速检查 YOLO 的原始输出字段（boxes.xyxy, boxes.cls, boxes.conf, boxes.id/ids 等）
- 为后续实现 detection_adapter 提供样本数据

输出：`<output>/raw_detections.jsonl`，每行是一个 frame 的 JSON：
{ "frame": int, "t": int, "detections": [ {bbox:[x1,y1,x2,y2], cx, cy, cls, conf, id}, ... ] }

用法示例：
python examples/trajectory_demo_stageA/yolo_runner_raw.py --source "/path/to/video.mp4" --weights yolo11n.pt --output runs/trajectory_demo_stageA

注意：
- 如果在容器/VM 中运行，请把视频文件复制或挂载到容器可见路径（见 README 说明）。
- 该脚本只保存原始检测，不会修改仓库其他文件。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

# allow running from repo root
sys.path.append(os.path.dirname(__file__))

from ultralytics import YOLO


def tensor_to_list(x: Any):
    """尝试把 torch tensor 转为 list。若失败，返回 None 或原始对象的 list 表示。."""
    if x is None:
        return None
    try:
        return x.cpu().numpy().tolist()
    except Exception:
        try:
            return list(x)
        except Exception:
            return None


def run_raw(source: str, weights: str, output: str):
    os.makedirs(output, exist_ok=True)
    out_file = os.path.join(output, "raw_detections.jsonl")
    model = YOLO(weights)
    print(f"Running raw YOLO on: {source}")
    print(f"Weights: {weights}")
    print(f"Output JSONL: {out_file}")

    with open(out_file, "w", encoding="utf-8") as f:
        for frame_idx, result in enumerate(model.track(source=source, stream=True, persist=True)):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                out = {"frame": frame_idx, "t": frame_idx, "detections": []}
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
                print(f"frame {frame_idx}: no boxes")
                continue

            xyxy = tensor_to_list(getattr(boxes, "xyxy", None))
            cls = tensor_to_list(getattr(boxes, "cls", None))
            conf = tensor_to_list(getattr(boxes, "conf", None)) or tensor_to_list(getattr(boxes, "confidence", None))
            ids = tensor_to_list(getattr(boxes, "id", None)) or tensor_to_list(getattr(boxes, "ids", None))

            dets = []
            if xyxy is None:
                dets = []
            else:
                for i, box in enumerate(xyxy):
                    try:
                        x1, y1, x2, y2 = box
                    except Exception:
                        # sometimes xyxy may be nested lists
                        x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    det = {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "cx": float(cx),
                        "cy": float(cy),
                        "cls": int(cls[i]) if cls is not None and i < len(cls) else None,
                        "conf": float(conf[i]) if conf is not None and i < len(conf) else None,
                        "id": int(ids[i]) if ids is not None and i < len(ids) else None,
                    }
                    dets.append(det)

            out = {"frame": frame_idx, "t": frame_idx, "detections": dets}
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            print(f"frame {frame_idx}: {len(dets)} detections")

    print(f"Saved raw detections to {out_file}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True, help="video file or camera")
    p.add_argument("--weights", type=str, default="yolo11n.pt")
    p.add_argument("--output", type=str, default="runs/trajectory_demo_stageA")
    args = p.parse_args()
    run_raw(args.source, args.weights, args.output)
