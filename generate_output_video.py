#!/usr/bin/env python3
"""
Generate output video with YOLO detections and multi-anchor keyframe overlays.
Combines YOLO detection frames with multi-anchor collision detection visualizations.
"""

import json
from pathlib import Path

import cv2


def generate_output_video(video_path, result_dir, output_path=None):
    """Generate output video combining YOLO detections with multi-anchor keyframes.

    Args:
        video_path: Path to input video
        result_dir: Path to results directory (from pipeline)
        output_path: Path to save output video (optional)
    """
    result_dir = Path(result_dir)

    # Default output path
    if output_path is None:
        output_path = result_dir / "output_video.mp4"
    else:
        output_path = Path(output_path)

    print(f"\n{'=' * 70}")
    print("【生成输出视频：YOLO检测 + 多锚点关键帧】")
    print(f"{'=' * 70}")

    # Load keyframe data
    keyframes_json = result_dir / "3_key_frames" / "proximity_events.json"
    keyframes_dir = result_dir / "3_key_frames"

    if not keyframes_json.exists():
        print(f"❌ 未找到关键帧JSON: {keyframes_json}")
        return

    with open(keyframes_json) as f:
        keyframe_events = json.load(f)

    # Build keyframe map: frame_num -> (tid1, tid2)
    keyframe_map = {}
    for event in keyframe_events:
        frame_num = event["frame"]
        tid1 = event["track_id_1"]
        tid2 = event["track_id_2"]
        keyframe_map[frame_num] = (tid1, tid2)

    print(f"✓ 加载 {len(keyframe_map)} 个关键帧")

    # Load detection frames
    detection_dir = result_dir / "1_yolo_detection"
    detection_json = detection_dir / "detections_pixel.json"

    if not detection_json.exists():
        print(f"❌ 未找到检测JSON: {detection_json}")
        return

    with open(detection_json) as f:
        all_detections = json.load(f)

    # Build detection map: frame_num -> detection_data
    detection_map = {d["frame"]: d for d in all_detections}
    print(f"✓ 加载 {len(detection_map)} 帧YOLO检测")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"✓ 视频: {frame_width}x{frame_height}, {fps:.1f}fps, {total_frames}帧")

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    if not out.isOpened():
        print(f"❌ 无法创建视频输出: {output_path}")
        return

    # Process frames
    frame_idx = 0
    keyframes_used = 0
    detections_drawn = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # If this is a keyframe, use the keyframe image instead
        if frame_idx in keyframe_map:
            tid1, tid2 = keyframe_map[frame_idx]
            keyframe_path = keyframes_dir / f"keyframe_{frame_idx:04d}_ID{tid1}_ID{tid2}.jpg"

            if keyframe_path.exists():
                frame = cv2.imread(str(keyframe_path))
                keyframes_used += 1
        else:
            # Draw YOLO detections for non-keyframe frames
            if frame_idx in detection_map:
                detection_data = detection_map[frame_idx]
                for obj in detection_data.get("objects", []):
                    x, y, w, h = obj["bbox_xywh"]
                    x1, y1 = int(x - w / 2), int(y - h / 2)
                    x2, y2 = int(x + w / 2), int(y + h / 2)
                    tid = obj.get("track_id", -1)
                    cls = obj.get("class_name", "unknown")

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Draw label
                    label = f"ID{tid} {cls}"
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    detections_drawn += 1

        # Write frame
        out.write(frame)

        frame_idx += 1
        if frame_idx % 20 == 0:
            print(f"  处理进度: {frame_idx}/{total_frames} 帧", end="\r")

    cap.release()
    out.release()

    print("\n✓ 视频生成完成!")
    print(f"  - 总帧数: {frame_idx}")
    print(f"  - 关键帧覆盖: {keyframes_used}")
    print(f"  - YOLO检测: {detections_drawn}")
    print(f"  - 输出: {output_path}")

    # Print file size
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)
        print(f"  - 文件大小: {file_size:.1f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate output video with YOLO detections and multi-anchor keyframes"
    )
    parser.add_argument("--video", type=str, required=True, help="Input video path")
    parser.add_argument("--result-dir", type=str, required=True, help="Results directory from pipeline")
    parser.add_argument("--output", type=str, default=None, help="Output video path (optional)")

    args = parser.parse_args()

    generate_output_video(video_path=args.video, result_dir=args.result_dir, output_path=args.output)
