# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Ultralytics YOLO26 Pose Estimation with optional Multi-Object Tracking using Axelera Voyager SDK.

Standalone example using the axelera-rt pipeline API.
No ultralytics dependency at runtime.

Ultralytics YOLO26 is NMS-free: the model outputs (1, 300, 57) already in final format
[x0, y0, x1, y1, score, class_id, kpt0_x, kpt0_y, kpt0_conf, ...].
We use a small ConfidenceFilter operator instead of decode_pose (which only
supports yolov8/yolo11 raw output layout).

When --tracker is set (default: tracktrack), the pipeline appends op.tracker()
for multi-object tracking. Use --tracker none to disable tracking.

Usage:
    python yolo26-pose-tracker.py --model yolo26n-pose.axm --source 0
    python yolo26-pose-tracker.py --model yolo26n-pose.axm --source video.mp4
    python yolo26-pose-tracker.py --model yolo26n-pose.axm --source video.mp4 --tracker bytetrack
    python yolo26-pose-tracker.py --model yolo26n-pose.axm --source image.jpg --tracker none
"""

from __future__ import annotations

import argparse
import colorsys

import cv2
import numpy as np
from axelera.runtime import op

# COCO skeleton: 19 limb connections (1-indexed keypoint pairs)
COCO_SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7],
]  # fmt: skip

# Pose color palette (RGB) from Ultralytics
POSE_PALETTE = np.array(
    [
        [255, 128, 0],
        [255, 153, 51],
        [255, 178, 102],
        [230, 230, 0],
        [255, 153, 255],
        [153, 204, 255],
        [255, 102, 255],
        [255, 51, 255],
        [102, 178, 255],
        [51, 153, 255],
        [255, 153, 153],
        [255, 102, 102],
        [255, 51, 51],
        [153, 255, 153],
        [102, 255, 102],
        [51, 255, 51],
        [0, 255, 0],
        [0, 0, 255],
        [255, 0, 0],
        [255, 255, 255],
    ],
    dtype=np.uint8,
)

# Per-keypoint colors (17 keypoints, palette indices) — RGB for no-tracker, BGR for tracker
KPT_COLORS = POSE_PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# Per-limb colors (19 limbs, palette indices)
LIMB_COLORS = POSE_PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]
# Golden ratio conjugate for well-distributed track colors
GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Generate consistent BGR color for a track_id using golden ratio hue distribution."""
    hue = (track_id * GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


class ConfidenceFilter(op.Operator):
    """Squeeze batch dimension and filter detections by confidence score.

    Receives the raw output of op.load() -- for Ultralytics YOLO26-pose this is shaped (1, 300, 57): each row is [x0,
    y0, x1, y1, score, class_id, 17x(kpt_x, kpt_y, kpt_conf)]. Column 4 (score) is used for thresholding.

    This replaces decode_pose which doesn't support Ultralytics YOLO26's column layout (class_id at column 5 shifts all
    keypoints by one).
    """

    threshold: float = 0.25
    score_col: int = 4

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter detections by confidence score."""
        if x.ndim == 3:
            x = x[0]
        return x[x[:, self.score_col] >= self.threshold]


def build_pipeline(model_path: str, conf: float = 0.25, tracker_algo: str | None = "tracktrack"):
    """Build Ultralytics YOLO26 pose estimation pipeline with optional tracking.

    Ultralytics YOLO26 is NMS-free (end-to-end), so no op.nms() is needed. When tracker_algo is provided, op.tracker()
    is appended for multi-object tracking. Calls .optimized() so the runtime can fuse operators for maximum throughput.
    """
    stages = [
        op.colorconvert("RGB", src="BGR"),  # OpenCV reads BGR; models expect RGB
        op.letterbox(640, 640),
        op.totensor(),
        op.load(model_path),
        ConfidenceFilter(threshold=conf),
        op.to_image_space(keypoint_cols=range(6, 57, 3)),
    ]
    if tracker_algo:
        stages.extend([op.axpose(num_keypoints=17), op.tracker(algo=tracker_algo)])
    return op.seq(*stages).optimized()


def draw_pose(image: np.ndarray, detections: np.ndarray, conf: float = 0.25) -> np.ndarray:
    """Draw pose estimation results on the image (in-place, no tracking)."""
    for det in detections:
        score = det[4]
        if score < conf:
            continue

        # Bounding box
        x0, y0, x1, y1 = map(int, det[:4])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Parse 17 keypoints: columns 6..56, stride 3 (x, y, conf)
        kpts = det[6:].reshape(17, 3)

        # Draw skeleton limbs
        for i, (a, b) in enumerate(COCO_SKELETON):
            kp_a, kp_b = kpts[a - 1], kpts[b - 1]
            if kp_a[2] > 0.5 and kp_b[2] > 0.5:
                color = tuple(int(c) for c in LIMB_COLORS[i][::-1])  # RGB -> BGR
                pt_a = (int(kp_a[0]), int(kp_a[1]))
                pt_b = (int(kp_b[0]), int(kp_b[1]))
                cv2.line(image, pt_a, pt_b, color, 2)

        # Draw keypoints
        for j, kp in enumerate(kpts):
            if kp[2] > 0.5:
                color = tuple(int(c) for c in KPT_COLORS[j][::-1])  # RGB -> BGR
                cv2.circle(image, (int(kp[0]), int(kp[1])), 4, color, -1)

    return image


def draw_tracked_poses(image: np.ndarray, tracked_poses: list) -> np.ndarray:
    """Draw tracked pose results: bbox with track ID color, skeleton, and keypoints (in-place)."""
    for tracked in tracked_poses:
        color = get_track_color(tracked.track_id)

        # Bounding box
        bbox = tracked.predicted_bbox
        x0, y0, x1, y1 = int(bbox.x0), int(bbox.y0), int(bbox.x1), int(bbox.y1)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cv2.putText(image, f"ID {tracked.track_id}", (x0, y0 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Access keypoints from the original PoseObject via tracked.tracked
        pose = tracked.tracked
        if not hasattr(pose, "keypoints") or not pose.keypoints:
            continue

        kpts = pose.keypoints

        # Draw skeleton limbs in track color (links skeleton to its box visually)
        for i, (a, b) in enumerate(COCO_SKELETON):
            kp_a, kp_b = kpts[a - 1], kpts[b - 1]
            if kp_a.confidence > 0.5 and kp_b.confidence > 0.5:
                pt_a = (int(kp_a.x), int(kp_a.y))
                pt_b = (int(kp_b.x), int(kp_b.y))
                cv2.line(image, pt_a, pt_b, color, 2)

        # Draw keypoints
        for j, kp in enumerate(kpts):
            if kp.confidence > 0.5:
                cv2.circle(image, (int(kp.x), int(kp.y)), 4, tuple(int(c) for c in KPT_COLORS[j][::-1]), -1)

    return image


def main():
    """Ultralytics YOLO26 Pose Estimation + Tracking example."""
    parser = argparse.ArgumentParser(description="Ultralytics YOLO26 Pose Estimation + Tracking - Axelera Voyager SDK")
    parser.add_argument("--model", type=str, required=True, help="Path to compiled .axm model")
    parser.add_argument("--source", type=str, default="0", help="Image, video path, or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--tracker",
        type=str,
        default="tracktrack",
        choices=["bytetrack", "oc-sort", "sort", "tracktrack", "none"],
        help="Tracking algorithm (use 'none' to disable tracking)",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable GUI window (headless mode, saves to --output)"
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path when --no-display is set")
    args = parser.parse_args()

    tracker_algo = None if args.tracker == "none" else args.tracker
    pipeline = build_pipeline(args.model, args.conf, tracker_algo)
    use_tracking = tracker_algo is not None

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    frame_count = 0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    is_image = frames == 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pipeline(frame)
        annotated = draw_tracked_poses(frame, results) if use_tracking else draw_pose(frame, results, args.conf)
        frame_count += 1

        if args.no_display:
            if writer is None:
                h, w = annotated.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)
            if frame_count % 100 == 0:
                print(f"  frame {frame_count}: {len(results)} {'tracks' if use_tracking else 'detections'}")
        else:
            title = "Ultralytics YOLO26 Pose Tracking" if use_tracking else "Ultralytics YOLO26 Pose"
            cv2.imshow(title, annotated)
            if cv2.waitKey(0 if is_image else 1) & 0xFF in [ord("q"), ord("Q"), 27]:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved {frame_count} frames to {args.output}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
