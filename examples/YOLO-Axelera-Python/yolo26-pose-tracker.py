# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Pose Estimation + Multi-Object Tracking with Axelera Voyager SDK.

Standalone example using the axelera-runtime2 pipeline API.
No ultralytics dependency at runtime.

Extends yolo26-pose.py with multi-object tracking via op.tracker().
The tracker returns TrackedObjects whose .tracked field is the original
PoseObject; keypoints and skeleton are drawn directly from it.

Usage:
    python yolo26-pose-tracker.py --model yolo26n-pose.axm --source 0
    python yolo26-pose-tracker.py --model yolo26n-pose.axm --source video.mp4
    python yolo26-pose-tracker.py --model yolo26n-pose.axm --source video.mp4 --tracker bytetrack
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

# Per-keypoint BGR colors (17 keypoints, precomputed for cv2)
KPT_COLORS = [
    tuple(int(c) for c in color[::-1])
    for color in POSE_PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
]
# Golden ratio conjugate for well-distributed track colors
GOLDEN_RATIO_CONJUGATE = 0.618033988749895


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Generate consistent BGR color for a track_id using golden ratio hue distribution."""
    hue = (track_id * GOLDEN_RATIO_CONJUGATE) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


class ConfidenceFilter(op.Operator):
    """Squeeze batch dimension and filter detections by confidence score.

    Receives the raw output of op.load() -- for YOLO26-pose this is shaped (1, 300, 57): each row is [x0, y0, x1, y1,
    score, class_id, 17x(kpt_x, kpt_y, kpt_conf)]. Column 4 (score) is used for thresholding.

    This replaces decode_pose which doesn't support YOLO26's column layout (class_id at column 5 shifts all keypoints by
    one).
    """

    threshold: float = 0.25
    score_col: int = 4

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Filter detections by confidence score."""
        if x.ndim == 3:
            x = x[0]
        mask = x[:, self.score_col] >= self.threshold
        return x[mask]


def build_pipeline(model_path: str, conf: float = 0.25, tracker_algo: str = "tracktrack"):
    """Build YOLO26 pose estimation + tracking pipeline.

    YOLO26 is NMS-free (end-to-end), so no op.nms() is needed. op.axpose() converts the numpy array to list[PoseObject],
    which the tracker consumes. tracked.tracked on each result is the original PoseObject.
    """
    return op.seq(
        op.colorconvert("RGB", src="BGR"),  # OpenCV reads BGR; models expect RGB
        op.letterbox(640, 640),
        op.totensor(),
        op.load(model_path),
        ConfidenceFilter(threshold=conf),
        op.to_image_space(keypoint_cols=range(6, 57, 3)),
        op.axpose(num_keypoints=17),
        op.tracker(algo=tracker_algo),
    ).optimized()


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
                cv2.circle(image, (int(kp.x), int(kp.y)), 4, KPT_COLORS[j], -1)

    return image


def main():
    """YOLO26 Pose Tracking example."""
    parser = argparse.ArgumentParser(description="YOLO26 Pose Tracking -- Axelera Voyager SDK")
    parser.add_argument("--model", type=str, required=True, help="Path to compiled .axm model")
    parser.add_argument("--source", type=str, default="0", help="Image, video path, or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument(
        "--tracker",
        type=str,
        default="tracktrack",
        choices=["bytetrack", "oc-sort", "sort", "tracktrack"],
        help="Tracking algorithm",
    )
    parser.add_argument(
        "--no-display", action="store_true", help="Disable GUI window (headless mode, saves to --output)"
    )
    parser.add_argument("--output", type=str, default="output.mp4", help="Output video path when --no-display is set")
    args = parser.parse_args()

    pipeline = build_pipeline(args.model, args.conf, args.tracker)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    writer = None
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        tracked_poses = pipeline(frame)
        annotated = draw_tracked_poses(frame, tracked_poses)
        frame_count += 1

        if args.no_display:
            if writer is None:
                h, w = annotated.shape[:2]
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer.write(annotated)
            if frame_count % 100 == 0:
                print(f"  frame {frame_count}: {len(tracked_poses)} tracks")
        else:
            cv2.imshow("YOLO26 Pose Tracking", annotated)
            if cv2.waitKey(1) & 0xFF in [ord("q"), ord("Q"), 27]:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"Saved {frame_count} frames to {args.output}")
    else:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
