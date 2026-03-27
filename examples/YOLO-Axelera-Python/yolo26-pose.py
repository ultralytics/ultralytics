# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO26 Pose Estimation with Axelera Voyager SDK.

Standalone example using the axelera-runtime2 pipeline API.
No ultralytics dependency at runtime.

YOLO26 is NMS-free: the model outputs (1, 300, 57) already in final format
[x0, y0, x1, y1, score, class_id, kpt0_x, kpt0_y, kpt0_conf, ...].
We use a small ConfidenceFilter operator instead of decode_pose (which only
supports yolov8/yolo11 raw output layout).

Usage:
    python yolo26-pose.py --model yolo26n-pose.axm --source 0
    python yolo26-pose.py --model yolo26n-pose.axm --source video.mp4
"""

from __future__ import annotations

import argparse

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

# Per-keypoint colors (17 keypoints, palette indices)
KPT_COLORS = POSE_PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]
# Per-limb colors (19 limbs, palette indices)
LIMB_COLORS = POSE_PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


class ConfidenceFilter(op.Operator):
    """Squeeze batch dimension and filter detections by confidence score.

    Receives the raw output of op.load() — for YOLO26-pose this is shaped (1, 300, 57): each row is [x0, y0, x1, y1,
    score, class_id, 17×(kpt_x, kpt_y, kpt_conf)]. Column 4 (score) is used for thresholding.

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


def build_pipeline(model_path: str, conf: float = 0.25):
    """Build YOLO26 pose estimation pipeline.

    YOLO26 is NMS-free (end-to-end), so no op.nms() is needed. Calls .optimized() so the runtime can fuse operators for
    maximum throughput.
    """
    model_op = op.load(model_path)
    return op.seq(
        op.colorconvert("RGB", src="BGR"),  # OpenCV reads BGR; models expect RGB
        op.letterbox(640, 640),
        op.totensor(),
        model_op,
        ConfidenceFilter(threshold=conf),
        op.to_image_space(keypoint_cols=range(6, 57, 3)),
    ).optimized()


def draw_pose(image: np.ndarray, detections: np.ndarray, conf: float = 0.25) -> np.ndarray:
    """Draw pose estimation results on the image (in-place)."""
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


def main():
    """YOLO26 Pose Estimation example."""
    parser = argparse.ArgumentParser(description="YOLO26 Pose Estimation -- Axelera Voyager SDK")
    parser.add_argument("--model", type=str, required=True, help="Path to compiled .axm model")
    parser.add_argument("--source", type=str, default="0", help="Image, video path, or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    args = parser.parse_args()

    pipeline = build_pipeline(args.model, args.conf)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open source: {args.source}")

    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    is_image = frames == 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = pipeline(frame)
        annotated = draw_pose(frame, detections, args.conf)

        cv2.imshow("YOLO26 Pose", annotated)
        if cv2.waitKey(0 if is_image else 1) & 0xFF in [ord("q"), ord("Q"), 27]:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
