# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
YOLO11 Instance Segmentation with Axelera Voyager SDK.

Standalone example using the axelera-rt pipeline API.
No ultralytics dependency at runtime.

Results are rendered by the built-in display app: each SegmentedObject draws its
own mask overlay, bounding box, and label, so no hand-written drawing code is
needed.

Usage:
    python yolo11-seg.py --model yolo11n-seg.axm --source 0
    python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4
    python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4 --no-display
    python yolo11-seg.py --model yolo11n-seg.axm --source video.mp4 --output out.mp4
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from axelera.runtime import cv, display, op

# Image-file suffixes that denote a single still frame (keep the window open afterward).
IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def build_pipeline(model_path: str, conf: float = 0.25, iou: float = 0.45):
    """Build the YOLO11 instance segmentation pipeline.

    The seg head emits two outputs (detections, prototypes), so post-processing splits the tuple, runs NMS on the
    detections only, then recombines them with the decoded
    masks. .optimized() lets the runtime fuse operators.
    """
    return op.seq(
        op.color_convert("RGB", src="BGR"),  # OpenCV reads BGR; models expect RGB
        op.letterbox(640, 640),
        op.to_tensor(),
        op.load(model_path),
        # decode_segmentation outputs (detections, protos); NMS only touches detections
        op.decode_segmentation(algo="yolo11", num_classes=80, num_mask_coeffs=32, confidence_threshold=conf),
        op.par(
            op.seq(op.itemgetter(0), op.nms(iou_threshold=iou, max_boxes=300)),
            op.itemgetter(1),
        ),
        # par unpacks its tuple, so this par receives (dets, protos) and returns
        # (rescaled dets, masks) for ax_segmentation
        op.par(
            op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),
            op.proto_to_mask(),
        ),
        op.ax_segmentation(class_id_type=op.CocoClasses),
    ).optimized()


def main(args):
    """Run the YOLO11 segmentation pipeline over the source and render results via the display app."""
    renderer = "none" if args.no_display else "auto"
    # --output saves rendered frames via save_output_video(), which reads the surface
    # frame sink — only the OpenCV renderer has one, so it can't run headless.
    expose_surface = bool(args.output)
    with display.App(renderer=renderer) as visualizer:
        vis = visualizer.create_window("YOLO11 Segmentation", (800, 500), expose_surface=expose_surface)
        pipeline = build_pipeline(args.model, args.conf, args.iou)

        # cv.create_source wants a path/URL string; map a bare camera index to its /dev/video node.
        source_arg = f"/dev/video{args.source}" if args.source.isdigit() else args.source
        # The ffmpeg backend doesn't support V4L2 /dev/video cameras, so decode those with OpenCV;
        # ffmpeg stays the default for files/streams (faster, wider format support).
        backend = "opencv" if source_arg.startswith("/dev/video") else "ffmpeg"
        with cv.create_source(source_arg, backend=backend) as source:
            if args.output:
                vis.save_output_video(args.output, int(source.fps) or 30)
            for frame_count, (img, segments) in enumerate(pipeline.stream(source), start=1):
                if vis.is_closed:
                    break
                vis(img, segments)
                # Headless has no window, so periodically log what was detected.
                if args.no_display and frame_count % 100 == 0:
                    counts = Counter(getattr(s.class_id, "name", str(s.class_id)) for s in segments)
                    summary = ", ".join(f"{name}x{n}" for name, n in counts.most_common()) or "none"
                    print(f"frame {frame_count}: {len(segments)} segments ({summary})")

        # Keep the window open for a still image; gate on the source being an image file, not the
        # frame count -- a 1-frame video or a stream that stops early isn't an image.
        is_image = not args.source.isdigit() and Path(args.source).suffix.lower() in IMAGE_SUFFIXES
        if is_image and not args.no_display and not vis.is_closed:
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO11 Instance Segmentation -- Axelera Voyager SDK")
    parser.add_argument("--model", type=str, required=True, help="Path to compiled .axm model")
    parser.add_argument("--source", type=str, required=True, help="Image, video path, or camera index")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--no-display", action="store_true", help="Disable GUI window (headless mode)")
    parser.add_argument(
        "--output", type=str, default="", help="Save the rendered video to this path (requires a display, not headless)"
    )
    args = parser.parse_args()
    if args.output and args.no_display:
        parser.error("--output requires the display renderer and cannot be combined with --no-display")
    main(args)
