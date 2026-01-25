import argparse
import sys

import cv2

from ultralytics import YOLO


def _open_camera(index: int) -> cv2.VideoCapture:
    backend = cv2.CAP_V4L2 if hasattr(cv2, "CAP_V4L2") else 0
    cap = cv2.VideoCapture(index, backend)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def _probe_camera(index: int, retries: int = 3) -> bool:
    cap = _open_camera(index)
    ok = False
    for _ in range(retries):
        ok, frame = cap.read()
        if ok and frame is not None:
            break
    cap.release()
    return bool(ok)


def _find_available_cameras(max_index: int) -> list[int]:
    available = []
    for i in range(max_index + 1):
        if _probe_camera(i):
            available.append(i)
    return available


def main() -> None:
    parser = argparse.ArgumentParser(description="YOLO camera inference")
    parser.add_argument("--source", type=int, default=0, help="Camera index, e.g. 0/1/2")
    parser.add_argument("--probe-max", type=int, default=4, help="Max camera index to probe")
    args = parser.parse_args()

    if not _probe_camera(args.source):
        available = _find_available_cameras(args.probe_max)
        print(f"Camera index {args.source} is not available.")
        print(f"Available indices: {available if available else 'None'}")
        if available:
            print(f"Try: python mycamera.py --source {available[0]}")
        print("If using RealSense, ensure it is exposed as /dev/video* and permissions are set.")
        sys.exit(1)

    model = YOLO(
        "/home/nanyuanchaliang/codespace/YOLO/ultralytics/runs/detect/train3/weights/best.pt"
    )  # Load a pretrained YOLOv10n model
    results = model(
        source=args.source,  # Use 0 for webcam input
        stream=True,  # Enable streaming mode
    )

    for result in results:
        annotated_frame = result.plot()  # Get the annotated frame with detections

        cv2.imshow("YOLOv10 Detection", annotated_frame)  # Display the annotated frame

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to exit
            break


if __name__ == "__main__":
    main()
