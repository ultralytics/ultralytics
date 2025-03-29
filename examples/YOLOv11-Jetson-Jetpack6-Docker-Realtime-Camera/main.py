import cv2

from ultralytics import YOLO


def show_camera_with_yolo() -> None:
    """
    Display real-time camera feed with YOLOv11 object detection.

    This function captures video from a specified camera device, processes each frame
    using a YOLOv11 model for object detection, and displays the annotated frames
    in a window. The function will continue to display the video feed until the
    'ESC' key or 'q' is pressed.

    Args:
        None

    Returns:
        None

    Examples:
        >>> show_camera_with_yolo()
    """
    # Load the YOLO model
    model = YOLO("yolo11n.pt")
    window_title = "Jetson Real Time Camera with YOLOv11"

    # Change to the camera ID you want to use "/dev/video0" or "/dev/video1"
    camera_id = "/dev/video0"
    video_capture = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

    # Set camera properties (optional)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    video_capture.set(cv2.CAP_PROP_FPS, 30)

    if video_capture.isOpened():
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    print("Error: Unable to capture frame")
                    break

                # Perform inference with YOLO model
                results = model(frame)

                # Draw results (Bounding boxes, labels, etc.)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0]
                        confidence = box.conf[0]
                        label = int(box.cls[0])

                        # Draw the bounding box and label
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            f"{label} {confidence:.2f}",
                            (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2,
                        )

                # Display the frame
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break

                # Stop the program on the ESC key or 'q'
                keyCode = cv2.waitKey(10) & 0xFF
                if keyCode == 27 or keyCode == ord("q"):
                    break

        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")


if __name__ == "__main__":
    show_camera_with_yolo()
