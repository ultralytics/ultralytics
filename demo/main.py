import cv2
import time
from FrameCapture import FrameCapture


def display_frames(framecapture):
    frame_count = 0
    start_time = time.time()

    while not framecapture.stopped:
        frame = framecapture.read()
        if frame is not None:
            # Increment frame count
            frame_count += 1

            # Calculate FPS
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > 0:
                fps = frame_count / elapsed_time
            else:
                fps = 0

            # Display FPS on the frame
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)

            # Show the frame
            cv2.imshow('Frame', frame)

            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    source = './videos/2.mp4'
    framecapture = FrameCapture(source)
    framecapture.start()
    try:
        display_frames(framecapture)
    finally:
        framecapture.stop()
