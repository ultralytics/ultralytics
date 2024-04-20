import cv2
import time
import queue
from frameCapture import FrameCapture
from frameProcessing import VideoWriter


def display_frames(framecapture, frame_queue):
    frame_count = 0
    start_time = time.time()

    while not framecapture.stopped:
        frame = framecapture.read()
        if frame is not None:
            frame_queue.put(frame)  # Send frame to the video writer queue
            current_time = time.time()
            elapsed_time = current_time - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # put the frame show
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Frame', frame)
            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    source = './videos/2.mp4'
    output_path = './videos/output.mp4'
    framecapture = FrameCapture(source)
    framecapture.start()
    frame_queue = queue.Queue()

    # Assuming FrameCapture provides the frame size
    frame_size = (1920, 1080)  # This should be dynamically set based on your actual video frame size

    video_writer = VideoWriter(output_path, frame_queue, frame_size)
    video_writer.start()

    try:
        display_frames(framecapture, frame_queue)
    finally:
        framecapture.stop()
        video_writer.stop()
