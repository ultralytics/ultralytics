import threading

import cv2
from vidgear.gears import WriteGear


class VideoWriter:
    def __init__(self, output_path, frame_size, compression_mode, logging, fps=30):
        self.output_path = output_path
        self.frame_size = frame_size
        self.fps = fps
        self.stopped = False

        # Make sure the codec is compatible with the file extension and system
        # self.writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'avc1'), fps, frame_size)
        # options = {"-vf": f"scale={frame_size[0]}:{frame_size[1]}", "-input_framerate": fps}
        self.writer = WriteGear(output_path, compression_mode=compression_mode, logging=logging)

        # Thread setup
        self.thread = threading.Thread(target=self.write_frames)
        self.thread.daemon = True

        # To store the frame to be written
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        self.thread.start()

    def write_frame(self, frame):
        """Store the frame to be written by the thread."""
        with self.lock:
            self.frame = frame

    def write_frames(self):
        """Thread function to write frames."""
        while not self.stopped or self.frame is not None:
            if self.frame is not None:
                with self.lock:
                    self.writer.write(self.frame)
                    self.frame = None
            else:
                # If no frame is available, give up control briefly
                threading.Event().wait(0.01)

    def stop(self):
        """Ensure the thread stops and the video file is properly closed."""
        self.stopped = True
        self.thread.join()
        # self.writer.release()
        self.writer.close()


# Usage example
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Adjust the source as needed
    ret, test_frame = cap.read()
    if ret:  # Make sure the capture is successful to get the frame dimensions
        frame_size = (test_frame.shape[1], test_frame.shape[0])
        writer = VideoWriter('output.mp4', frame_size, 30)
        writer.start()

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write_frame(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()
        writer.stop()
    else:
        print("Failed to capture video.")
