import threading
import time

import cv2
from threading import Thread


class VideoDisplay:
    def __init__(self, frame_queue, window_name="Video Display"):
        self.frame_queue = frame_queue
        self.window_name = window_name
        self.stopped = False
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)

    def start(self):
        Thread(target=self.display_frames, daemon=True).start()

    def display_frames(self):
        while not self.stopped:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                cv2.imshow(self.window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.stop()

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()


class VideoWriter:
    def __init__(self, output_path, frame_queue, frame_size, fps=20):
        super().__init__()
        self.output_path = output_path
        self.frame_queue = frame_queue
        self.stopped = False
        self.frame_size = frame_size
        self.fps = fps
        self.out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()
    def update(self):
        while not self.stopped or not self.frame_queue.empty():
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is not None:
                    self.out.write(frame)
            else:
                time.sleep(0.01)  # Pause a bit when queue is empty

    def stop(self):
        self.stopped = True
        if threading.current_thread() != self.thread:
            self.thread.join()
        self.out.release()
