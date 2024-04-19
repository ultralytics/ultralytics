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