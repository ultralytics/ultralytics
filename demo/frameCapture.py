import cv2
import time
import queue
from threading import Thread, Lock
import threading

import cv2
import time
class FrameCapture:
    def __init__(self, source=0, buffer_size=10):
        self.source = source  # default is 0 for primary camera
        self.vcap = cv2.VideoCapture(self.source)
        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(cv2.CAP_PROP_FPS))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        self.buffer_size = buffer_size
        self.frame_queue = queue.Queue(maxsize=self.buffer_size)
        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = False
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True

    def start(self):
        self.stopped = False
        self.thread.start()

    def update(self):
        while not self.stopped:
            if not self.frame_queue.full():
                grabbed, frame = self.vcap.read()
                if not grabbed:
                    self.stop()
                    break
                self.frame_queue.put((grabbed, frame))

    def read(self):
        if not self.frame_queue.empty():
            self.grabbed, self.frame = self.frame_queue.get()
        return self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True
        if threading.current_thread() != self.thread:
            self.thread.join()
        self.vcap.release()

if __name__ == '__main__':
    stream = FrameCapture('./videos/2.mp4')
    stream.start()
    frame_count = 0
    start_time = time.time()

    try:
        while True:
            frame = stream.read()
            if frame is not None:
                frame_count += 1
                elapsed_time = time.time() - start_time
                if elapsed_time > 0:
                    fps = frame_count / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        cv2.destroyAllWindows()
        stream.stop()
        print("Exiting program")
