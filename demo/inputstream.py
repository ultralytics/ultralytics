import cv2
import time
import queue
from threading import Thread, Lock
import threading


class WebcamStream:
    def __init__(self, stream_id=0):
        self.stream_id = stream_id  # default is 0 for primary camera
        self.vcap = cv2.VideoCapture(self.stream_id)
        if not self.vcap.isOpened():
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(cv2.CAP_PROP_FPS))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))

        self.grabbed, self.frame = self.vcap.read()
        if not self.grabbed:
            print('[Exiting] No more frames to read')
            exit(0)

        self.stopped = False
        self.lock = Lock()
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True

    def start(self):
        self.stopped = False
        self.t.start()

    def update(self):
        while True:
            if self.stopped:
                break
            grabbed, frame = self.vcap.read()
            if not grabbed:
                self.stop()
                break
            with self.lock:
                self.grabbed, self.frame = grabbed, frame
        self.vcap.release()

    def read(self):
        with self.lock:
            return self.frame.copy() if self.grabbed else None

    def stop(self):
        self.stopped = True
        self.t.join()


class FileVideoStream:
    def __init__(self, filepath, queueSize=10):
        self.filepath = filepath
        self.capture = cv2.VideoCapture(self.filepath)
        if not self.capture.isOpened():
            raise ValueError("Error opening video file")
        self.queue = queue.Queue(maxsize=queueSize)
        self.stopped = False

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            if not self.queue.full():
                ret, frame = self.capture.read()
                if not ret:
                    self.stop()
                    break
                self.queue.put(frame)

    def read(self):
        return self.queue.get() if not self.queue.empty() else None

    def stop(self):
        self.stopped = True
        self.capture.release()


if __name__ == '__main__':
    stream = WebcamStream()
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
                    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Webcam', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    finally:
        stream.stop()
        cv2.destroyAllWindows()
