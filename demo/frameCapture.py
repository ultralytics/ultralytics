import time

import cv2
from vidgear.gears import VideoGear, CamGear
import supervision as sv

class FrameCapture:
    def __init__(self, source=0, stabilize=False, stream_mode=False, logging=False):
        self.source = source
        if isinstance(self.source, int):
            # Use CamGear for optimized live stream handling
            self.vcap = CamGear(source=self.source, stream_mode=stream_mode, logging=logging)
            height, width, _ = self.vcap.frame.shape
            self.fps = self.vcap.framerate
        else:
            # Use VideoGear for general video file handling
            self.vcap = VideoGear(source=self.source, stabilize=stabilize, stream_mode=stream_mode, logging=logging)
            height, width, _ = self.vcap.stream.frame.shape
            self.fps = self.vcap.stream.framerate
        self.video_info = sv.VideoInfo.from_video_path(self.source)
        self.frame_count = 0
        self.frame_size = (width, height)
        self.stopped = False
    def start(self):
        self.stopped = False
        self.vcap.start()

    def read(self):
        # Simply read from the VidGear capture
        if not self.stopped:
            self.frame_count += 1
            return self.vcap.read()

    def get_frame_count(self):
        return self.frame_count

    def get_fps(self):
        return self.fps

    def get_frame_size(self):
        return self.frame_size

    def stop(self):
        # Safely close the video stream
        self.frame_count = 0
        # self.vcap.release()
        self.vcap.stop()
        self.stopped = True


if __name__ == '__main__':
    # Example usage:
    stream = FrameCapture(0)  # 0 for default webcam
    stream.start()

    try:
        input("Press Enter to stop...")
    finally:
        stream.stop()
        print("Exiting program")

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
