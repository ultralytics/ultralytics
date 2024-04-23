import sys
import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget

class FrameWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)

    def __init__(self, video_path):
        super(FrameWorker, self).__init__()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Video cannot be opened")

    def process_frames(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                # Emit the frame as an np.ndarray
                self.frame_ready.emit(frame)
            else:
                break
        self.cap.release()

class FrameDisplay(QWidget):
    def __init__(self, video_path):
        super(FrameDisplay, self).__init__()
        self.setWindowTitle("Video Display")
        self.video_label = QLabel()
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        self.setLayout(layout)


        # Start the frame update thread
        self.start_frame_update_thread(video_path)

    def update_image(self, img):
        # Convert numpy array to QImage and then to QPixmap
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QPixmap.fromImage(q_img))

    def start_frame_update_thread(self, video_path):
        self.thread = QThread()
        self.worker = FrameWorker(video_path)
        self.worker.moveToThread(self.thread)
        self.worker.frame_ready.connect(self.update_image)
        self.thread.started.connect(self.worker.process_frames)
        self.thread.start()

def main():
    app = QApplication(sys.argv)
    video_path = "./videos/2.mp4"  # Specify the path to your video file
    display = FrameDisplay(video_path)
    display.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
