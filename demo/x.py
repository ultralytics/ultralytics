import sys
import cv2
import numpy as np
from time import time
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer, Qt, QRectF
from PyQt6.QtGui import QImage, QPixmap, QKeyEvent, QPainter, QColor, QFont
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
import os

os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
class FrameWorker(QObject):
    frame_ready = pyqtSignal(QImage, float)  # Emit both the QImage and the calculated FPS

    def __init__(self, video_path):
        super(FrameWorker, self).__init__()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Video cannot be opened")
        self.paused = False
        self.last_timestamp = time()

    def process_frames(self):
        if not self.paused:
            ret, frame = self.cap.read()
            if ret:
                current_timestamp = time()
                elapsed_time = current_timestamp - self.last_timestamp
                fps = 1 / elapsed_time if elapsed_time else 0
                self.last_timestamp = current_timestamp

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channel = rgb_frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(rgb_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                self.frame_ready.emit(q_image, fps)
            else:
                self.cap.release()

    def toggle_pause(self):
        self.paused = not self.paused

class VideoDisplay(QGraphicsView):
    def __init__(self, video_path):
        super(VideoDisplay, self).__init__()
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.thread = QThread()
        self.worker = FrameWorker(video_path)
        self.worker.moveToThread(self.thread)
        self.worker.frame_ready.connect(self.update_image)
        self.thread.start()

        self.timer = QTimer()
        self.timer.timeout.connect(self.worker.process_frames)
        self.timer.start(int(1000 / 30))  # Assume 30 FPS

    def update_image(self, q_image, fps):
        # Draw FPS on the image using QPainter
        painter = QPainter(q_image)
        painter.setFont(QFont("Arial", 32))
        painter.setPen(QColor("yellow"))
        painter.drawText(q_image.rect(), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f"FPS: {fps:.2f}")
        painter.end()

        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key.Key_P:
            self.worker.toggle_pause()
        elif event.key() == Qt.Key.Key_Q:
            self.close()
            self.thread.quit()


def main():
    app = QApplication(sys.argv)
    video_path = "./videos/2.mp4"
    display = VideoDisplay(video_path)
    display.show()
    display.resize(640, 360)  # Initial small window size
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
