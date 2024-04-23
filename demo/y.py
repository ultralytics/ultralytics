from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QThread, QMetaObject, Qt, Q_ARG, QTimer
import numpy as np
from PyQt6.QtGui import QImage, QPainter, QFont, QColor, QPixmap
import cv2
from time import time

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem


class FrameWorkerV2(QObject):
    frame_ready = pyqtSignal(QImage, float)

    def __init__(self):
        super(FrameWorkerV2, self).__init__()
        self.paused = False
        self.last_timestamp = time()

    def process_frames(self, frame_rgb, fps):
        if not self.paused:
            if frame_rgb is not None:
                height, width, channel = frame_rgb.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
                print(f"Emitting frame ready signal")
                self.frame_ready.emit(q_image, fps)

    def toggle_pause(self):
        self.paused = not self.paused

class VideoDisplayV2(QGraphicsView):
    def __init__(self, parent=None):
        super(VideoDisplayV2, self).__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        self.thread = QThread()
        self.worker = FrameWorkerV2()
        self.worker.moveToThread(self.thread)
        self.worker.frame_ready.connect(self.update_display)
        self.thread.start()

    def update_display(self, q_image, fps):
        print("Update image called")  # Debug output
        if q_image.isNull():
            print("QImage is null")
        painter = QPainter(q_image)
        painter.setFont(QFont("Arial", 32))
        painter.setPen(QColor("yellow"))
        painter.drawText(q_image.rect(), Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop, f"FPS: {fps:.2f}")
        painter.end()

        pixmap = QPixmap.fromImage(q_image)
        self.pixmap_item.setPixmap(pixmap)
        self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
