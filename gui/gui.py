import os
import sys
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from torchvision.transforms import functional as F
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QImage, QIcon
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QMessageBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

from pathlib import Path

root = Path(__file__).parent.absolute().__str__()
parent_root = Path(__file__).parent.parent.absolute().__str__()

sys.path.append(parent_root)

def model_inference(model_path, image_np):
    model = YOLO(model_path)

    output = model.predict(source=image_np, save=False)

    return output

def post_process(output):
    boxes = []
    texts = []

    for result in output:
        # Detection
        boxes.append(result.boxes.xyxy)   # box with xyxy format, (N, 4)
        texts.append(result.boxes.cls)    # cls, (N, 1)

    print("boxes :", boxes)
    print("texts :", texts)
        
    return boxes, texts

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Bone Fracture Detection App")
        self.setGeometry(50, 50, 200, 200)

        pagelayout = QHBoxLayout()
        button_layout = QVBoxLayout()
        right_layout = QVBoxLayout()
        images_layout = QHBoxLayout()
        labels_layout = QVBoxLayout()

        ## Buttons
        btn_open_img = QPushButton("Open Image")
        btn_open_img.setIcon(QIcon(os.path.join(parent_root, "icons", "folder.png")))
        btn_open_img.setFixedSize(162, 32)
        btn_open_img.pressed.connect(self.open_image)
        btn_open_img.resize(50, 50)
        button_layout.addWidget(btn_open_img)

        self.btn_predict = QPushButton("Predict")
        self.btn_predict.setIcon(QIcon(os.path.join(parent_root, "icons", "hand_radiography.png")))
        self.btn_predict.setFixedSize(162, 32)
        self.btn_predict.setEnabled(False)
        self.btn_predict.pressed.connect(self.predict)
        button_layout.addWidget(self.btn_predict)

        self.btn_save = QPushButton("Save")
        self.btn_save.setIcon(QIcon(os.path.join(parent_root, "icons", "diskette.png")))
        self.btn_save.setFixedSize(162, 32)
        self.btn_save.setEnabled(False)
        self.btn_save.pressed.connect(self.save)
        button_layout.addWidget(self.btn_save)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.setIcon(QIcon(os.path.join(parent_root, "icons", "restart.png")))
        self.btn_clear.setFixedSize(162, 32)
        self.btn_clear.setEnabled(False)
        self.btn_clear.pressed.connect(self.clear)
        button_layout.addWidget(self.btn_clear)

        btn_quit = QPushButton("Quit")
        btn_quit.setIcon(QIcon(os.path.join(parent_root, "icons", "sign-out.png")))
        btn_quit.setFixedSize(162, 32)
        btn_quit.pressed.connect(self.quit)
        button_layout.addWidget(btn_quit)

        ### Images
        self.original_image = QLabel()
        self.predicted_image = QLabel()
        images_layout.addWidget(self.original_image)
        images_layout.addWidget(self.predicted_image)

        ## Labels
        self.loaded_image_label = QLabel()
        self.save_log = QLabel()
        labels_layout.addWidget(self.loaded_image_label)
        labels_layout.addWidget(self.save_log)

        ## Layouts
        right_layout.addLayout(images_layout)
        right_layout.addLayout(labels_layout)
        pagelayout.addLayout(button_layout)
        pagelayout.addLayout(right_layout)

        widget = QWidget()
        widget.setLayout(pagelayout)
        self.setCentralWidget(widget)
    
    def _convert_cv_qt(self, cv_img):
        """ Convert from an opencv image to QPixmap
        """
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 640, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def open_image(self):
        self.file_name = QFileDialog.getOpenFileName(self, "Open Image", parent_root, "Image Files (*.png *.jpg *.bmp)")[0]
        print(f"Image selected: {self.file_name}")

        self.img = cv2.imread(self.file_name)[..., ::-1]
        print(f"Image shape: {self.img.shape}")
        pixmap = self._convert_cv_qt(self.img)
        self.original_image.setPixmap(pixmap)

        self.btn_predict.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_save.setEnabled(False)
        self.predicted_image.clear()
        self.save_log.clear()

        self.loaded_image_label.setText(f"Uploaded image: {os.path.basename(self.file_name)}\n")


    def predict(self):
        model_path = os.path.join(parent_root, "my_yolov8s.pt")
        img_path = str(self.file_name)  # Use self.file_name instead of self.img
        print(f"Image path: {img_path}")
        output = model_inference(model_path, img_path)

        # Extract the output image tensor from the output dictionary
        out_img = output[0].plot()

        pixmap = self._convert_cv_qt(out_img)
        self.predicted_image.setPixmap(pixmap)
        self.btn_save.setEnabled(True)


    def save(self):
        save_file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "JPEG (*.jpg;*.jpeg);;PNG (*.png)")
        if save_file_path:
            print(f"Save image in: {save_file_path}")
            save_file_name = str(save_file_path)
            cv2.imwrite(save_file_name, self.out_img)
            QMessageBox.information(self, "Image Saved", "Image saved successfully!")

    def clear(self):

        self.original_image.clear()
        self.predicted_image.clear()
        self.loaded_image_label.clear()
        self.save_log.clear()
        self.btn_predict.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_clear.setEnabled(False)

    def quit(self):
        msg = QMessageBox.question(self, "Close the app?", "Are you sure to close the app?", QMessageBox.Yes | QMessageBox.No)
        if (msg == QMessageBox.Yes):
            self.close()

app = QApplication(sys.argv)

window = MainWindow()
window.show()

app.exec()
