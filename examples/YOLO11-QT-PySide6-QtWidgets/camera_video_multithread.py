import os
import sys
import time
from datetime import datetime
from queue import Queue

import cv2
import numpy as np
from PySide6.QtCore import QStandardPaths, QThread, Signal, Slot
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtMultimedia import (
    QMediaFormat,
)
from PySide6.QtWidgets import QApplication, QDialog, QFileDialog, QMainWindow
from ui_camera_video import Ui_camera_video

from ultralytics import YOLO

# pyside6-uic camera_video.ui -o ui_camera_video.py
AVI = "video/x-msvideo"  # AVI
MP4 = "video/mp4"


def get_supported_mime_types():
    """Get supported mime types of media like avi."""
    result = []
    for f in QMediaFormat().supportedFileFormats(QMediaFormat.Decode):
        mime_type = QMediaFormat(f).mimeType()
        result.append(mime_type.name())
    return result


allimg = []
Image_in_queue_maxsize = 100
Result_in_queue_maxsize = 100
image_que = Queue(maxsize=Image_in_queue_maxsize)
result_que = Queue(maxsize=Result_in_queue_maxsize)
informa_que = Queue(1)


class get_image(QThread):
    """A thread generate frame to a queue."""

    def __init__(self, parent=None):
        """Initialize variables."""
        QThread.__init__(self, parent)
        self.trained_file = None
        self.status = True
        self.cap = True
        self.input = "camera"
        self.video = ""
        self.m = None

    def set_input(self, fname):
        """Set input type."""
        self.input = fname

    """
    #########################
    method 1
    #########################
    """

    def run(self):
        """Open a video or directory and push frame to a queue."""
        global image_que, informa_que
        if self.input == "camera":
            self.cap = cv2.VideoCapture(0)
        elif self.input == "video":
            self.cap = cv2.VideoCapture(self.video)
        if self.input in ["camera", "video"] and not self.cap.isOpened():
            self.cap.release()
            cv2.destroyAllWindows()
            image_que.put(0)
            informa_que.put((-1, -1, -1))
            return
        fps = 24
        sizes = (640, 500)
        if self.input in ["camera", "video"]:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            informa_que.put((fps, int(frame_width), int(frame_height)))
            success, frame = self.cap.read()
            while success:
                image_que.put(frame)
                success, frame = self.cap.read()
            self.cap.release()
            cv2.destroyAllWindows()
        else:
            for i in os.listdir(self.video):
                if ".jpg" in i or ".jpeg" in i or ".png" in i or ".bmp" in i:
                    allimg.append(i)
            for i in allimg:
                frame = cv2.imread(os.path.join(self.video, i))
                frame = cv2.resize(frame, sizes)
                image_que.put(frame)
        image_que.put(0)


class predict_image(QThread):
    """A thread get a frame from the queue, predict the frame, the push result to another queue."""

    def __init__(self, parent=None):
        """Initialize variables."""
        QThread.__init__(self, parent)
        self.model = "yolo11n.pt"
        self.m = None

    def set_model(self):
        """Initialize the model."""
        if self.m is not None:
            del self.m
            import gc

            gc.collect()
        self.m = YOLO(self.model)

    def run(self):
        """Get a frame and predict the frame, push the result to another queue."""
        global result_que, image_que
        self.set_model()
        images = []
        while True:
            image = 0
            while len(images) < 1:
                image = image_que.get()
                if isinstance(image, int):
                    image_que.task_done()
                    break
                images.append(image)
            if len(images) == 0:
                break
            result = self.m.predict(images, batch=1)
            # result = self.m.track(images, batch = 1, persist=True)
            result_que.put(result)
            for _ in range(len(images)):
                image_que.task_done()
            images = []
            if isinstance(image, int):
                break
        result_que.put(0)


class write_video(QThread):
    """Get the predicted result and push it to a queue."""

    updateFrame = Signal(cv2.Mat)

    def __init__(self, parent=None):
        """Initialize variables."""
        QThread.__init__(self, parent)
        self.checked = False
        self.videowriter = True
        self.image_stop = False

    def run(self):
        """Get a result frame, send it to the slot, write to video."""
        global informa_que, result_que
        if self.checked:
            fps, frame_width, frame_height = informa_que.get()
            # if fps < 0 and frame_width < 0 and frame_height < 0:
            #     return
            # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fourcc = cv2.VideoWriter_fourcc("X", "V", "I", "D")
            movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
            if "/" in movies_location:
                movies_location = movies_location.replace("/", os.sep)
            if "\\" in movies_location:
                movies_location = movies_location.replace("\\", os.sep)
            date = datetime.now().isoformat().__str__().replace(".", "_").replace(":", "_")
            if fps <= 0:
                fps = 24
            outpath = movies_location + os.sep + date + ".avi"
            self.videowriter = cv2.VideoWriter(outpath, fourcc, fps, (frame_width, frame_height))
            informa_que.task_done()
            informa_que.join()

        while True:
            if self.image_stop:
                break
            result = result_que.get()
            if isinstance(result, int):
                result_que.task_done()
                self.videowriter.release()
                cv2.destroyAllWindows()
                break
            for ind, _ in enumerate(result):
                frame = result[ind].plot()
                # frame = result[ind].orig_img
                if self.checked:
                    self.videowriter.write(frame)

                color_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # # Creating and scaling QImage
                # h, w, ch = color_frame.shape
                # img = QImage(color_frame.data, w, h, ch * w, QImage.Format_RGB888)

                # Emit signal
                self.updateFrame.emit(color_frame)
            result_que.task_done()
        if self.checked:
            self.videowriter.release()
            self.videowriter = True


class Camera(QMainWindow):
    """Including slot functions, bind with slots."""

    def __init__(self):
        """Initialize variables."""
        super().__init__()
        self._ui = Ui_camera_video()
        self._ui.setupUi(self)
        self._ui.start.setEnabled(True)
        self._ui.stop.setEnabled(False)
        self._ui.save.setEnabled(True)
        self._ui.video.setEnabled(False)
        self._ui.lineEdit_2.setEnabled(False)
        # Thread in charge of updating the image
        self.th = get_image(self)
        self.predict = predict_image(self)
        self.write = write_video(self)

        self.write.updateFrame.connect(self.setImage)
        self.write.finished.connect(self.setlast)

        self._ui.input.currentTextChanged.connect(self.input)
        self.nowstatus = self.saveState()
        self.centralwidget_status = self._ui.centralwidget.saveGeometry()
        self.setWindowTitle("Yolov* Camera Video")

    @Slot()
    def open(self):
        """Open the FileDialog and Choose a model from the directory."""
        file_dialog = QFileDialog(self)
        # self._mime_types = ["pth", "pt", "caffemodel", "pb", "tflite", "weight"]
        # file_dialog.setMimeTypeFilters(self._mime_types)
        movies_location = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation)
        file_dialog.setDirectory(movies_location)
        if file_dialog.exec() == QDialog.Accepted:
            self.url = file_dialog.selectedUrls()[0].toString()[10 - 2 :]
            self._ui.lineEdit.setEnabled(True)
            self._ui.lineEdit.setText(self.url)
            self.predict.model = self.url
        else:
            self.predict.model = self._ui.lineEdit.text()

    @Slot()
    def open_video(self):
        """
        Open the FileDialog.

        Choose a video file or Choose a directory which contains images.
        """
        self._ui.save.setEnabled(True)
        self.th.set_input(self._ui.input.currentText())
        if self.th.input == "video":
            file_dialog = QFileDialog(self)
            is_windows = sys.platform == "win32"
            self._mime_types = []
            if not self._mime_types:
                self._mime_types = get_supported_mime_types()
                if is_windows and AVI not in self._mime_types:
                    self._mime_types.append(AVI)
                elif MP4 not in self._mime_types:
                    self._mime_types.append(MP4)

            file_dialog.setMimeTypeFilters(self._mime_types)

            default_mimetype = AVI if is_windows else MP4
            if default_mimetype in self._mime_types:
                file_dialog.selectMimeTypeFilter(default_mimetype)

            movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
            file_dialog.setDirectory(movies_location)
            if file_dialog.exec() == QDialog.Accepted:
                self.url = file_dialog.selectedUrls()[0].toString()[10 - 2 :]
                self._ui.lineEdit_2.setEnabled(True)
                self._ui.lineEdit_2.setText(self.url)
                self.th.video = self.url
            else:
                self.th.video = self._ui.lineEdit_2.text()
        else:
            file_dialog = QFileDialog(self)
            file_dialog.setFileMode(QFileDialog.FileMode.Directory)
            # movies_location = QStandardPaths.writableLocation(QStandardPaths.MoviesLocation)
            # file_dialog.setDirectory(movies_location)
            if file_dialog.exec() == QDialog.Accepted:
                self.url = file_dialog.selectedUrls()[0].toString()[10 - 2 :]
                self._ui.lineEdit_2.setEnabled(True)
                self._ui.lineEdit_2.setText(self.url)
                self.th.video = self.url

                self._ui.start.setEnabled(True)
                self._ui.stop.setEnabled(False)
                self._ui.model.setEnabled(False)
                self._ui.input.setEnabled(True)
                self._ui.lineEdit.setEnabled(False)
                self._ui.lineEdit_2.setEnabled(False)
                self._ui.video.setEnabled(False)

    @Slot()
    def kill_thread(self):
        """A common function called by stop() terminate the thread and reinitialize variable."""
        if not isinstance(self.th.cap, bool):
            self.th.cap.release()
        if self.write.checked:
            if not isinstance(self.write.videowriter, bool):
                self.write.videowriter.release()
        self.th.status = False
        cv2.destroyAllWindows()
        self.status = False
        time.sleep(1)
        self.th.terminate()
        self.predict.terminate()
        self.write.terminate()
        global result_que, image_que, informa_que
        del result_que, image_que
        Queue(maxsize=Image_in_queue_maxsize)
        Queue(maxsize=Result_in_queue_maxsize)
        informa_que = Queue(1)
        print("Finishing...")
        # Give time for the thread to finish
        self.th.cap = True
        self.write.videowriter = True
        self.write.image_stop = False

    @Slot()
    def start(self):
        """Unable some buttons and start the Thread."""
        self._ui.video.setEnabled(False)
        self._ui.stop.setEnabled(True)
        self._ui.start.setEnabled(False)
        self._ui.model.setEnabled(False)
        self._ui.lineEdit.setEnabled(False)
        self._ui.lineEdit_2.setEnabled(False)
        self._ui.input.setEnabled(False)
        self._ui.save.setEnabled(False)
        self.write.checked = self._ui.save.isChecked()
        self.th.set_input(self._ui.input.currentText())
        self.predict.model = self._ui.lineEdit.text()
        self.th.video = self._ui.lineEdit_2.text()
        self._ui.label.setScaledContents(True)
        self.th.start()
        self.predict.start()
        self.write.start()

        # priority1 = QThread.Priority(QThread.Priority.HighestPriority)
        # self.predict.setPriority(priority1)
        priority = QThread.Priority(QThread.Priority.TimeCriticalPriority)
        self.write.setPriority(priority)

    @Slot()
    def setlast(self):
        """When stopped, set a blank image to replace the last frame."""
        size = self._ui.label.size()
        hei = size.height()
        wid = size.width()
        image = np.zeros((wid, hei, 3))
        # image = cv2.resize(image, (wid, hei))
        image = QImage(image.data, wid, hei, 3 * wid, QImage.Format_RGB888)
        self._ui.label.setPixmap(QPixmap.fromImage(image))

    @Slot()
    def stop(self):
        """A slot which receive signal from QtWidgets.Stop the Thread."""
        self.write.image_stop = True
        self.kill_thread()
        self._ui.start.setEnabled(True)
        self._ui.input.setEnabled(True)
        self._ui.model.setEnabled(True)
        self._ui.stop.setEnabled(False)
        self._ui.lineEdit.setEnabled(True)
        self._ui.save.setEnabled(True)
        if self.th.input == "camera":
            self._ui.video.setEnabled(False)
            self._ui.lineEdit_2.setEnabled(False)
        else:
            self._ui.video.setEnabled(True)
            self._ui.lineEdit_2.setEnabled(True)

    @Slot()
    def input(self, text):
        """A slot which set the input type."""
        self._ui.save.setEnabled(True)
        if text == "camera":
            self._ui.start.setEnabled(True)
            self._ui.video.setEnabled(False)
            self._ui.lineEdit_2.setEnabled(False)
        elif text == "video":
            self._ui.start.setEnabled(True)
            self._ui.video.setEnabled(True)
            self._ui.lineEdit_2.setEnabled(True)
        else:
            if os.path.isdir(self.th.video):
                self._ui.start.setEnabled(True)
            else:
                self._ui.start.setEnabled(False)
            self._ui.stop.setEnabled(False)
            self._ui.video.setEnabled(True)
            self._ui.lineEdit_2.setEnabled(True)

        self.th.set_input(text)

    @Slot()
    def save(
        self,
    ):
        """A slot which set whether or not save plotted frame to a video."""
        self.write.checked = self._ui.save.isChecked()

    @Slot(cv2.Mat)
    def setImage(self, image: cv2.Mat):
        """A slot which pad the plotted frame with letterbox padding and draw it to the label."""
        # Creating and scaling QImage
        h, w, ch = image.shape
        size = self._ui.label.size()
        hei = size.height()
        wid = size.width()
        r = min(wid / w, hei / h)
        hh = int(h * r)
        ww = int(w * r)
        image = cv2.resize(image, (ww, hh), interpolation=cv2.INTER_LINEAR)
        left = (wid - ww) // 2
        right = wid - left - ww
        top = (hei - hh) // 2
        bottom = hei - hh - top
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        h, w, ch = image.shape
        image = QImage(image.data, w, h, ch * w, QImage.Format_RGB888)
        # image = image.scaled(self._ui.label.size(), Qt.KeepAspectRatio,
        #   Qt.SmoothTransformation)
        self._ui.label.setPixmap(QPixmap.fromImage(image))


if __name__ == "__main__":
    # pyside6-rcc style.qrc -o style_rc.py
    app = QApplication(sys.argv)
    # QCoreApplication.setApplicationName("Yolo11 Camera Video")
    camera = Camera()
    camera.show()
    icon = QIcon(":/h0rzq1qg.png")
    app.setWindowIcon(icon)
    # app.setApplicationDisplayName("Yolo11 Camera Video")
    # app.setObjectName("Yolo11 Camera Video")
    # app.setDesktopFileName("Yolo11 Camera Video")
    # app.setOrganizationName("Yolo11 Camera Video")
    # app.setApplicationName("Yolo11 Camera Video")

    # # Force the style to be the same on all OSs:
    # app.setStyle("Fusion")

    # # Now use a palette to switch to dark colors:
    # palette = QPalette() # 调色板
    # palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))  # 给窗口整体配置颜色
    # palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)  # 给窗口文字配置颜色
    # palette.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53)) # 给按钮本身配置颜色
    # palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white) # 给按钮文字配置颜色
    # palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red) # 给高亮文本配置颜色，
    # app.setPalette(palette)

    sys.exit(app.exec())
