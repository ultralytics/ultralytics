################################################################################
## Form generated from reading UI file 'camera_video.ui'
##
## Created by: Qt User Interface Compiler version 6.7.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (
    QCoreApplication,
    QMetaObject,
    QRect,
    QSize,
    Qt,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QLabel,
    QLineEdit,
    QMenuBar,
    QPushButton,
    QSizePolicy,
    QStatusBar,
    QWidget,
)


class Ui_camera_video:
    """WARNING,All changes made in this file will be lost when recompiling UI file."""

    def setupUi(self, camera_video):
        """WARNING,All changes made in this file will be lost when recompiling UI file."""
        if not camera_video.objectName():
            camera_video.setObjectName("camera_video")
        camera_video.setEnabled(True)
        camera_video.resize(616, 484)
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(camera_video.sizePolicy().hasHeightForWidth())
        camera_video.setSizePolicy(sizePolicy)
        camera_video.setDocumentMode(False)
        self.centralwidget = QWidget(camera_video)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("")
        self.gridLayout = QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.label = QLabel(self.centralwidget)
        self.label.setObjectName("label")
        sizePolicy1 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        sizePolicy1.setHorizontalStretch(0)
        sizePolicy1.setVerticalStretch(0)
        sizePolicy1.setHeightForWidth(self.label.sizePolicy().hasHeightForWidth())
        self.label.setSizePolicy(sizePolicy1)
        self.label.setMinimumSize(QSize(1, 1))
        self.label.setStyleSheet("background-color: rgb(0, 0, 0);")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.gridLayout.addWidget(self.label, 0, 0, 1, 2)

        self.widget = QWidget(self.centralwidget)
        self.widget.setObjectName("widget")
        sizePolicy2 = QSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
        sizePolicy2.setHorizontalStretch(0)
        sizePolicy2.setVerticalStretch(0)
        sizePolicy2.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy2)
        self.gridLayout_2 = QGridLayout(self.widget)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.lineEdit_2 = QLineEdit(self.widget)
        self.lineEdit_2.setObjectName("lineEdit_2")

        self.gridLayout_2.addWidget(self.lineEdit_2, 0, 5, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.model = QPushButton(self.widget)
        self.model.setObjectName("model")
        self.model.setStyleSheet(
            "selection-background-color: qconicalgradient(cx:0.5, cy:0.5, angle:0, stop:0 rgba(255, 255, 255, 255), stop:0.373979 rgba(255, 255, 255, 255), stop:0.373991 rgba(33, 30, 255, 255), stop:0.624018 rgba(33, 30, 255, 255), stop:0.624043 rgba(255, 0, 0, 255), stop:1 rgba(255, 0, 0, 255));\n"
            "selection-background-color: rgb(255, 56, 255);\n"
            "selection-color: rgb(43, 82, 255);"
        )

        self.gridLayout_2.addWidget(self.model, 0, 3, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.lineEdit = QLineEdit(self.widget)
        self.lineEdit.setObjectName("lineEdit")

        self.gridLayout_2.addWidget(self.lineEdit, 0, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.video = QPushButton(self.widget)
        self.video.setObjectName("video")

        self.gridLayout_2.addWidget(self.video, 0, 6, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.input = QComboBox(self.widget)
        self.input.addItem("")
        self.input.addItem("")
        self.input.addItem("")
        self.input.setObjectName("input")
        self.input.setStyleSheet("")

        self.gridLayout_2.addWidget(self.input, 0, 0, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.start = QPushButton(self.widget)
        self.start.setObjectName("start")

        self.gridLayout_2.addWidget(self.start, 2, 0, 1, 1)

        self.stop = QPushButton(self.widget)
        self.stop.setObjectName("stop")

        self.gridLayout_2.addWidget(self.stop, 2, 1, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.save = QCheckBox(self.widget)
        self.save.setObjectName("save")
        self.save.setStyleSheet(
            "selection-background-color: rgb(166, 106, 255);\n"
            "selection-background-color: qconicalgradient(cx:0.5, cy:0.5, angle:0, stop:0 rgba(255, 255, 255, 255), stop:0.373979 rgba(255, 255, 255, 255), stop:0.373991 rgba(33, 30, 255, 255), stop:0.624018 rgba(33, 30, 255, 255), stop:0.624043 rgba(255, 0, 0, 255), stop:1 rgba(255, 0, 0, 255));\n"
            "selection-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0, y2:1, stop:0 rgba(255, 0, 0, 255), stop:0.339795 rgba(255, 0, 0, 255), stop:0.339799 rgba(255, 255, 255, 255), stop:0.662444 rgba(255, 255, 255, 255), stop:0.662469 rgba(0, 0, 255, 255), stop:1 rgba(0, 0, 255, 255));"
        )

        self.gridLayout_2.addWidget(self.save, 2, 3, 1, 1, Qt.AlignmentFlag.AlignLeft)

        self.gridLayout.addWidget(self.widget, 2, 0, 1, 2)

        camera_video.setCentralWidget(self.centralwidget)
        self.menubar = QMenuBar(camera_video)
        self.menubar.setObjectName("menubar")
        self.menubar.setGeometry(QRect(0, 0, 616, 22))
        camera_video.setMenuBar(self.menubar)
        self.statusbar = QStatusBar(camera_video)
        self.statusbar.setObjectName("statusbar")
        self.statusbar.setEnabled(True)
        camera_video.setStatusBar(self.statusbar)

        self.retranslateUi(camera_video)
        self.start.clicked.connect(camera_video.start)
        self.save.checkStateChanged.connect(camera_video.save)
        self.stop.clicked.connect(camera_video.stop)
        self.model.clicked.connect(camera_video.open)
        self.video.clicked.connect(camera_video.open_video)
        self.stop.clicked.connect(camera_video.setlast)

        QMetaObject.connectSlotsByName(camera_video)

    # setupUi

    def retranslateUi(self, camera_video):
        """WARNING,All changes made in this file will be lost when recompiling UI file."""
        camera_video.setWindowTitle(QCoreApplication.translate("camera_video", "Yolo11 Camera Video", None))
        self.label.setText("")
        self.model.setText(QCoreApplication.translate("camera_video", "model", None))
        self.lineEdit.setText(QCoreApplication.translate("camera_video", "yolo11n.pt", None))
        self.video.setText(QCoreApplication.translate("camera_video", "video or image", None))
        self.input.setItemText(0, QCoreApplication.translate("camera_video", "camera", None))
        self.input.setItemText(1, QCoreApplication.translate("camera_video", "video", None))
        self.input.setItemText(2, QCoreApplication.translate("camera_video", "image directory", None))

        self.start.setText(QCoreApplication.translate("camera_video", "media start", None))
        self.stop.setText(QCoreApplication.translate("camera_video", "stop", None))
        self.save.setText(QCoreApplication.translate("camera_video", "save video", None))

    # retranslateUi
