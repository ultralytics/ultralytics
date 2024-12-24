// Copyright (C) 2021 The Qt Company Ltd.
// SPDX-License-Identifier: LicenseRef-Qt-Commercial


import QtQuick
import QtQuick.Layouts
import QtQuick.Controls
import QtQuick.Window
import QtQuick.Controls.Material
import QtQuick.Dialogs
import io.qt.textproperties

// 窗口
ApplicationWindow {
    id: root
    width: 790
    height: 600
    minimumHeight: 460
    minimumWidth: 790
    visible: true
    // color: Config.mainColor
    title: qsTr("Yolov* camera video")
    Material.theme: Material.Dark
    Material.accent: Material.Red

    // Camera_video { // 和main.py中的槽类对应，将下面的信号signals connect到槽slot所在的类
    //     id: my_image_provider
    // }

    // Used as an example of a backend - this would usually be
    // e.g. a C++ type exposed to QML.
    QtObject {
        id: backend
        property int modifier
    }

    signal model_fileOpened(path: url)

    //https://stackoverflow.com/questions/72729052/how-to-show-opencv-camera-feed-in-a-qml-application#
    //https://stackoverflow.com/questions/73684543/how-to-display-opencv-camera-feed-in-a-qml-pyside6-application?r=SearchResults#
    Connections { //https://doc.qt.io/qt-6/qtqml-syntax-signals.html
        target: my_image_provider
        function onImageChange(image) {
            // console.log("emit")
            image_element.reLoadImage()
        }
    }

    FileDialog {
        id: fileDialog_model
        currentFolder: StandardPaths.standardLocations(StandardPaths.MoviesLocation)[0]
        nameFilters: ["*"]
        // selectedNameFilter.index: root.selectedNameFilter
        title: qsTr("Please choose a model")
        selectedFile: fileDialog_model.selectedFile
        onAccepted: {
            input_or_choose_mode_path.text = fileDialog_model.selectedFile.toString()
            my_image_provider.get_model(fileDialog_model.selectedFile.toString())
        }
    }

    FileDialog {
        id: fileDialog_video_image
        currentFolder: StandardPaths.standardLocations(StandardPaths.MoviesLocation)[0]
        nameFilters: ["*"]
        // selectedNameFilter.index: root.selectedNameFilter
        title: qsTr("Please choose a video")
        selectedFile: fileDialog_video_image.selectedFile
        onAccepted: {
            choose_video_or_image.text = fileDialog_video_image.selectedFile.toString()
            my_image_provider.get_video(fileDialog_video_image.selectedFile.toString())
        }
    }

    FolderDialog {
        id: folderDialog
        currentFolder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0]
        selectedFolder: viewer.folder
        title: qsTr("Please choose a images directory")
        onAccepted: {
            choose_video_or_image.text = folderDialog.selectedFolder.toString()
            my_image_provider.get_video(folderDialog.selectedFolder.toString())
        }
    }

    GridLayout { // 布局2行1列
        id: grid // 名称
        anchors.fill: parent
        anchors.margins: 2
        columns: 1
        rows: 3
        rowSpacing: 1
        columnSpacing: 1

        RowLayout { //第一行
            id: row1
            spacing: 1
            // Layout.columnSpan: 1
            // Layout.preferredWidth: 400
            // Layout.preferredHeight: 400
            Layout.alignment: Qt.AlignHCenter | Qt.AlignVCenter

            Image { // 放置一个图片 https://doc.qt.io/qt-6/qml-qtquick-image.html
                id: image_element
                fillMode: Image.PreserveAspectFit // 填充模式
                anchors.centerIn: root // 在矩形内部居中放置
                source: "image://My_image_provider/img"   // 图片路径
                opacity: 1.0           // 不透明度
                cache: false
                // onImageChange {
                //     source: "image://Camera_video/"
                // }
                property int counter: 0
                function reLoadImage() {
                    // counter!=counter
                    // source = "./logo.png"
                    source = "image://My_image_provider/img?id=" + counter
                    counter++
                    counter%=999999999
                }
            }
        }
        RowLayout { // 第2列
            id: rightcolumn
            spacing: 2
            Layout.columnSpan: 1
            // Layout.preferredWidth: 400
            // Layout.preferredHeight: 100
            Layout.fillWidth: true

            // Layout.alignment: Qt.AlignBottom
            Layout.alignment: Qt.AlignLeft
            //Center

            ComboBox {
                id: combox
                textRole: "text"
                valueRole: "value"
                // When an item is selected, update the backend.
                // onActivated: 
                // Set the initial currentIndex to the value stored in the backend.
                // Component.onCompleted: currentIndex = indexOfValue(backend.modifier)
                model: [
                    { value: Qt.NoModifier, text: qsTr("camera") },
                    { value: Qt.ShiftModifier, text: qsTr("video") },
                    { value: Qt.ControlModifier, text: qsTr("image directory") }
                ]
                background: Rectangle {
                    implicitWidth: 130
                    implicitHeight: 40
                    color: "#353637"
                    border.color: "#21be2b"
                }
                onActivated: {
                    choose_video_or_image.enabled = combox.currentText=="camera"? false:true
                    video_or_image.enabled = combox.currentText=="camera"? false:true
                }
            }

            TextField {
                id: input_or_choose_mode_path
                // placeholderText: qsTr("yolo11n.pt")
                // placeholderTextColor: "#ffffff"
                color: "#ffffff"
                text: "yolo11n.pt"

                background: Rectangle {
                    implicitWidth: 160
                    implicitHeight: 40
                    color: "#353637"
                    border.color: "#21be2b"
                }
            }

            Button { // 按钮
                id: model
                text: "model"
                highlighted: true
                Material.accent: Material.Red
                onClicked: { // 点击以后调用槽函数slot
                    // bridge.open()
                    fileDialog_model.open()
                }
            }
            TextField {
                id: choose_video_or_image
                placeholderText: qsTr("video or images directory")
                placeholderTextColor: "#ffffff"
                color: "#ffffff"
                text: ""
                enabled: false
                background: Rectangle {
                    implicitWidth: 160
                    implicitHeight: 40
                    color: "#353637"
                    border.color: "#21be2b"
                }
            }
            Button {
                id: video_or_image
                text: "video or image"
                highlighted: true
                enabled:false
                Material.accent: Material.Green
                onClicked: { // 点击以后调用槽函数slot
                    combox.currentText=="video"? fileDialog_video_image.open():folderDialog.open()
                }
            }
        }
        RowLayout {
            id: rightcolumn2
            spacing: 2
            // Layout.columnSpan: 1
            // Layout.preferredWidth: 400
            // Layout.preferredHeight: 400
            Layout.fillWidth: true

            Layout.alignment: Qt.AlignLeft

            Button { // 按钮
                id: media_start
                text: "media start"
                highlighted: true
                Material.accent: Material.Red
                onClicked: { // 点击以后调用槽函数slot
                    my_image_provider.get_type(combox.currentText)
                    my_image_provider.get_model(input_or_choose_mode_path.text)
                    my_image_provider.get_video(choose_video_or_image.text)
                    combox.enabled = false
                    stop.enabled = true
                    media_start.enabled = false
                    model.enabled = false
                    input_or_choose_mode_path.enabled = false
                    choose_video_or_image.enabled = false
                    video_or_image.enabled = false
                    control.enabled = false
                    my_image_provider.start()
                }
            }

            Button {
                id: stop
                text: "stop"
                highlighted: true
                Material.accent: Material.Green
                onClicked: { // 点击以后调用槽函数slot
                    combox.enabled = true
                    media_start.enabled = true
                    model.enabled = true
                    input_or_choose_mode_path.enabled = true
                    control.enabled = true
                    choose_video_or_image.enabled = combox.currentText=="camera"? false:true
                    video_or_image.enabled = combox.currentText=="camera"? false:true
                    my_image_provider.stop()
                    stop.enabled = false
                }
            }
            CheckBox {
                id: control
                text: qsTr("save video")
                checked: false
                background: Rectangle {
                    implicitWidth: 60
                    implicitHeight: 30
                    visible: control.down || control.highlighted
                    color: control.down ? "#bdbebf" : "#00ee00"
                }
                onClicked: {
                    my_image_provider.get_checked(control.checked)
                }
            }
        }
    }
}
