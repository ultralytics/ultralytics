# YOLOv8-QT-PySide6-QML

Creating a GUI interface by using the pyside6's QML, not qtwidgets. The input can be a camera、a video or a directory which contains many pictures. You can also save the output video.

<img src='.\2.jpg' width = 100%/>

## Usage

### Run

- `pip install ultralytics`
- `pip install pyside6`

```python []
cd YOLOv8-QT-PySide6-QML
python main.py
```

### 1、camera input

- choose camera
- set a model, like yolov8n.pt, yolov8m-pose.pt, etc
- press the button **"media start"** to start
- press the button **"stop"** to stop
- choose the **"save video"** to save the video

### 2、video input

- choose video
- set a model, like yolov8n.pt, yolov8m-pose.pt, etc
- set **"video or image"** to choose a video source, like _.avi, _.mp4 and so on.
- press the button **"media start"** to start
- press the button **"stop"** to stop
- choose the **"save video"** to save the video

### 3、images input

- choose image directory
- set a model, like yolov8n.pt, yolov8m-pose.pt, etc
- set **"video or image"** to choose a image directory
- press the button **"media start"** to start
- press the button **"stop"** to stop
- choose the **"save video"** to save the video

## frequency per second

the saved video save all frame, no frame is missing, so you can try adjust the frequency per second.
