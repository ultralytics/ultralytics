# YOLO11-QT-PySide6-QML

Creating a GUI interface by using the pyside6's QML, not qtwidgets. The input can be a camera、a video or a directory which contains many pictures. You can also save the output video.

<img src='https://private-user-images.githubusercontent.com/26833433/356584949-9a566688-7ce8-41ef-8ed8-24c9dba1ac37.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MjMzMzg3NTIsIm5iZiI6MTcyMzMzODQ1MiwicGF0aCI6Ii8yNjgzMzQzMy8zNTY1ODQ5NDktOWE1NjY2ODgtN2NlOC00MWVmLThlZDgtMjRjOWRiYTFhYzM3LnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA4MTElMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwODExVDAxMDczMlomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTEzOGVkYjA5OGFjYTFjY2U4NDFlYTEwMzkxNzExYmJjNTc5NjgwMDQyMGY3YWU4ZTRmYTBiNmRmYjM2ZGU3N2UmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.mwf4hmyVf8uVRBAAaUnd2BjbWOLAv57k6f58kS9GYwg' width = 100%/>

## Usage

### Run

- `pip install ultralytics`
- `pip install pyside6`

```python []
cd YOLO11-QT-PySide6-QML
python main.py
```

### 1、camera input

- choose camera
- set a model, like yolo11n.pt, yolo11m-pose.pt, etc
- press the button **"media start"** to start
- press the button **"stop"** to stop
- choose the **"save video"** to save the video

### 2、video input

- choose video
- set a model, like yolo11n.pt, yolo11m-pose.pt, etc
- set **"video or image"** to choose a video source, like _.avi, _.mp4 and so on.
- press the button **"media start"** to start
- press the button **"stop"** to stop
- choose the **"save video"** to save the video

### 3、images input

- choose image directory
- set a model, like yolo11n.pt, yolo11m-pose.pt, etc
- set **"video or image"** to choose a image directory
- press the button **"media start"** to start
- press the button **"stop"** to stop
- choose the **"save video"** to save the video

## frequency per second

the saved video save all frame, no frame is missing, so you can try adjust the frequency per second.
