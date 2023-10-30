---
comments: true
description: Find solutions to your common Ultralytics YOLO related queries from our Youtube Videos. Learn about hardware requirements, fine-tuning YOLO models, Export and more.
keywords: Ultralytics, YOLO, FAQ, Object Detection, Image Classification, Object Tracking, Object Segmentation, YOLO accuracy
---

# Ultralytics YOLO Frequently Asked Questions (FAQ) YouTube

This FAQ section addresses some common questions and issues users might encounter while following Ultralytics YouTube Videos.

#### 1. Extract and Use Detection Data?
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
results = model('https://ultralytics.com/images/bus.jpg')

# Extract bounding boxes, classes, names, and confidences
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names
confidences = results[0].boxes.conf.tolist()

# Iterate through the results
for box, cls, conf in zip(boxes, classes, confidences):
    x1, y1, x2, y2 = box
    name = names[int(cls)]
    #..... continue according to your need
```

#### 2. Run Image Classification Using Ultralytics YOLOv8 With WebCam?
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load an official model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image
```

#### 3. Print Top 1 Class Accuracy Using Ultralytics YOLOv8 Image Classification
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-cls.pt')  # load an official model

# Predict with the model
results = model('https://ultralytics.com/images/bus.jpg')  # predict on an image

names = results[0].names
print("Top 1 Class Name : {}".format(names[results[0].probs.top1]))
print("Top 1 Class Accuracy : {}".format(results[0].probs.top1conf.cpu()))
```

#### 4. Object Tracking Using Ultralytics YOLOv8
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Object tracking on image
results = model.track(source='https://ultralytics.com/images/bus.jpg', persist=True, show=True)
```

#### 5. Export Ultralytics YOLOv8 Model Into Different Formats
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

model.export(format="onnx")  # onnx
model.export(format="openvino")  # openvino
model.export(format="engine")  # engine
model.export(format="saved_model")  # saved_model
model.export(format="tflite")  # tflite
model.export(format="ncnn")  # ncnn
```

#### 6. Extract KeyPoints using Ultralytics YOLOv8-Pose
```python
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

results = model.predict(source='https://ultralytics.com/images/bus.jpg')
key_points = results.keypoints.data  # KeyPoints Data

# Iterate over each keypoint
for ind, kpt in enumerate(reversed(key_points)):
    print(kpt)  # k will include every keypoint of object

    # ...later you can use this according to your needs
```

#### 7. Security Alarm System Project Using Ultralytics YOLOv8
```python
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Set up the parameters of the message
password = ""
from_email = ""
to_email = ""

# create server
server = smtplib.SMTP('smtp.gmail.com: 587')
server.starttls()

# Login Credentials for sending the mail
server.login(from_email, password)

def send_email(to_email, from_email, people_detected=1):
    message = MIMEMultipart()
    message['From'] = from_email
    message['To'] = to_email
    message['Subject'] = "Security Alert"
    # add in the message body
    message.attach(MIMEText(f'ALERT - {people_detected} persons has been detected!!', 'plain'))
    server.sendmail(from_email, to_email, message.as_string())


class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.email_sent = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.annotator = None

    def load_model(self):
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        class_ids = []
        self.annotator = Annotator(frame, 3, results[0].names)
        boxes = results[0].boxes.xyxy.cpu()
        clss = results[0].boxes.cls.cpu().tolist()
        names = results[0].names
        for box, cls in zip(boxes, clss):
            class_ids.append(cls)
            self.annotator.box_label(box, label=names[int(cls)], color=colors(int(cls), True))
        return frame, class_ids

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        frame_count = 0
        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret
            results = self.predict(frame)
            frame, class_ids = self.plot_bboxes(results, frame)
            if len(class_ids) > 0:
                if not self.email_sent:  # Only send email if it hasn't been sent for the current detection
                    send_email(to_email, from_email, len(class_ids))
                    self.email_sent = True  # Set the flag to True after sending the email
            else:
                self.email_sent = False  # Reset the flag when no person is detected

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Detection', frame)
            frame_count += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break
        cap.release()
        cv2.destroyAllWindows()
        server.quit()

detector = ObjectDetection(capture_index=0)
detector()
```

#### 8. Object Segmentation with Ultralytics YOLOv8
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n-seg.pt')  # load an official model

# Object segmentation on image
results = model.predict(source='https://ultralytics.com/images/bus.jpg')
```

#### 9. How to change hyperparameter values?
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8s.pt')  # load an official model

# Model training with hyperparameters tuning
results = model.train(data="coco.yaml", lr0=0.01, weight_decay=0.0005)
```

* For more information, you can check YOLOv8 training arguments: https://docs.ultralytics.com/modes/train/#arguments

#### 10. Does Ultralytics YOLOv8 research paper exist?
There is no official paper on YOLOv8, but rather a series of improvements and extensions made by Ultralytics to the YOLOv5 architecture. Most of the changes made in YOLOv8 relate to model scaling and architecture tweaks, which can be found in the code and the documentation in the Ultralytics YOLOv8 repository. For more information, you can visit mentioned links.

* Ultralytics GitHub: https://github.com/ultralytics/ultralytics
* Ultralytics Docs: https://docs.ultralytics.com/

#### 11. How to detect specific classes with Ultralytics YOLOv8?
```python
from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.pt')  # load an official model

# Object detection on image
results = model.predict(source='https://ultralytics.com/images/bus.jpg', classes=2)
```

#### 12. What Pose structure Ultralytics YOLOv8 Pose follows?
![204121854-ba14ce4c-d9fd-4244-92b7-038ef93199af](https://github.com/RizwanMunawar/ultralytics/assets/62513924/e3802057-fd8e-4826-a96e-2b49e6414549)

#### 13. Ultralytics YOLOv8 License Information
Any project that incorporates Ultralytics models, architecture, or code, whether modified or not, requires open-sourcing under the AGPL license. This means that if you're using a YOLO model from Ultralytics in your project and distributing it for commercial purposes, you are obligated to make the entire project open-source, even if you haven't made any changes to the original code or datasets.

We understand licensing can be complex, so we highly recommend consulting with legal counsel to ensure full compliance with AGPL and any other relevant licenses, especially when considering commercial applications.
