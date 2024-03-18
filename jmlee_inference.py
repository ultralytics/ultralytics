from ultralytics import YOLO
import cv2
import os
from ultralytics.utils.metrics import plot_pr_curve, ap_per_class
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.ops import xywh2xyxy
import torch
model = YOLO("runs/detect/train_val_yolo8s_default/weights/best.pt")
# validator = model.validator()
# target_cls = validator.stats.get('target_cls', None)
# results = model.predict(source="pe_module_24_1_26/images/test", show=False)
results = model.predict(source="pe_module_24_1_26/images/test/20231212_112240.mp4_frame90.jpg", show=False)

# for idx, frame in enumerate(results):
img_path = os.path.join("./test", 'img.png')
cv2.imwrite(img_path, results[0].plot(font_size=0.1))


#     cv2.imwrite(img_path, frame.plot(font_size=0.1))

# label_path = "./pe_module_24_1_26/labels"
# labels = convert_labels_to_yolo_obb(label_path)

# target_cls = []
# for filename in os.listdir(label_path):
#     if filename.endswith(".txt"): 
#         file_path = os.path.join(label_path, filename)
#         with open(file_path, "r") as f:
#             lines = f.readlines()
#             for line in lines:
#                 parts = line.strip().split()
#                 class_index = int(parts[0])
#                 x1, y1, x2, y2 = xywh2xyxy(map(float, parts[1:]))
#                 target_cls.append([class_index, x1,y1,x2,y2])
# def process_label_file(file_path):
#   label_cls = []
#   with open(file_path, "r") as f:
#     for line in f:
#       print("f.name")
#       print(f.name)

#       parts = line.strip().split()
#       class_index = int(parts[0])
#       print("class_index")
#       print(class_index)

#       coordinates = list(map(float, parts[1:]))  # Convert strings to floats
#       print("coordinates")
#       print(coordinates)
#       x1, y1, x2, y2 = xywh2xyxy(coordinates)  # Pass coordinates as a list
#       label_cls.append([class_index, x1, y1, x2, y2])
#   return label_cls

# target_cls = []
# for filename in os.listdir(label_path):
#   if filename.endswith(".txt"):
#     file_path = os.path.join(label_path, filename)
#     data = process_label_file(file_path)
#     target_cls.extend(data)
                
# print(target_cls[0])

# for idx, frame in enumerate(results):
#     ap_per_class(frame.boxes.conf, frame.boxes.cls, target_cls[idx], plot=True)

