import os.path

import cv2

from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train33/weights/best.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
images = [
    "/home/lixiang/PycharmProjects/ultralytics/rotate_0.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_1.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_2.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_3.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_4.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_5.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_6.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_7.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_8.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_9.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_10.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_11.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_12.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_13.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_14.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_15.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_16.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_17.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_18.jpg",
    "/home/lixiang/PycharmProjects/ultralytics/rotate_19.jpg",
]

img_names = [x.split("/")[-1].split(".")[0] for x in images]

for idx, image in enumerate(images):
    results = model.predict(image, imgsz=640, conf=0.5)  # return a list of Results objects

    # Process results list
    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        result.names = {
            0: "人行道路面破损",
            1: "沿街晾晒",
            2: "垃圾满冒",
            3: "乱扔垃圾",
            4: "垃圾正常盛放",
        }
        for cls, box in zip(result.boxes.cls, result.boxes.xyxyn):
            cls_np = cls.cpu().detach().numpy().squeeze()
            box_np = box.cpu().detach().numpy().squeeze()
        result.save(filename=f"道路破损/{img_names[idx]}.jpg")  # save to disk
