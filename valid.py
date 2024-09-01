# Author: Zenghui Tang
# Date: 2024/3/26
# Time: 13:50

from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s.pt")  # pretrained YOLOv8n model

# Run batched inference on a list of images
results = model([r"E:\04_ClassicNet\yolov8\ultralytics\assets\bus.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.show()  # display to screen
    # result.save(filename='result.jpg')  # save to disk


if __name__ == "__main__":
    pass
