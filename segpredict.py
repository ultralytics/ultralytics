from ultralytics import FastSAM
import cv2

# Load a model
# model = YOLO('yoloxlast.pt')  # load a custom model
model = FastSAM('last.pt')
# model = YOLO('yolox_15_1024/last.pt')
# model = YOLO('yolos_640/yolos_23_640_10e.pt')
# Predict with the model

# results = model('cake.png', device='4', retina_masks=True, imgsz=1024, conf=0.01, iou=0.9,)  # predict on an image
everything_results = model('sa_548233.jpg', device='0', retina_masks=True, imgsz=1024, conf=0.25, iou=0.7,) 
# 为了画图改了ultralytics/yolo/engine/results.py 第221行
image = everything_results[0].plot(labels=False, boxes=False, masks=True, probs=False)
# cv2.imwrite("yolox_20_1024_overlap_e33_street_pred_1024_iou0.9.jpg", image)
cv2.imwrite("test9.jpg", image)