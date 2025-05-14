from ultralytics import YOLO, YOLOManitou
from ultralytics.models import yolo_manitou

data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou.yaml" 

model = YOLOManitou('yolo11s.yaml').load('yolo11s.pt')  # load a model

epochs = 1
batch_size_per_gpu = 8
device = [0,]  # list of GPU devices
batch_size = batch_size_per_gpu * len(device)  # total batch size
imgsz = (1552, 1936)  # (height, width)

# results = model.train(data=data_cfg, epochs=epochs, imgsz=imgsz, trainer=yolo_manitou.detect.ManitouTrainer, batch=batch_size, device=device)  # train the model


# # Test the validation
# batch_size = 16
# imgsz = (1552, 1936)  # (height, width)
# checkpoint = '/home/shu/Documents/PROTECH/ultralytics/runs/detect/train/weights/best.pt'
# model = YOLOManitou(checkpoint)  # load a model
# metrics = model.val(data=data_cfg, imgsz=imgsz, conf=0.25, batch=batch_size)



# Test the prediction
import glob
path = '/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/key_frames/rosbag2_2024_11_26-10_38_53/camera1/'
checkpoint = '/home/shu/Documents/PROTECH/ultralytics/runs/detect/train/weights/best.pt'
model = YOLOManitou(checkpoint)
results = model.track(source=path, imgsz=imgsz, conf=0.25, max_det=100, device=device, save_frames=True, tracker="bytetrack.yaml")  # predict the model

for result in results:
    boxes = result.boxes.xyxy.cpu().numpy()  # get the bounding boxes
    confs = result.boxes.conf.cpu().numpy()  # get the confidence scores
    cls = result.boxes.cls.cpu().numpy()  # get the class labels
    print(f"Boxes: {boxes}, Confidences: {confs}, Class Labels: {cls}")
    # result.save(font_size=0.8, line_width=2)  # save the results