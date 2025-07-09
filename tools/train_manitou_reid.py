from ultralytics import YOLOManitou_MultiCam
from ultralytics.models import yolo_manitou

project = "runs/manitou_remap"
# data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou.yaml"
data_cfg = "/root/workspace/ultralytics/ultralytics/cfg/datasets/manitou.yaml"
epochs = 80
batch_size_per_gpu = 1
device = [1, 2, 3]  # list of GPU devices
batch_size = batch_size_per_gpu * len(device)  # total batch size
imgsz = (1552, 1936)  # (height, width)
# weights = "/datasets/dataset/best.pt"
weights = "yolo11s.pt"
model = YOLOManitou_MultiCam("yolo11s.yaml").load(weights)  # load a model
results = model.train(
    data=data_cfg,
    epochs=epochs,
    imgsz=imgsz,
    trainer=yolo_manitou.detect_multiCam.ManitouTrainer_MultiCam,
    batch=batch_size,
    device=device,
    project=project,
)  # train the model
