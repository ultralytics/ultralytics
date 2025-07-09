from ultralytics import YOLOManitou
from ultralytics.models import yolo_manitou

project = "runs/manitou_remap"
data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou_mini.yaml"
epochs = 1
batch_size_per_gpu = 8
device = [
    0,
]  # list of GPU devices
batch_size = batch_size_per_gpu * len(device)  # total batch size
imgsz = (1552, 1936)  # (height, width)

model = YOLOManitou("yolo11s.yaml").load("yolo11s.pt")  # load a model
results = model.train(
    data=data_cfg,
    epochs=epochs,
    imgsz=imgsz,
    trainer=yolo_manitou.detect.ManitouTrainer,
    batch=batch_size,
    device=device,
    project=project,
)  # train the model


# # Test the validation
# batch_size = 64
# imgsz = (1552, 1936)  # (height, width)
# checkpoint = '/home/shu/Documents/PROTECH/ultralytics/runs/detect/train6/weights/best.pt'
# model = YOLOManitou(checkpoint)  # load a model
# metrics = model.val(data=data_cfg, imgsz=imgsz, batch=batch_size)
