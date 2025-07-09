from ultralytics import YOLOManitou_MultiCam
from ultralytics.models import yolo_manitou

project = "runs/manitou_remap_mini"
data_cfg = "/home/shu/Documents/PROTECH/ultralytics/ultralytics/cfg/datasets/manitou_mini.yaml"
epochs = 80
batch_size_per_gpu = 2
device = [
    0,
]  # list of GPU devices
batch_size = batch_size_per_gpu * len(device)  # total batch size
imgsz = (1552, 1936)  # (height, width)
max_det = 100
ref_img_sampler = dict(scope=5, num_ref_imgs=1, method="uniform")
use_radar = True

# model = YOLOManitou('yolo11s.yaml').load('yolo11s.pt')  # load a model
model = YOLOManitou_MultiCam("/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train/weights/best.pt")
results = model.train(
    data=data_cfg,
    epochs=epochs,
    imgsz=imgsz,
    max_det=100,
    use_radar=use_radar,
    ref_img_sampler=ref_img_sampler,
    trainer=yolo_manitou.detect_multiCam.ManitouTrainer_MultiCam,
    batch=batch_size,
    device=device,
    project=project,
)  # train the model


# # Test the validation
# batch_size = 64
# imgsz = (1552, 1936)  # (height, width)
# checkpoint = ''
# model = YOLOManitou(checkpoint)  # load a model
# metrics = model.val(data=data_cfg, imgsz=imgsz, batch=batch_size)
