from ultralytics import YOLO
from ultralytics import RTDETR
from ultralytics.utils import ROOT, YAML
import torch

# model = YOLO('ultralytics/cfg/models/26/yolo26l.yaml')
# model.load('yolo26l.pt')

# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
# model.load('rtdetr-l.pt')
# model = RTDETR('rtdetr-l')

# Yolo26-rtdetr results
# model = RTDETR('ultralytics/cfg/models/26/yolo26l-rtdetr.yaml')
# model.load('/Users/esat/workspace/runs/rtdetr_yolo26l_PObj_origaugV2_imgsz640_epc90_clsmos15_lrf05/weights/best.pt')
# # Load COCO class names from the dataset config
# coco_names = YAML.load(ROOT / "cfg/datasets/coco.yaml").get("names")
# model.model.names = coco_names  # Set names on the underlying model object

# Yolo26n-rtdetr results
model = RTDETR('ultralytics/cfg/models/26/yolo26n-rtdetr_p4_l3_efms_365.yaml')
# model = YOLO('ultralytics/cfg/models/26/yolo26n.yaml')
model.load('/Users/esat/workspace/rtdetrLightp4_yolo26n_scratch_wu1_lr4x_origaugV2_150epc/weights/best.pt')
# model.load("yolo26n-objv1-150-detr.pt")
# Load COCO class names from the dataset config
obj365_names = YAML.load("working_dir/datasets/Objects365v1.yaml").get("names")
model.model.names = obj365_names  # Set names on the underlying model object


# ckpt = torch.load("yolo26s-objv1-150.pt", weights_only=False)
# # ckpt = torch.load("yolo26l.pt", weights_only=False)
# train_args = ckpt.get("train_args")
# print(ckpt["train_args"])

# 2. Run inference on a source (can be a file path, URL, or '0' for webcam)
# We use a standard URL image for this example.
results = model('https://ultralytics.com/images/bus.jpg', conf=0.50)

# 3. Iterate through results (usually a list, one per image)
for i, r in enumerate(results):
    print(f"\n--- Debugging Image {i+1} ---")

    # 'r.boxes' contains the detection data
    # .data gives you the raw tensor: [x1, y1, x2, y2, conf, class_id]
    print(f"Total Detections: {len(r.boxes)}")
    
    # Iterate over each box to print specific debug info
    for box in r.boxes:
        # Get coordinates (x1, y1, x2, y2)
        # .cpu().numpy() converts the tensor to a readable numpy array
        coords = box.xyxy[0].cpu().numpy() 
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = r.names[cls_id] # Map ID to class name
        
        print(f"Object: {cls_name} | Conf: {conf:.2f} | Box: {coords}")

    # OPTIONAL: Visualize the result
    # r.show()  # Opens a window with the image
    r.save(filename=f'working_dir/results/result_{i}.jpg') # Saves the image to disk
    print(f"\nSaved visualization to 'working_dir/results/result_{i}.jpg'")