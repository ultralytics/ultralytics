from ultralytics import YOLO
from ultralytics import RTDETR

# 1. Load the YOLO11 model (n = nano, s = small, etc.)
# It will automatically download 'yolo11n.pt' if not present.
model = YOLO('yolo11l.pt')
# model = RTDETR('rtdetr-resnet50.yaml')
# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50_pretrained_timm.yaml')
# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50_pretrained.yaml')
# model = RTDETR('rtdetr-l')
# model = RTDETR('ultralytics/cfg/models/11/yolo11-rtdetr_p2_l3_efms.yaml')
# model = RTDETR('ultralytics/cfg/models/11/yolo11l-rtdetr.yaml')
# model.load('yolo11n.pt')
# model = RTDETR('ultralytics/cfg/models/rt-detr/rtdetr-resnet50d_gluon_in1k_timm.yaml')
# model.load('yolo11n.pt')
# model = YOLO('ultralytics/cfg/models/11/yolo11.yaml')
# model.load('yolo11n-cls.pt')
# model = YOLO("https://ultralytics.com/assets/yolo11n.pt")


# 2. Run inference on a source (can be a file path, URL, or '0' for webcam)
# We use a standard URL image for this example.
results = model('https://ultralytics.com/images/bus.jpg')

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
    r.save(filename=f'result_{i}.jpg') # Saves the image to disk
    print(f"\nSaved visualization to 'result_{i}.jpg'")