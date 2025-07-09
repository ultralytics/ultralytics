from ultralytics import YOLOManitou

# Test the prediction
imgsz = (1552, 1936)  # (height, width)
# path = '/home/shu/Documents/PROTECH/ultralytics/datasets/manitou_mini/data/rosbag2_2025_02_17-14_04_34/camera1'
path = "/home/shu/Documents/PROTECH/ultralytics/datasets/manitou/key_frames/rosbag2_2025_02_17-14_04_34/camera1/"
checkpoint = "/home/shu/Documents/PROTECH/ultralytics/runs/manitou_remap/train/weights/best.pt"
model = YOLOManitou(checkpoint)

# results = model.track(source=path, imgsz=imgsz, conf=0.25, max_det=100, save_frames=True, tracker="bytetrack.yaml")
# for result in results:
#     # boxes = result.boxes.xyxy.cpu().numpy()  # get the bounding boxes
#     # confs = result.boxes.conf.cpu().numpy()  # get the confidence scores
#     cls = result.boxes.cls.cpu().numpy()  # get the class labels
#     track_ids = result.boxes.id.cpu().numpy()  if result.boxes.id is not None else []

#     print(f" Class Labels: {cls}, Track IDs: {track_ids}")  # print the results
#     result.save(font_size=0.8, line_width=2)  # save the results

from pathlib import Path

path = Path(path).glob("*.jpg")
path = sorted(path)  # sort the path to get the correct order
for p in path:
    p = str(p)
    result = model.track(
        source=p, imgsz=imgsz, conf=0.25, max_det=100, save_frames=True, tracker="bytetrack.yaml", persist=True
    )[0]
    result.save(font_size=0.8, line_width=2)  # save the results
