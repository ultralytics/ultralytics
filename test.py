from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("./runs/detect/train/weights/best.pt")
    model.val(data="LROC.yaml", split="test", epochs=200, imgsz=608)