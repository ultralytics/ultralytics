from ultralytics import YOLO


if __name__ == '__main__':
    # for i in range(3):
    #     print(f"第{i+1}次训练")
    model = YOLO('yolov8n.yaml').load('yolov8n.pt')
    results = model.train(data="crater.yaml",epochs=200,imgsz=640)
