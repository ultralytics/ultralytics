from ultralytics import RTDETR
model = RTDETR('rtdetr-x.pt')
results = model.train(data="coco8.yaml", epochs=1000, cache=True, batch=24, imgsz=896, device='cpu', exist_ok=True, patience=100, workers=2)