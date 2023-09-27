from ultralytics import YOLO

# Load a model
# model = YOLO('yolov8n.pt')  # load a pretrained model (recommended for training)
model = YOLO('yolov8x.pt')  # load a pretrained model (recommended for training)

# Train the model with 2 GPUs
results = model.train(data='noaa_10000.yaml', epochs=100, imgsz=640, batch=16, device=[0, 1], name='noaa_10000(yolov8x)')