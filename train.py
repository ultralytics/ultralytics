import datetime

from ultralytics import YOLO

time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

model_name = 'yolov8n-baseline'
epochs = 200
batch_size = 32
patience = 50
img_size = 608
optimizer = 'SGD'

model = YOLO(f'ultralytics/cfg/models/military/{model_name}.yaml')

model.train(
    data='military.yaml',
    epochs=epochs,
    patience=patience,
    batch=batch_size,
    imgsz=img_size,
    project='military',
    name=f'{model_name}-{img_size}-{optimizer}-{epochs}-{time}',
    exist_ok=True,
    optimizer=optimizer,
    verbose=True,
    cos_lr=True,
    close_mosaic=200,
)
