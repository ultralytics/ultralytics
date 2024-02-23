from ultralytics import YOLO


model_name = '8s-100e-128b-auto'

# Initialize model and load matching weights
model = YOLO('./models/'+model_name+'.pt')

metrics = model.val(
    data='custom_dataset.yaml',
    imgsz=640,
    batch=16,
    device=[3],
    verbose=True,
    save=True,
    save_json=True,
    plots=True,
    # save results to project/name relative to script directory or absolute path
    project='validations',
    name=model_name,
)