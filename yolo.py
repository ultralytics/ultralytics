import os
from glob import glob
from ultralytics import YOLO

# Load a model
yolo = YOLO("./models/my_yolov8s.pt")  # load your model (recommended for training)

# Define the directory containing the images
images_dir = 'images/'

# Loop through the images folder
for file in glob(os.path.join(images_dir, '*.png')):
    # Predict on each image
    res = yolo.predict(source=file, save=True) # save annotated images to the runs/detect/prediction
