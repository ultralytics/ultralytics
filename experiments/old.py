from pathlib import Path

import matplotlib.pyplot as plt
import torch
from matplotlib import image as mpimg

from ultralytics import YOLO
# from ultralytics import YOLO
from ultralytics.models.yolo.segment.train import SegmentationTrainer
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.build import build_dataloader, build_yolo_dataset
from ultralytics.models.yolo.segment.train import SegmentationTrainer
import matplotlib.pyplot as plt
from PIL import Image


st = SegmentationTrainer(cfg="args.yaml", overrides=dict(data="dataset.yaml", device="cpu"))
# st._setup_train(10)

dataloader = st.get_dataloader("/Users/thomas/Documents/VBTI/DataAnalysis/autodl/dataset/yolo-converted/cucumber_dataset/train/images", batch_size=1)


model = YOLO("/Users/thomas/Documents/School/TU:e/1. Master/Year 3/Graduation/Preparation Phase/Data Analysis/models/vdl-smart-trim-s-0-models-20230726-100018__yolo8m_imgsize_960/weights/best.pt")

for i in range(5):
    # Get a batch of training data
    data = next(iter(dataloader))

    inputs = data['im_file'][0]

    # Forward pass the data through the model
    # Perform object detection on an image using the model
    results = model(inputs)

    images = []
    # Show the results
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        im.show()  # show image
        # im.save('results.jpg')  # save image
        # images.append(im)


# # Perform object detection on an image using the model
# results = model('https://ultralytics.com/images/bus.jpg')
#
# # Show the results
# for r in results:
#     im_array = r.plot()  # plot a BGR numpy array of predictions
#     im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
#     im.show()  # show image
#     im.save('results.jpg')  # save image
#










# # Define what device we are using
# print("mps Available: ",torch.has_mps)
# device = torch.device("cpu")
#
# train_features = next(iter(dataloader))
# print(train_features)
# print(train_features['im_file'])
# img = plt.imread(train_features['im_file'][0])
# fig, ax = plt.subplots()
# img = ax.imshow(img)
# plt.show()
#
# print(train_features['masks'])


# # ds = build_yolo_dataset("dataset.yaml")
#
# ds = YOLODataset("/Users/thomas/Documents/VBTI/DataAnalysis/autodl/dataset/yolo-converted/cucumber_dataset/train/images", data="dataset.yaml", use_segments=True)
#
# # dl = YOLODataset.buil
#
#
# # Load a pretrained YOLO model (recommended for training)
# model = YOLO('args.yaml', task='segment').load("/Users/thomas/Documents/School/TU:e/1. Master/Year 3/Graduation/Preparation Phase/Data Analysis/models/vdl-smart-trim-s-0-models-20230726-100018__yolo8m_imgsize_960/weights/best.pt")
# data = "dataset.yaml"
# print(dir(model))
