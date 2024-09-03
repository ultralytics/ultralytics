This is a fork from the original ultralytics package. The original package is a great tool for object detection and image classification. The original package is still available at 
https://github.com/ultralytics/ultralytics

While the original package is great for object detection, its not suited for detection on multispectral images. This package is designed to work with multispectral images, and it is also designed to work with the new version of the PyTorch library. The original package is designed to work with the old version of the PyTorch library.
I have adjusted some of the code to be able to handle a custom defiend number of bands. Users can choose to train from scratch or to initialize the weights from a pre-trained model. The package is also designed to work with the new version of the PyTorch library. When initialising from a pretrained checkpoint with a diffrent number of input channels, the package uses a uniform xavier to initialize the missing weights.

Sofar the following models and tasks have been adjusted to work with multispectral images:
- YOLO models for the tasks (classification) and detection. (tested for yolov8, yolov9 and yolov10 models)
- Others might work, but have not been tested yet.

- Other tasks are currently broken and will be part of future efforts to fix them.

Additionaly the support of Siamese and Dual_stream yolo models have been included. 
- A config for a siamese YOLOv9e model has been included. Futher configs will be add in the future.

As this package is currently subject to heavy changes please expect the code to be unstable.
I would advise to use a fixed version of the package to avoid any issues. 





