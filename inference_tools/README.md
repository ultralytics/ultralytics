# README TO RUN INFERENCE 🚀
This README provides instructions on how to run inference and export using a Docker image. Follow the steps below to build and run the Docker image:
## 👨🏽‍💻 Prerequisites 
- Docker installed on your system.

## 👻 Steps
### 1. Build the Docker image using the following command:
```bash
docker build -t detector_onnx .
```
This command builds a Docker image from the Dockerfile in the current directory and tags the image as detector_onnx.
If you are running the Docker image in Apple M1/M2 processor, you should use the following command to emulate an AMD64 
processor (emulation is much slower than native execution):
```bash
docker build -t detector_onnx . --platform linux/amd64
```
The image is build using the requirements.txt file located inside /inference_tools folder. This file matches the specified software 
requirements made by the client.
### 2. Run the Docker image using the following command:
``` bash
sudo docker run -it -v absolute_path_to/inference_tools/models:/app/models -v absolute_path_to/inference_tools/outputs:/app/outputs detector_onnx
```
Notice that two volumes are mounted in the container to store the outputs and the models produced by both inference 
(infer.py) and exportation (export.py). The first volume is mounted in the /app/models directory of the container and 
the second volume is mounted in the /app/outputs directory of the container. The absolute paths to the models and outputs 
directories in your system should be provided in the command.

This command starts a new container from the onnx image and opens an interactive terminal session (-it option). 
Automatically the started container runs the exportation script (export.py) and then the inference script (infer.py).

Firstly exports the /inference_tools/models/custom_best.pt model to /inference_tools/models/custom_best.onnx.
For a proper inference, the model should have been exported to ONNX format using the same software version
as the one used to run inference. This is because the ONNX format is not backward compatible.

Then inference is performed on the images located in the /inference_tools/images/test directory and stores the outputs 
in the /outputs directory. The outputs are: original images with bounding boxes, a text file with detection results for 
each image in YOLO format and a csv with inference time for each image.