# 📚 README TO RUN TRACKER 🚀
This README provides instructions on how to create a Docker image,
export your model to onnx format and how to run the tracker in video data.


Welcome to the README for running the Tracker 📹!. Here you can find a guide 
to create a Docker image, exporting your model to ONNX format, and running 
the tracker on video data. Let's get started!

## 🐳 In Docker Image
### 👨🏽‍💻 Prerequisites 

Before diving into the Docker image setup, make sure you have the following
prerequisites in place:

- Docker installed 🐋
- Your favorite code editor ready 📝

### 👨🏽‍💻 Steps 

### 1. Build the Docker image using the following command:

```bash
docker build -t tracker_onnx -f Dockerfile.tracker . --platform linux/amd64
```

### 2. Run the Docker image with volumes using the following command:
```bash
sudo docker run -it -v /Users/ejbejaranos/Documents/Edison/2023/UB/inference_tools/models:/app/models tracker_onnx
```


## 💻 In Local Environment
To run the Tracker in your local environment, follow these steps:

Follow the steps below to build and run the Docker image:
### 👨🏽‍💻 Prerequisites 
- PyCharm or VsCode.
- Python3.8
### Download Video for Testing Tracker
To download the video for testing the tracker, you can use the following command:


Install the library:
```python
pip install -q gdown 
```
Execute this code into your terminal
```bash
gdown -O "./tracker/data/traffic_analysis.mov" "https://drive.google.com/uc?id=1qadBd7lgpediafCpL_yedGjQPk-FLK-W"
```

