### This is a sample implementation of YOLO Pose detection model on iOS natively using ONNX

#### Steps to run the app

- clone/fork this repo
- Download the yolov8_pose_e2e model from [the official onnx link](https://onnxruntime.ai/docs/tutorials/mobile/pose-detection.html)
- Run the yolov8_pose_e2e.py command with a test image to download the full onnx model.
- Include the model in the build settings of this app by copying the same.
- Build and run

<p>
The app is currently tested for one image. There are several areas of improvement as indicated in the code comments
Performance isn't that great yet, inference takes around 500-600 milliseconds or more.
</p>
