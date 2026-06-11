// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include <iostream>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>

#include "inference.h"
#include "yolo_draw.hpp"
#include "yolo_show.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
    std::string projectBasePath = "/home/user/ultralytics"; // Set your ultralytics base path

    bool runOnGPU = false;  // requires an OpenCV build with the CUDA DNN backend
    const bool show = yolo::ShowRequested(argc, argv);  // pass --show to display the result

    //
    // Pass in either:
    //
    // "yolo11s.onnx" or "yolov5s.onnx"
    //
    // To run Inference with Ultralytics YOLO11/YOLOv8 or YOLOv5 (ONNX)
    //

    // Note that in this example the classes are hard-coded and 'classes.txt' is a place holder.
    Inference inf(projectBasePath + "/yolo11s.onnx", cv::Size(640, 640), "classes.txt", runOnGPU);

    std::vector<std::string> imageNames;
    imageNames.push_back(projectBasePath + "/ultralytics/assets/bus.jpg");
    imageNames.push_back(projectBasePath + "/ultralytics/assets/zidane.jpg");

    for (int i = 0; i < imageNames.size(); ++i)
    {
        cv::Mat frame = cv::imread(imageNames[i]);

        // Inference starts here...
        std::vector<Detection> output = inf.runInference(frame);

        int detections = output.size();
        std::cout << "Number of detections:" << detections << std::endl;

        for (int i = 0; i < detections; ++i)
        {
            Detection detection = output[i];
            yolo::DrawBox(frame, detection.box, yolo::Label(detection.className, detection.confidence), detection.class_id);
        }
        // Inference ends here...

        // Write the annotated image; pass --show to also open a window.
        cv::imwrite("yolo_opencv_dnn.jpg", frame);
        std::cout << "Result image written to yolo_opencv_dnn.jpg" << std::endl;
        yolo::Show("Inference", frame, show);
    }
}
