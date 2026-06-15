// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef INFERENCE_H
#define INFERENCE_H

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>

#include "yolo_types.hpp"

namespace yolo {

// Runtime configuration for the OpenCV DNN backend.
struct Config {
    std::string model_path = "yolo11n.onnx";  // grid model; OpenCV DNN cannot run YOLO26 end2end ops
    float conf = 0.25f;         // confidence threshold
    float iou = 0.45f;          // NMS IoU threshold (grid models only)
    int imgsz = 640;            // square input size of the exported model
    bool cuda = false;          // use the OpenCV CUDA DNN backend (requires a CUDA-enabled OpenCV)
    Task task = Task::Unknown;  // optional override; OpenCV cannot read the task from the model
};

// Loads any YOLO ONNX model with the OpenCV DNN module and runs inference. OpenCV
// cannot read the task or class names from the model, so the task is inferred from
// the output shapes (override with Config::task for grid pose/obb), and class names
// fall back to the COCO list in common/coco_names.hpp. Post-processing is shared with
// the other examples via common/yolo_postprocess.hpp.
class Predictor {
   public:
    explicit Predictor(const Config& config);

    Task task() const { return task_; }
    const std::vector<std::string>& names() const { return names_; }

    // For Semantic the class-id map is written to `semantic`; otherwise it returns detections.
    std::vector<Result> predict(const cv::Mat& image, cv::Mat& semantic);

   private:
    std::vector<cv::Mat> forward(const std::vector<float>& blob);

    Config config_;
    Task task_ = Task::Detect;
    std::vector<std::string> names_;
    cv::dnn::Net net_;
};

}  // namespace yolo

#endif  // INFERENCE_H
