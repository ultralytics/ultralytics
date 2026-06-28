// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <torch/script.h>

#include "yolo_types.hpp"

namespace yolo {

// Runtime configuration for the LibTorch backend.
struct Config {
    std::string model_path = "yolo26n.torchscript";
    float conf = 0.25f;  // confidence threshold
    float iou = 0.45f;   // NMS IoU threshold (grid models only)
    bool cuda = false;   // use CUDA if the LibTorch build and a device are available
};

// Loads any YOLO TorchScript model (any task and generation) and runs inference.
// The task, class names, and input size are read from the metadata that Ultralytics
// stores in the TorchScript "config.txt" extra file. Post-processing is shared with
// the other examples via common/yolo_postprocess.hpp.
class Predictor {
   public:
    explicit Predictor(const Config& config);

    Task task() const { return task_; }
    const std::vector<std::string>& names() const { return names_; }

    // For Semantic the class-id map is written to `semantic`; otherwise it returns detections.
    std::vector<Result> predict(const cv::Mat& image, cv::Mat& semantic);

   private:
    void load_metadata(const std::string& config_json);

    Config config_;
    int imgsz_ = 640;
    Task task_ = Task::Detect;
    std::vector<std::string> names_;

    torch::jit::script::Module module_;
    torch::Device device_;
};

}  // namespace yolo
