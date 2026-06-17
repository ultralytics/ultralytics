// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

#include "yolo_types.hpp"

namespace yolo {

// Runtime configuration for the ONNX Runtime backend.
struct Config {
    std::string model_path = "yolo26n.onnx";
    float conf = 0.25f;  // confidence threshold
    float iou = 0.45f;   // NMS IoU threshold (grid models only)
    bool cuda = false;   // use the CUDA execution provider
};

// Loads any YOLO ONNX model (any task / generation) and runs inference. The task,
// class names, and input size come from the model metadata; the output layout
// (grid vs end2end) is detected from the tensor shape inside the shared post-processing.
class Predictor {
   public:
    explicit Predictor(const Config& config);
    ~Predictor();

    Task task() const { return task_; }
    const std::vector<std::string>& names() const { return names_; }

    // For Semantic the class-id map is written to `semantic`; otherwise it returns detections.
    std::vector<Result> predict(const cv::Mat& image, cv::Mat& semantic);

   private:
    void load_metadata(Ort::AllocatorWithDefaultOptions& allocator);

    Config config_;
    int imgsz_ = 640;
    Task task_ = Task::Detect;
    bool input_fp16_ = false;  // model expects half-precision input (FP16 export)
    std::vector<std::string> names_;

    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "yolo"};
    Ort::SessionOptions session_options_;
    Ort::Session* session_ = nullptr;
    Ort::RunOptions run_options_{nullptr};
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
};

}  // namespace yolo
