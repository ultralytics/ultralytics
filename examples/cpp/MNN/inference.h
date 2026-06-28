// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <MNN/Interpreter.hpp>

#include "yolo_types.hpp"

namespace yolo {

// Runtime configuration for the MNN backend.
struct Config {
    std::string model_path = "yolo26n.mnn";
    float conf = 0.25f;  // confidence threshold
    float iou = 0.45f;   // NMS IoU threshold (grid models only)
    int threads = 4;     // CPU threads
};

// Loads any YOLO MNN model (any task and generation) and runs inference with the MNN
// Interpreter API. The task, class names, and input size are read from the model
// bizCode metadata that Ultralytics embeds on export. Post-processing is shared with
// the other examples via common/yolo_postprocess.hpp, and drawing uses OpenCV.
class Predictor {
   public:
    explicit Predictor(const Config& config);
    ~Predictor();

    Task task() const { return task_; }
    const std::vector<std::string>& names() const { return names_; }

    // For Semantic the class-id map is written to `semantic`; otherwise it returns detections.
    std::vector<Result> predict(const cv::Mat& image, cv::Mat& semantic);

   private:
    void load_metadata(const std::string& biz_code, const std::vector<std::vector<int64_t>>& output_shapes);

    Config config_;
    int imgsz_ = 640;
    Task task_ = Task::Unknown;  // set from bizCode metadata or inferred from output shapes
    std::vector<std::string> names_;

    MNN::Interpreter* interpreter_ = nullptr;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_ = nullptr;
};

}  // namespace yolo
