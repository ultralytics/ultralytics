// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "grpc_client.h"

#include "yolo_types.hpp"

namespace yolo {

namespace tc = triton::client;

// Runtime configuration for the Triton Inference Server backend.
struct Config {
    std::string url = "localhost:8001";  // Triton gRPC endpoint
    std::string model = "yolo26n";       // model name as deployed on the server
    std::string model_version = "";      // "" lets Triton pick the latest version
    Task task = Task::Unknown;           // optional override; otherwise inferred from output shapes
    int imgsz = 640;                     // fallback input size if the model shape is dynamic
    float conf = 0.25f;                  // confidence threshold
    float iou = 0.45f;                   // NMS IoU threshold (grid models only)
};

// Connects to a Triton-served YOLO model (any task / generation) over gRPC and runs
// inference. Triton carries no task field, so the task is inferred from the output
// shapes; class names fall back to COCO. The FP16/FP32 input and output layout are
// read from the model metadata. Post-processing is shared with the other examples
// via common/yolo_postprocess.hpp.
class Predictor {
   public:
    explicit Predictor(const Config& config);

    Task task() const { return task_; }
    const std::vector<std::string>& names() const { return names_; }

    // For Semantic the class-id map is written to `semantic`; otherwise it returns detections.
    std::vector<Result> predict(const cv::Mat& image, cv::Mat& semantic);

   private:
    void load_metadata();

    Config config_;
    int imgsz_ = 640;
    Task task_ = Task::Detect;
    std::vector<std::string> names_;

    std::unique_ptr<tc::InferenceServerGrpcClient> client_;
    std::string input_name_;
    bool input_fp16_ = true;                  // model input datatype (FP16 vs FP32)
    std::vector<std::string> output_names_;   // server output tensor names
    std::vector<bool> output_fp16_;           // datatype per output (parallel to output_names_)
};

}  // namespace yolo
