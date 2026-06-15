// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#ifndef YOLO_INFERENCE_H_
#define YOLO_INFERENCE_H_

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

#include "yolo_types.hpp"

namespace yolo {

// Runtime configuration for the OpenVINO backend.
struct Config {
    std::string model_path = "yolo26n.onnx";  // OpenVINO IR (.xml) or ONNX (.onnx)
    float conf = 0.25f;                        // confidence threshold
    float iou = 0.45f;                         // NMS IoU threshold (grid models only)
    std::string device = "AUTO";               // OpenVINO device: AUTO / CPU / GPU
};

// Loads any YOLO model (IR or ONNX, any task / generation) and runs inference with
// OpenVINO. Class names come from the IR rt_info "labels"; the task is inferred from
// the output shapes (the IR carries no explicit task field). Post-processing is shared
// with the other examples via common/yolo_postprocess.hpp.
class Predictor {
   public:
    explicit Predictor(const Config& config);

    Task task() const { return task_; }
    const std::vector<std::string>& names() const { return names_; }

    // For Semantic the class-id map is written to `semantic`; otherwise it returns detections.
    std::vector<Result> predict(const cv::Mat& image, cv::Mat& semantic);

   private:
    void load_names(const std::shared_ptr<ov::Model>& model);

    Config config_;
    int imgsz_ = 640;
    Task task_ = Task::Detect;
    std::vector<std::string> names_;

    ov::Core core_;
    ov::CompiledModel compiled_model_;
    ov::InferRequest request_;
};

}  // namespace yolo

#endif  // YOLO_INFERENCE_H_
