// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <string>
#include <vector>

#include <opencv2/opencv.hpp>
#include "onnxruntime_cxx_api.h"

namespace yolo {

// Task type, read from the model metadata that Ultralytics bakes into every export.
enum class Task { Detect, Segment, Pose, Obb, Classify, Semantic, Unknown };

Task TaskFromString(const std::string& s);
std::string TaskName(Task task);

// Runtime configuration shared by every task.
struct Config {
    std::string model_path = "yolo26n.onnx";
    float conf = 0.25f;  // confidence threshold
    float iou = 0.45f;   // NMS IoU threshold (grid models only; end2end models are already NMS-free)
    bool cuda = false;   // use the CUDA execution provider
};

// A single prediction. Fields are populated per task: box (all detection tasks),
// keypoints (pose), mask (segment), angle (obb). Classify uses class_id/confidence.
struct Result {
    int class_id = 0;
    float confidence = 0.0f;
    cv::Rect box;
    float angle = 0.0f;                        // OBB rotation in radians
    std::vector<cv::Point2f> keypoints;        // Pose keypoints in original-image coordinates
    std::vector<float> keypoint_scores;        // Per-keypoint confidence
    cv::Mat mask;                              // Segment: 8-bit binary mask at original image size
};

// One predictor for every YOLO task and generation. The task and class names come
// from the model metadata; the output layout (grid YOLOv8/11 vs end2end YOLO26) is
// detected automatically from the output tensor shape.
class Predictor {
   public:
    explicit Predictor(const Config& config);
    ~Predictor();

    Task task() const { return task_; }
    const std::vector<std::string>& names() const { return names_; }

    // Run inference on a BGR image. Returns detection-style results; for Semantic the
    // class-id map is written to `semantic` (CV_32S, original image size) and the
    // returned vector is empty.
    std::vector<Result> predict(const cv::Mat& image, cv::Mat& semantic);

   private:
    cv::Mat preprocess(const cv::Mat& image, float& scale);
    void load_metadata(Ort::AllocatorWithDefaultOptions& allocator);

    // Per-task post-processing.
    std::vector<Result> postprocess_detect(const float* data, const std::vector<int64_t>& shape, float scale);
    std::vector<Result> postprocess_classify(const float* data, const std::vector<int64_t>& shape);
    std::vector<Result> postprocess_pose(const float* data, const std::vector<int64_t>& shape, float scale);
    std::vector<Result> postprocess_obb(const float* data, const std::vector<int64_t>& shape, float scale);
    std::vector<Result> postprocess_segment(const float* det, const std::vector<int64_t>& det_shape,
                                            const float* protos, const std::vector<int64_t>& proto_shape,
                                            float scale, const cv::Size& orig_size);
    void postprocess_semantic(const float* data, const std::vector<int64_t>& shape, float scale,
                              const cv::Size& orig_size, cv::Mat& out);

    Config config_;
    int imgsz_ = 640;
    Task task_ = Task::Detect;
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
