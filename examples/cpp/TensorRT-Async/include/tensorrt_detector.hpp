// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

#include "yolo_types.hpp"

namespace yolo {

struct InferenceTiming {
    float preprocess_ms = 0.0f;
    float gpu_ms = 0.0f;
    float postprocess_ms = 0.0f;
};

class TensorRTDetector {
public:
    TensorRTDetector(
        const std::string& engine_path, int expected_class_count, float confidence_threshold, float iou_threshold
    );
    ~TensorRTDetector();

    TensorRTDetector(const TensorRTDetector&) = delete;
    TensorRTDetector& operator=(const TensorRTDetector&) = delete;

    std::vector<Result> infer(const cv::Mat& image, InferenceTiming& timing);
    int inputWidth() const { return input_width_; }
    int inputHeight() const { return input_height_; }
    int classCount() const { return num_classes_; }

private:
    class Logger final : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* message) noexcept override;
    };

    struct LetterboxTransform {
        float scale = 1.0f;
        int pad_x = 0;
        int pad_y = 0;
        int original_width = 0;
        int original_height = 0;
    };

    LetterboxTransform preprocess(const cv::Mat& image);
    std::vector<Result> postprocess(const LetterboxTransform& transform);
    void release() noexcept;

    Logger logger_;
    std::unique_ptr<nvinfer1::IRuntime> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    std::string input_name_;
    std::string output_name_;
    int input_width_ = 0;
    int input_height_ = 0;
    int num_candidates_ = 0;
    int num_attributes_ = 0;
    int num_classes_ = 0;
    bool channel_major_output_ = true;

    std::size_t input_bytes_ = 0;
    std::size_t output_bytes_ = 0;
    float* host_input_ = nullptr;
    float* host_output_ = nullptr;
    void* device_input_ = nullptr;
    void* device_output_ = nullptr;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t gpu_start_ = nullptr;
    cudaEvent_t gpu_end_ = nullptr;

    float confidence_threshold_;
    float iou_threshold_;
    cv::Mat resized_;
    cv::Mat letterboxed_;
    std::vector<cv::Rect> candidate_boxes_;
    std::vector<cv::Rect> nms_boxes_;
    std::vector<float> candidate_scores_;
    std::vector<int> candidate_classes_;
    std::vector<int> kept_indices_;
};

}  // namespace yolo
