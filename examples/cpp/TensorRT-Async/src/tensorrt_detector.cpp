#include "tensorrt_detector.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/dnn/dnn.hpp>
#include <opencv2/imgproc.hpp>

namespace yolo {
namespace {

using Clock = std::chrono::steady_clock;

float Milliseconds(Clock::time_point start, Clock::time_point end) {
    return std::chrono::duration<float, std::milli>(end - start).count();
}

void CheckCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + " failed: " + cudaGetErrorString(error));
    }
}

std::size_t TensorVolume(const nvinfer1::Dims& dimensions, const char* tensor_name) {
    if (dimensions.nbDims <= 0) throw std::runtime_error(std::string(tensor_name) + " has no dimensions");
    std::size_t volume = 1;
    for (int i = 0; i < dimensions.nbDims; ++i) {
        if (dimensions.d[i] <= 0) {
            throw std::runtime_error(
                std::string(tensor_name) +
                " has a dynamic shape; build a fixed batch=1 engine with trtexec before running this example");
        }
        const auto dimension = static_cast<std::size_t>(dimensions.d[i]);
        if (volume > std::numeric_limits<std::size_t>::max() / dimension) {
            throw std::runtime_error(std::string(tensor_name) + " is too large");
        }
        volume *= dimension;
    }
    return volume;
}

std::vector<char> ReadEngine(const std::string& path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("cannot open TensorRT engine: " + path);

    const std::streamsize size = file.tellg();
    if (size <= 0) throw std::runtime_error("TensorRT engine is empty: " + path);
    file.seekg(0, std::ios::beg);

    std::vector<char> bytes(static_cast<std::size_t>(size));
    if (!file.read(bytes.data(), size)) throw std::runtime_error("cannot read TensorRT engine: " + path);
    return bytes;
}

}  // namespace

void TensorRTDetector::Logger::log(Severity severity, const char* message) noexcept {
    if (severity <= Severity::kWARNING) std::cerr << "[TensorRT] " << message << '\n';
}

TensorRTDetector::TensorRTDetector(
    const std::string& engine_path, float confidence_threshold, float iou_threshold
)
    : confidence_threshold_(confidence_threshold), iou_threshold_(iou_threshold) {
    try {
        const std::vector<char> serialized_engine = ReadEngine(engine_path);
        runtime_.reset(nvinfer1::createInferRuntime(logger_));
        if (!runtime_) throw std::runtime_error("failed to create TensorRT runtime");

        engine_.reset(runtime_->deserializeCudaEngine(serialized_engine.data(), serialized_engine.size()));
        if (!engine_) {
            throw std::runtime_error(
                "failed to deserialize the engine; TensorRT version and target GPU must match the build environment");
        }
        context_.reset(engine_->createExecutionContext());
        if (!context_) throw std::runtime_error("failed to create TensorRT execution context");

        for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
            const char* name = engine_->getIOTensorName(i);
            if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
                if (!input_name_.empty()) throw std::runtime_error("only one input tensor is supported");
                input_name_ = name;
            } else {
                if (!output_name_.empty()) throw std::runtime_error("only one output tensor is supported");
                output_name_ = name;
            }
        }
        if (input_name_.empty() || output_name_.empty()) {
            throw std::runtime_error("the engine must contain exactly one input and one output tensor");
        }
        if (engine_->getTensorDataType(input_name_.c_str()) != nvinfer1::DataType::kFLOAT ||
            engine_->getTensorDataType(output_name_.c_str()) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(
                "this example expects FP32 engine I/O; use trtexec --fp16 without forcing FP16 I/O formats");
        }

        const nvinfer1::Dims input_shape = engine_->getTensorShape(input_name_.c_str());
        if (input_shape.nbDims != 4 || input_shape.d[0] != 1 || input_shape.d[1] != 3) {
            throw std::runtime_error("expected input shape [1, 3, height, width]");
        }
        input_height_ = input_shape.d[2];
        input_width_ = input_shape.d[3];

        const nvinfer1::Dims output_shape = engine_->getTensorShape(output_name_.c_str());
        if (output_shape.nbDims != 3 || output_shape.d[0] != 1) {
            throw std::runtime_error(
                "expected a raw YOLOv8 detection output shaped [1, 4+classes, anchors] or [1, anchors, 4+classes]");
        }
        const std::size_t input_elements = TensorVolume(input_shape, input_name_.c_str());
        const std::size_t output_elements = TensorVolume(output_shape, output_name_.c_str());

        channel_major_output_ = output_shape.d[1] < output_shape.d[2];
        num_attributes_ = channel_major_output_ ? output_shape.d[1] : output_shape.d[2];
        num_candidates_ = channel_major_output_ ? output_shape.d[2] : output_shape.d[1];
        num_classes_ = num_attributes_ - 4;
        if (num_classes_ <= 0) throw std::runtime_error("YOLOv8 output has no class scores");

        input_bytes_ = input_elements * sizeof(float);
        output_bytes_ = output_elements * sizeof(float);

        CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&host_input_), input_bytes_), "cudaMallocHost(input)");
        CheckCuda(cudaMallocHost(reinterpret_cast<void**>(&host_output_), output_bytes_), "cudaMallocHost(output)");
        CheckCuda(cudaMalloc(&device_input_, input_bytes_), "cudaMalloc(input)");
        CheckCuda(cudaMalloc(&device_output_, output_bytes_), "cudaMalloc(output)");
        CheckCuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "cudaStreamCreateWithFlags");
        CheckCuda(cudaEventCreate(&gpu_start_), "cudaEventCreate(start)");
        CheckCuda(cudaEventCreate(&gpu_end_), "cudaEventCreate(end)");

        if (!context_->setTensorAddress(input_name_.c_str(), device_input_) ||
            !context_->setTensorAddress(output_name_.c_str(), device_output_)) {
            throw std::runtime_error("failed to bind TensorRT input/output memory");
        }

        letterboxed_.create(input_height_, input_width_, CV_8UC3);
        candidate_boxes_.reserve(static_cast<std::size_t>(num_candidates_));
        nms_boxes_.reserve(static_cast<std::size_t>(num_candidates_));
        candidate_scores_.reserve(static_cast<std::size_t>(num_candidates_));
        candidate_classes_.reserve(static_cast<std::size_t>(num_candidates_));
        kept_indices_.reserve(static_cast<std::size_t>(num_candidates_));
    } catch (...) {
        release();
        throw;
    }
}

TensorRTDetector::~TensorRTDetector() {
    release();
}

void TensorRTDetector::release() noexcept {
    if (stream_) cudaStreamSynchronize(stream_);
    if (gpu_end_) cudaEventDestroy(gpu_end_);
    if (gpu_start_) cudaEventDestroy(gpu_start_);
    if (stream_) cudaStreamDestroy(stream_);
    if (device_output_) cudaFree(device_output_);
    if (device_input_) cudaFree(device_input_);
    if (host_output_) cudaFreeHost(host_output_);
    if (host_input_) cudaFreeHost(host_input_);
    gpu_end_ = nullptr;
    gpu_start_ = nullptr;
    stream_ = nullptr;
    device_output_ = nullptr;
    device_input_ = nullptr;
    host_output_ = nullptr;
    host_input_ = nullptr;
    context_.reset();
    engine_.reset();
    runtime_.reset();
}

TensorRTDetector::LetterboxTransform TensorRTDetector::preprocess(const cv::Mat& image) {
    if (image.empty() || image.type() != CV_8UC3) {
        throw std::runtime_error("camera frame must be a non-empty 8-bit BGR image");
    }

    LetterboxTransform transform;
    transform.original_width = image.cols;
    transform.original_height = image.rows;
    transform.scale = std::min(
        input_width_ / static_cast<float>(image.cols), input_height_ / static_cast<float>(image.rows)
    );
    const int resized_width = static_cast<int>(std::round(image.cols * transform.scale));
    const int resized_height = static_cast<int>(std::round(image.rows * transform.scale));
    transform.pad_x = (input_width_ - resized_width) / 2;
    transform.pad_y = (input_height_ - resized_height) / 2;

    cv::resize(image, resized_, cv::Size(resized_width, resized_height), 0.0, 0.0, cv::INTER_LINEAR);
    letterboxed_.setTo(cv::Scalar(114, 114, 114));
    resized_.copyTo(letterboxed_(cv::Rect(transform.pad_x, transform.pad_y, resized_width, resized_height)));

    const int area = input_width_ * input_height_;
    for (int y = 0; y < input_height_; ++y) {
        const cv::Vec3b* row = letterboxed_.ptr<cv::Vec3b>(y);
        for (int x = 0; x < input_width_; ++x) {
            const int index = y * input_width_ + x;
            host_input_[index] = row[x][2] / 255.0f;
            host_input_[area + index] = row[x][1] / 255.0f;
            host_input_[2 * area + index] = row[x][0] / 255.0f;
        }
    }
    return transform;
}

std::vector<Result> TensorRTDetector::postprocess(const LetterboxTransform& transform) {
    candidate_boxes_.clear();
    nms_boxes_.clear();
    candidate_scores_.clear();
    candidate_classes_.clear();
    kept_indices_.clear();

    const auto value = [this](int attribute, int candidate) {
        if (channel_major_output_) return host_output_[attribute * num_candidates_ + candidate];
        return host_output_[candidate * num_attributes_ + attribute];
    };

    const cv::Rect image_bounds(0, 0, transform.original_width, transform.original_height);
    const int class_offset = std::max(transform.original_width, transform.original_height) + 1;
    for (int candidate = 0; candidate < num_candidates_; ++candidate) {
        int class_id = 0;
        float confidence = value(4, candidate);
        for (int class_index = 1; class_index < num_classes_; ++class_index) {
            const float score = value(4 + class_index, candidate);
            if (score > confidence) {
                confidence = score;
                class_id = class_index;
            }
        }
        if (confidence < confidence_threshold_) continue;

        const float center_x = value(0, candidate);
        const float center_y = value(1, candidate);
        const float width = value(2, candidate);
        const float height = value(3, candidate);
        const float left = (center_x - width * 0.5f - transform.pad_x) / transform.scale;
        const float top = (center_y - height * 0.5f - transform.pad_y) / transform.scale;
        const float right = (center_x + width * 0.5f - transform.pad_x) / transform.scale;
        const float bottom = (center_y + height * 0.5f - transform.pad_y) / transform.scale;

        const int x1 = static_cast<int>(std::floor(left));
        const int y1 = static_cast<int>(std::floor(top));
        const int x2 = static_cast<int>(std::ceil(right));
        const int y2 = static_cast<int>(std::ceil(bottom));
        cv::Rect box(x1, y1, x2 - x1, y2 - y1);
        box &= image_bounds;
        if (box.empty()) continue;

        candidate_boxes_.push_back(box);
        nms_boxes_.emplace_back(box.x + class_id * class_offset, box.y, box.width, box.height);
        candidate_scores_.push_back(confidence);
        candidate_classes_.push_back(class_id);
    }

    cv::dnn::NMSBoxes(
        nms_boxes_, candidate_scores_, confidence_threshold_, iou_threshold_, kept_indices_
    );

    std::vector<Result> results;
    results.reserve(kept_indices_.size());
    for (int index : kept_indices_) {
        Result result;
        result.class_id = candidate_classes_[static_cast<std::size_t>(index)];
        result.confidence = candidate_scores_[static_cast<std::size_t>(index)];
        result.box = candidate_boxes_[static_cast<std::size_t>(index)];
        results.push_back(result);
    }
    return results;
}

std::vector<Result> TensorRTDetector::infer(const cv::Mat& image, InferenceTiming& timing) {
    const auto preprocess_start = Clock::now();
    const LetterboxTransform transform = preprocess(image);
    const auto preprocess_end = Clock::now();

    CheckCuda(cudaEventRecord(gpu_start_, stream_), "cudaEventRecord(start)");
    CheckCuda(
        cudaMemcpyAsync(device_input_, host_input_, input_bytes_, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync(input)"
    );
    if (!context_->enqueueV3(stream_)) throw std::runtime_error("TensorRT enqueueV3 failed");
    CheckCuda(
        cudaMemcpyAsync(host_output_, device_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync(output)"
    );
    CheckCuda(cudaEventRecord(gpu_end_, stream_), "cudaEventRecord(end)");
    CheckCuda(cudaEventSynchronize(gpu_end_), "cudaEventSynchronize(end)");
    CheckCuda(cudaEventElapsedTime(&timing.gpu_ms, gpu_start_, gpu_end_), "cudaEventElapsedTime");

    const auto postprocess_start = Clock::now();
    std::vector<Result> results = postprocess(transform);
    const auto postprocess_end = Clock::now();

    timing.preprocess_ms = Milliseconds(preprocess_start, preprocess_end);
    timing.postprocess_ms = Milliseconds(postprocess_start, postprocess_end);
    return results;
}

}  // namespace yolo
