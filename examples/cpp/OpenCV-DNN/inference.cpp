// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"

#include <vector>

#include "coco_names.hpp"
#include "yolo_postprocess.hpp"

namespace yolo {
namespace {

std::vector<int64_t> MatShape(const cv::Mat& m) {
    std::vector<int64_t> shape;
    for (int i = 0; i < m.dims; ++i) shape.push_back(m.size[i]);
    return shape;
}

}  // namespace

Predictor::Predictor(const Config& config) : config_(config) {
    net_ = cv::dnn::readNetFromONNX(config_.model_path);
    if (config_.cuda) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    } else {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    names_ = CocoNames();  // OpenCV cannot read class names from the ONNX metadata

    // OpenCV cannot read the task either; use the override or infer from output shapes.
    if (config_.task != Task::Unknown) {
        task_ = config_.task;
    } else {
        std::vector<float> zeros(3 * config_.imgsz * config_.imgsz, 0.0f);
        std::vector<cv::Mat> outputs = forward(zeros);
        std::vector<std::vector<int64_t>> shapes;
        for (const auto& m : outputs) shapes.push_back(MatShape(m));
        task_ = InferTask(shapes, 0);  // label count unknown, so a grid 3D output defaults to detect
    }
}

std::vector<cv::Mat> Predictor::forward(const std::vector<float>& blob) {
    int sizes[4] = {1, 3, config_.imgsz, config_.imgsz};
    cv::Mat input(4, sizes, CV_32F, const_cast<float*>(blob.data()));
    net_.setInput(input);
    std::vector<cv::Mat> outputs;
    net_.forward(outputs, net_.getUnconnectedOutLayersNames());
    return outputs;
}

std::vector<Result> Predictor::predict(const cv::Mat& image, cv::Mat& semantic) {
    float scale = 1.0f;
    cv::Mat input = Preprocess(image, config_.imgsz, task_ == Task::Classify, scale);
    std::vector<float> blob = ToBlob(input, config_.imgsz);
    std::vector<cv::Mat> outputs = forward(blob);

    int main_idx = 0, aux_idx = -1;
    std::vector<std::vector<int64_t>> shapes(outputs.size());
    for (size_t i = 0; i < outputs.size(); ++i) {
        shapes[i] = MatShape(outputs[i]);
        if (outputs[i].dims == 4) aux_idx = static_cast<int>(i);
        else main_idx = static_cast<int>(i);
    }
    if (outputs.empty()) return {};
    const float* data = reinterpret_cast<const float*>(outputs[main_idx].data);
    const std::vector<int64_t>& shp = shapes[main_idx];

    switch (task_) {
        case Task::Detect:
            return PostprocessDetect(data, shp, scale, config_.conf, config_.iou);
        case Task::Pose:
            return PostprocessPose(data, shp, scale, config_.conf, config_.iou);
        case Task::Obb:
            return PostprocessObb(data, shp, scale, config_.conf, config_.iou);
        case Task::Classify:
            return PostprocessClassify(data, shp);
        case Task::Segment: {
            if (aux_idx < 0) return {};
            const float* pdata = reinterpret_cast<const float*>(outputs[aux_idx].data);
            return PostprocessSegment(data, shp, pdata, shapes[aux_idx], scale, config_.conf, config_.iou,
                                      config_.imgsz, image.size());
        }
        case Task::Semantic:
            PostprocessSemantic(data, shp, scale, config_.imgsz, image.size(), semantic);
            return {};
        default:
            return {};
    }
}

}  // namespace yolo
