// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"

#include <algorithm>
#include <iostream>
#include <sstream>

#include "coco_names.hpp"
#include "yolo_postprocess.hpp"

namespace yolo {

Predictor::Predictor(const Config& config) : config_(config) {
    std::shared_ptr<ov::Model> model = core_.read_model(config_.model_path);

    // Input size from the model's input shape [1, 3, H, W].
    const ov::Shape input_shape = model->input().get_shape();
    if (input_shape.size() == 4) imgsz_ = static_cast<int>(input_shape[2]);

    load_names(model);

    // Infer the task from the output shapes + label count (the IR has no task field).
    std::vector<std::vector<int64_t>> output_shapes;
    for (const auto& output : model->outputs()) {
        const ov::Shape s = output.get_shape();
        output_shapes.emplace_back(s.begin(), s.end());
    }
    task_ = InferTask(output_shapes, static_cast<int>(names_.size()));

    compiled_model_ = core_.compile_model(model, config_.device);
    request_ = compiled_model_.create_infer_request();
}

void Predictor::load_names(const std::shared_ptr<ov::Model>& model) {
    names_.clear();
    try {
        if (model->has_rt_info("model_info", "labels")) {
            // Space-separated labels with underscores inside multi-word names, e.g.
            // "person ... traffic_light ... cell_phone".
            const std::string labels = model->get_rt_info<std::string>("model_info", "labels");
            std::istringstream iss(labels);
            std::string word;
            while (iss >> word) {
                std::replace(word.begin(), word.end(), '_', ' ');
                names_.push_back(word);
            }
        }
    } catch (const std::exception&) {
        names_.clear();
    }
    if (names_.empty()) names_ = CocoNames();
}

std::vector<Result> Predictor::predict(const cv::Mat& image, cv::Mat& semantic) {
    float scale = 1.0f;
    cv::Mat input = Preprocess(image, imgsz_, task_ == Task::Classify, scale);
    std::vector<float> blob = ToBlob(input, imgsz_);

    const ov::Shape shape{1, 3, static_cast<size_t>(imgsz_), static_cast<size_t>(imgsz_)};
    ov::Tensor input_tensor(ov::element::f32, shape, blob.data());
    request_.set_input_tensor(input_tensor);
    request_.infer();

    // Primary output (2D/3D) plus the 4D proto/semantic tensor when present.
    const size_t num_outputs = compiled_model_.outputs().size();
    int main_idx = 0, aux_idx = -1;
    std::vector<ov::Tensor> tensors(num_outputs);
    std::vector<std::vector<int64_t>> shapes(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        tensors[i] = request_.get_output_tensor(i);
        const ov::Shape s = tensors[i].get_shape();
        shapes[i].assign(s.begin(), s.end());
        if (s.size() == 4) aux_idx = static_cast<int>(i);
        else main_idx = static_cast<int>(i);
    }
    const float* data = tensors[main_idx].data<const float>();
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
            const float* pdata = tensors[aux_idx].data<const float>();
            return PostprocessSegment(data, shp, pdata, shapes[aux_idx], scale, config_.conf, config_.iou,
                                      imgsz_, image.size());
        }
        case Task::Semantic:
            PostprocessSemantic(data, shp, scale, imgsz_, image.size(), semantic);
            return {};
        default:
            std::cerr << "[yolo] task '" << TaskName(task_) << "' is not supported." << std::endl;
            return {};
    }
}

}  // namespace yolo
