// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"

#include <regex>
#include <unordered_map>

#include <torch/torch.h>

#include "coco_names.hpp"
#include "yolo_postprocess.hpp"

namespace yolo {

Predictor::Predictor(const Config& config) : config_(config), device_(torch::kCPU) {
    if (config_.cuda && torch::cuda::is_available()) device_ = torch::Device(torch::kCUDA);

    // Ultralytics stores the model metadata as JSON in the "config.txt" extra file.
    std::unordered_map<std::string, std::string> extra_files;
    extra_files["config.txt"] = "";
    module_ = torch::jit::load(config_.model_path, device_, extra_files);
    module_.eval();

    load_metadata(extra_files["config.txt"]);
}

void Predictor::load_metadata(const std::string& cfg) {
    std::smatch m;
    if (std::regex_search(cfg, m, std::regex("\"task\"\\s*:\\s*\"([^\"]+)\""))) task_ = TaskFromString(m[1]);
    if (std::regex_search(cfg, m, std::regex("\"imgsz\"\\s*:\\s*\\[\\s*(\\d+)"))) imgsz_ = std::stoi(m[1]);

    names_.clear();
    // Handle both Python-dict ("0: 'person'") and JSON ("\"0\": \"person\"") key styles.
    std::regex item("['\"]?(\\d+)['\"]?\\s*:\\s*['\"]([^'\"]*)['\"]");
    for (std::sregex_iterator it(cfg.begin(), cfg.end(), item), end; it != end; ++it) {
        int idx = std::stoi((*it)[1].str());
        if (static_cast<int>(names_.size()) <= idx) names_.resize(idx + 1);
        names_[idx] = (*it)[2].str();
    }
    if (names_.empty()) names_ = CocoNames();
}

std::vector<Result> Predictor::predict(const cv::Mat& image, cv::Mat& semantic) {
    float scale = 1.0f;
    cv::Mat input = Preprocess(image, imgsz_, task_ == Task::Classify, scale);
    std::vector<float> blob = ToBlob(input, imgsz_);

    torch::NoGradGuard no_grad;
    torch::Tensor input_tensor =
        torch::from_blob(blob.data(), {1, 3, imgsz_, imgsz_}, torch::kFloat32).clone().to(device_);
    torch::jit::IValue output = module_.forward({input_tensor});

    // Collect output tensors (a single tensor for most tasks, a tuple for segment).
    std::vector<torch::Tensor> tensors;
    if (output.isTensor()) {
        tensors.push_back(output.toTensor());
    } else if (output.isTuple()) {
        for (const auto& e : output.toTuple()->elements())
            if (e.isTensor()) tensors.push_back(e.toTensor());
    } else if (output.isTensorList()) {
        for (const auto& t : output.toTensorList()) tensors.push_back(t);
    }

    // Primary output (2D or 3D) plus the 4D proto tensor for segment.
    int main_idx = 0, aux_idx = -1;
    std::vector<std::vector<int64_t>> shapes(tensors.size());
    for (size_t i = 0; i < tensors.size(); ++i) {
        tensors[i] = tensors[i].to(torch::kCPU).contiguous();
        const auto sz = tensors[i].sizes();
        shapes[i].assign(sz.begin(), sz.end());
        if (sz.size() == 4) aux_idx = static_cast<int>(i);
        else main_idx = static_cast<int>(i);
    }
    if (tensors.empty()) return {};
    const float* data = tensors[main_idx].data_ptr<float>();
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
            const float* pdata = tensors[aux_idx].data_ptr<float>();
            return PostprocessSegment(data, shp, pdata, shapes[aux_idx], scale, config_.conf, config_.iou,
                                      imgsz_, image.size());
        }
        case Task::Semantic:
            PostprocessSemantic(tensors[main_idx].data_ptr<float>(), shapes[main_idx], scale, imgsz_,
                                image.size(), semantic);
            return {};
        default:
            return {};
    }
}

}  // namespace yolo
