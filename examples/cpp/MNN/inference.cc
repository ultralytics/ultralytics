// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.h"

#include <memory>
#include <regex>

#include <MNN/Tensor.hpp>

#include "coco_names.hpp"
#include "yolo_postprocess.hpp"

namespace yolo {
namespace {

std::vector<int64_t> TensorShape(const MNN::Tensor* t) {
    const std::vector<int> dims = t->shape();
    return std::vector<int64_t>(dims.begin(), dims.end());
}

}  // namespace

Predictor::Predictor(const Config& config) : config_(config) {
    interpreter_ = MNN::Interpreter::createFromFile(config_.model_path.c_str());

    MNN::ScheduleConfig schedule;
    schedule.type = MNN_FORWARD_CPU;
    schedule.numThread = config_.threads;
    session_ = interpreter_->createSession(schedule);
    input_ = interpreter_->getSessionInput(session_, nullptr);

    // Parse the bizCode metadata (names, imgsz, and task when present) before resizing.
    load_metadata(interpreter_->bizCode(), {});

    interpreter_->resizeTensor(input_, {1, 3, imgsz_, imgsz_});
    interpreter_->resizeSession(session_);

    // Warm up to learn the output shapes, then infer the task if bizCode had none.
    std::vector<float> zeros(3 * imgsz_ * imgsz_, 0.0f);
    std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create<float>({1, 3, imgsz_, imgsz_}, zeros.data(), MNN::Tensor::CAFFE));
    input_->copyFromHostTensor(host.get());
    interpreter_->runSession(session_);
    if (task_ == Task::Unknown) {
        std::vector<std::vector<int64_t>> shapes;
        for (const auto& kv : interpreter_->getSessionOutputAll(session_)) {
            // Read the logical NCHW shape from a host tensor, not the device tensor.
            std::unique_ptr<MNN::Tensor> h(new MNN::Tensor(kv.second, MNN::Tensor::CAFFE));
            shapes.push_back(TensorShape(h.get()));
        }
        task_ = InferTask(shapes, static_cast<int>(names_.size()));
    }
}

Predictor::~Predictor() {
    if (interpreter_) MNN::Interpreter::destroy(interpreter_);
}

void Predictor::load_metadata(const std::string& biz, const std::vector<std::vector<int64_t>>&) {
    std::smatch m;
    if (std::regex_search(biz, m, std::regex("\"task\"\\s*:\\s*\"([^\"]+)\""))) task_ = TaskFromString(m[1]);
    if (std::regex_search(biz, m, std::regex("\"imgsz\"\\s*:\\s*\\[\\s*(\\d+)"))) imgsz_ = std::stoi(m[1]);

    names_.clear();
    std::regex item("['\"]?(\\d+)['\"]?\\s*:\\s*['\"]([^'\"]*)['\"]");
    for (std::sregex_iterator it(biz.begin(), biz.end(), item), end; it != end; ++it) {
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

    std::unique_ptr<MNN::Tensor> host(MNN::Tensor::create<float>({1, 3, imgsz_, imgsz_}, blob.data(), MNN::Tensor::CAFFE));
    input_->copyFromHostTensor(host.get());
    interpreter_->runSession(session_);

    // Copy every output to a host NCHW tensor and collect a float pointer + shape.
    std::vector<std::shared_ptr<MNN::Tensor>> hosts;
    std::vector<std::vector<int64_t>> shapes;
    std::vector<const float*> data;
    int main_idx = 0, aux_idx = -1, i = 0;
    for (const auto& kv : interpreter_->getSessionOutputAll(session_)) {
        std::shared_ptr<MNN::Tensor> h(new MNN::Tensor(kv.second, MNN::Tensor::CAFFE));
        kv.second->copyToHostTensor(h.get());
        const std::vector<int64_t> shape = TensorShape(h.get());
        hosts.push_back(h);
        shapes.push_back(shape);
        data.push_back(h->host<float>());
        if (shape.size() == 4) aux_idx = i;
        else main_idx = i;
        ++i;
    }
    if (data.empty()) return {};
    const float* data = data[main_idx];
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
            return PostprocessSegment(data, shp, data[aux_idx], shapes[aux_idx], scale, config_.conf, config_.iou,
                                      imgsz_, image.size());
        }
        case Task::Semantic:
            PostprocessSemantic(data, shp, scale, imgsz_, image.size(), semantic);
            return {};
        default:
            return {};
    }
}

}  // namespace yolo
