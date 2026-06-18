// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

#include "inference.hpp"

#include <immintrin.h>

#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "coco_names.hpp"
#include "yolo_postprocess.hpp"

namespace yolo {
namespace {

// IEEE 754 half <-> float using F16C intrinsics (enabled by -mf16c in CMakeLists).
inline uint16_t FloatToHalf(float value) { return _cvtss_sh(value, 0); }
inline float HalfToFloat(uint16_t half) { return _cvtsh_ss(half); }

// Abort with a clear message when a Triton call fails.
void Check(const tc::Error& err, const std::string& what) {
    if (!err.IsOk()) throw std::runtime_error(what + ": " + err.Message());
}

}  // namespace

Predictor::Predictor(const Config& config) : config_(config), imgsz_(config.imgsz) {
    Check(tc::InferenceServerGrpcClient::Create(&client_, config_.url), "failed to create gRPC client");

    bool ready = false;
    Check(client_->IsServerReady(&ready), "server-ready check failed");
    if (!ready) throw std::runtime_error("Triton server at " + config_.url + " is not ready");

    Check(client_->IsModelReady(&ready, config_.model, config_.model_version), "model-ready check failed");
    if (!ready) throw std::runtime_error("model '" + config_.model + "' is not ready on the server");

    load_metadata();

    if (config_.task != Task::Unknown) task_ = config_.task;
    names_ = CocoNames();  // Triton exposes no class names; fall back to COCO
}

void Predictor::load_metadata() {
    inference::ModelMetadataResponse metadata;
    Check(client_->ModelMetadata(&metadata, config_.model, config_.model_version), "failed to read model metadata");

    if (metadata.inputs_size() < 1) throw std::runtime_error("model has no inputs");
    const auto& input = metadata.inputs(0);
    input_name_ = input.name();
    input_fp16_ = input.datatype() == "FP16";
    // Input shape is [N, 3, H, W]; use the spatial dim when it is static (> 0).
    if (input.shape_size() == 4 && input.shape(2) > 0) imgsz_ = static_cast<int>(input.shape(2));

    // Collect output names/datatypes and infer the task from the output shapes.
    std::vector<std::vector<int64_t>> output_shapes;
    for (int i = 0; i < metadata.outputs_size(); ++i) {
        const auto& output = metadata.outputs(i);
        output_names_.push_back(output.name());
        output_fp16_.push_back(output.datatype() == "FP16");
        output_shapes.emplace_back(output.shape().begin(), output.shape().end());
    }
    task_ = InferTask(output_shapes, static_cast<int>(CocoNames().size()));
}

std::vector<Result> Predictor::predict(const cv::Mat& image, cv::Mat& semantic) {
    float scale = 1.0f;
    cv::Mat input = Preprocess(image, imgsz_, task_ == Task::Classify, scale);
    std::vector<float> blob = ToBlob(input, imgsz_);

    // Build the input tensor in the datatype the model expects (FP16 or FP32).
    const std::vector<int64_t> shape{1, 3, imgsz_, imgsz_};
    tc::InferInput* input_ptr = nullptr;
    Check(tc::InferInput::Create(&input_ptr, input_name_, shape, input_fp16_ ? "FP16" : "FP32"),
          "failed to create input");
    std::shared_ptr<tc::InferInput> input_handle(input_ptr);

    std::vector<uint16_t> half;
    if (input_fp16_) {
        half.resize(blob.size());
        for (size_t i = 0; i < blob.size(); ++i) half[i] = FloatToHalf(blob[i]);
        Check(input_handle->AppendRaw(reinterpret_cast<const uint8_t*>(half.data()), half.size() * sizeof(uint16_t)),
              "failed to append input data");
    } else {
        Check(input_handle->AppendRaw(reinterpret_cast<const uint8_t*>(blob.data()), blob.size() * sizeof(float)),
              "failed to append input data");
    }
    std::vector<tc::InferInput*> inputs{input_handle.get()};

    // Request every output the model exposes (e.g. detections + segment protos).
    std::vector<std::shared_ptr<tc::InferRequestedOutput>> output_handles;
    std::vector<const tc::InferRequestedOutput*> outputs;
    for (const std::string& name : output_names_) {
        tc::InferRequestedOutput* out = nullptr;
        Check(tc::InferRequestedOutput::Create(&out, name), "failed to create requested output");
        output_handles.emplace_back(out);
        outputs.push_back(out);
    }

    tc::InferOptions options(config_.model);
    options.model_version_ = config_.model_version;
    tc::InferResult* result_ptr = nullptr;
    Check(client_->Infer(&result_ptr, options, inputs, outputs), "inference failed");
    std::shared_ptr<tc::InferResult> result(result_ptr);
    Check(result->RequestStatus(), "inference request failed");

    // Decode each output tensor to float, mirroring the other backends: the primary
    // detection tensor is 2D/3D, the optional proto/semantic tensor is 4D.
    std::vector<std::vector<float>> storage(output_names_.size());
    std::vector<std::vector<int64_t>> shapes(output_names_.size());
    int main_idx = 0, aux_idx = -1;
    for (size_t i = 0; i < output_names_.size(); ++i) {
        std::vector<int64_t> out_shape;
        Check(result->Shape(output_names_[i], &out_shape), "failed to read output shape");
        shapes[i] = out_shape;

        const uint8_t* raw = nullptr;
        size_t bytes = 0;
        Check(result->RawData(output_names_[i], &raw, &bytes), "failed to read output data");
        if (output_fp16_[i]) {
            const size_t count = bytes / sizeof(uint16_t);
            const uint16_t* half_out = reinterpret_cast<const uint16_t*>(raw);
            storage[i].resize(count);
            for (size_t j = 0; j < count; ++j) storage[i][j] = HalfToFloat(half_out[j]);
        } else {
            const size_t count = bytes / sizeof(float);
            storage[i].assign(reinterpret_cast<const float*>(raw), reinterpret_cast<const float*>(raw) + count);
        }

        if (out_shape.size() == 4) aux_idx = static_cast<int>(i);
        else main_idx = static_cast<int>(i);
    }

    // Refine the task now that the concrete output shapes are known. Metadata shapes
    // can be dynamic (e.g. [-1, 84, -1]), which is ambiguous between detect/pose/obb;
    // the real inference result carries fixed dimensions. Preprocessing is unaffected
    // (only Classify differs, and that is unambiguous from the 2D metadata shape).
    if (config_.task == Task::Unknown) task_ = InferTask(shapes, static_cast<int>(names_.size()));

    const float* data = storage[main_idx].data();
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
            return PostprocessSegment(data, shp, storage[aux_idx].data(), shapes[aux_idx], scale, config_.conf,
                                      config_.iou, imgsz_, image.size());
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
