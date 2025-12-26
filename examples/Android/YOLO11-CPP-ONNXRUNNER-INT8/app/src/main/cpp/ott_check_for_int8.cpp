//
// Created by wangh on 2025/11/4.
//

#include "ott_check_for_int8.h"
#include <unordered_set>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>

OttCheckForInt8::OttCheckForInt8() {
    env = new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Default");
    session = nullptr;
    ott_input_w = 640;
    ott_input_h = 640;
}

OttCheckForInt8::~OttCheckForInt8() {
    input_names.clear();
    output_names.clear();
    input_node_dims.clear();
    output_node_dims.clear();
    class_names.clear();
    if (session != nullptr) {
        delete session;
        session = nullptr;
    }
    if (env != nullptr) {
        delete env;
        env = nullptr;
    }
}

std::string OttCheckForInt8::RemoveOuterBrackets(const std::string& input) {
    if (input.length() < 2) {
        return input;
    }
    if (input.front() == '{' && input.back() == '}') {
        return input.substr(1, input.length() - 2);
    }
    return input;
}

std::vector<std::string> OttCheckForInt8::Split(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

bool OttCheckForInt8::Init(const std::vector<char>& model_data, int bit_core_number) {
    if (env == nullptr) {
        return false;
    }
    if (session != nullptr) {
        return true;
    }

    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
#ifdef ENABLE_NNAPI

    uint32_t nnapi_flags = 0;
    nnapi_flags |= NNAPI_FLAG_USE_FP16;
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Nnapi(so, nnapi_flags));
#endif
#ifdef ENABLE_XNNPACK
    std::unordered_map<std::string,std::string> xnnopt;
    xnnopt.insert(std::pair<std::string,std::string>("intra_op_num_threads", std::to_string(bit_core_number)));
    so.AppendExecutionProvider("XNNPACK", xnnopt);
#endif
    try {
        session = new Ort::Session(*env, model_data.data(), model_data.size(), so);
    } catch (const std::exception& e) {
        std::cerr << "Session creation failed: " << e.what() << std::endl;
        return false;
    }

    // prepare meta to get class and input info
    Ort::ModelMetadata meta_data = session->GetModelMetadata();
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr prop_value = meta_data.LookupCustomMetadataMapAllocated("names", allocator);
    if (!prop_value) {
        std::cerr << "No class names found in model metadata" << std::endl;
        return false;
    }

    std::string labels_str = prop_value.get();
    std::string classString = RemoveOuterBrackets(labels_str);
    std::vector<std::string> classTmp = Split(classString, ',');
    for (const auto& it : classTmp) {
        std::vector<std::string> kv = Split(it, ':');
        if (kv.size() < 2) continue;
        const std::string& v = kv[1];
        std::vector<std::string> cut1 = Split(v, '\'');
        if (cut1.size() < 2) continue;
        const std::string& cut1ans = cut1[1];
        std::vector<std::string> cut2 = Split(cut1ans, '\'');
        if (cut2.empty()) continue;
        class_names.push_back(cut2[0]);
    }

    // get input and output info
    size_t det_num_input_nodes = session->GetInputCount();
    size_t det_num_output_nodes = session->GetOutputCount();

    // deal input (NHWC format:1xHxWx3)
    for (int i = 0; i < det_num_input_nodes; i++) {
        auto input_name = session->GetInputNameAllocated(i, allocator);
        input_names.push_back(input_name.get());

        Ort::TypeInfo input_type_info = session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }

    // deal output
    for (int i = 0; i < det_num_output_nodes; i++) {
        auto output_name = session->GetOutputNameAllocated(i, allocator);
        output_names.push_back(output_name.get());

        Ort::TypeInfo output_type_info = session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }

    // get H and W (shape is[1, H, W, 3])
    if (!input_node_dims.empty() && input_node_dims[0].size() == 4) {
        ott_input_h = static_cast<int>(input_node_dims[0][1]);
        ott_input_w = static_cast<int>(input_node_dims[0][2]);
    }
    if (ott_input_w <= 0) ott_input_w = 640;
    if (ott_input_h <= 0) ott_input_h = 640;

    return true;
}

cv::Mat OttCheckForInt8::PreprocessResizePad(const cv::Mat& input, float& scale_ratio, int& pad_x, int& pad_y) const {
    int src_w = input.cols;
    int src_h = input.rows;
    cv::Mat dst(ott_input_h, ott_input_w, input.type(), cv::Scalar(114, 114, 114));  // 常用填充值

    scale_ratio = std::min(static_cast<float>(ott_input_w) / src_w,
                           static_cast<float>(ott_input_h) / src_h);
    int resize_w = static_cast<int>(src_w * scale_ratio);
    int resize_h = static_cast<int>(src_h * scale_ratio);

    pad_x = (ott_input_w - resize_w) / 2;
    pad_y = (ott_input_h - resize_h) / 2;

    if (resize_w > 0 && resize_h > 0) {
        cv::Mat resized;
        cv::resize(input, resized, cv::Size(resize_w, resize_h), 0, 0, cv::INTER_LINEAR);
        resized.copyTo(dst(cv::Rect(pad_x, pad_y, resize_w, resize_h)));
    }

    return dst;
}

bool OttCheckForInt8::RunOnnx(cv::Mat& inputData, std::vector<Ort::Value>& outData) {
    if (session == nullptr) {
        std::cerr << "Session not initialized" << std::endl;
        return false;
    }

    // NHWC input shape：[1, H, W, 3]
    std::array<int64_t, 4> input_shape_info{1, ott_input_h, ott_input_w, 3};
    size_t tpixels = ott_input_h * ott_input_w * 3;  // H*W*3 is count of item

    // check input（CV_32F, HxWx3）
    if (inputData.type() != CV_32FC3 ||
        inputData.rows != ott_input_h ||
        inputData.cols != ott_input_w) {
        std::cerr << "Invalid input data format for NHWC" << std::endl;
        return false;
    }

    try {
        auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                allocator_info,
                inputData.ptr<float>(),  // use HxWx3 float ptr
                tpixels,
                input_shape_info.data(),
                input_shape_info.size()
        );

        std::vector<const char*> input_node_names = {input_names[0].c_str()};
        std::vector<const char*> output_node_names = {output_names[0].c_str()};

        outData = session->Run(Ort::RunOptions{nullptr},
                               input_node_names.data(),
                               &input_tensor, 1,
                               output_node_names.data(), 1);
    } catch (const std::exception& e) {
        std::cerr << "ONNX inference failed: " << e.what() << std::endl;
        return false;
    }

    return true;
}

std::vector<OttCheckAns> OttCheckForInt8::DealOnnxOutWithExportNmsFalseFaster(
        std::vector<Ort::Value>& outData,
        const float minscore,
        const float x_factor,
        const float y_factor,
        const int pad_x,
        const int pad_y) {
    if (outData.empty()) {
        return {};
    }

    auto output_tensor_info = outData[0].GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    if (output_dims.size() != 3) {
        return {};
    }

    const int num_boxes = static_cast<int>(output_dims[2]);
    const int feat_dim = static_cast<int>(output_dims[1]);
    const int num_classes = feat_dim - 4;
    const float* pdata = outData[0].GetTensorData<float>();

    std::vector<cv::Rect> boxes;
    std::vector<int> classIds;
    std::vector<float> confidences;
    boxes.reserve(num_boxes);
    classIds.reserve(num_boxes);
    confidences.reserve(num_boxes);

    std::vector<std::vector<int>> class_indices(num_classes);

    for (int i = 0; i < num_boxes; ++i) {
        float cx = pdata[0 * num_boxes + i] * ott_input_w;  // ott_input_w=640
        float cy = pdata[1 * num_boxes + i] * ott_input_h;  // ott_input_h=640
        float ow = pdata[2 * num_boxes + i] * ott_input_w;
        float oh = pdata[3 * num_boxes + i] * ott_input_h;

        float max_score = -1.0f;
        int classId = -1;
        for (int c = 0; c < num_classes; ++c) {
            float score = pdata[(4 + c) * num_boxes + i];
            if (score > max_score) {
                max_score = score;
                classId = c;
            }
        }

        if (max_score > minscore && classId >= 0 && classId < static_cast<int>(class_names.size())) {
            int x = static_cast<int>((cx - 0.5f * ow - pad_x) / x_factor);
            int y = static_cast<int>((cy - 0.5f * oh - pad_y) / y_factor);
            int w = static_cast<int>(ow / x_factor);
            int h = static_cast<int>(oh / y_factor);

            x = std::max(0, x);
            y = std::max(0, y);
            w = std::max(1, w);
            h = std::max(1, h);

            const int current_idx = boxes.size();
            boxes.emplace_back(x, y, w, h);
            classIds.push_back(classId);
            confidences.push_back(max_score);

            if (classId < num_classes) {
                class_indices[classId].push_back(current_idx);
            }
        }
    }

    if (boxes.empty()) {
        return {};
    }

    std::vector<int> final_indices;
    final_indices.reserve(boxes.size());

    const float nms_thre = 0.7f;
    for (int cls = 0; cls < num_classes; ++cls) {
        const auto& indices = class_indices[cls];
        if (indices.empty()) continue;

        std::vector<cv::Rect> cls_boxes;
        std::vector<float> cls_confs;
        cls_boxes.reserve(indices.size());
        cls_confs.reserve(indices.size());

        for (int idx : indices) {
            cls_boxes.push_back(boxes[idx]);
            cls_confs.push_back(confidences[idx]);
        }

        std::vector<int> nms_indices;
        cv::dnn::NMSBoxes(cls_boxes, cls_confs, minscore, nms_thre, nms_indices);

        for (int ni : nms_indices) {
            final_indices.push_back(indices[ni]);
        }
    }

    std::vector<OttCheckAns> retans;
    retans.reserve(final_indices.size());
    for (int idx : final_indices) {
        const cv::Rect& box = boxes[idx];
        int id = classIds[idx];
        float score = confidences[idx];
        const std::string& boxName = class_names[id];

        OttCheckAns tmpAns;
        tmpAns.boxName = boxName;
        tmpAns.score = score;
        tmpAns.startPoint = {static_cast<double>(box.x), static_cast<double>(box.y)};
        tmpAns.endPoint = {static_cast<double>(box.x + box.width), static_cast<double>(box.y + box.height)};
        retans.push_back(std::move(tmpAns));
    }

    return retans;
}

bool OttCheckForInt8::Process(const cv::Mat& input, std::vector<OttCheckAns>& retans) {
    if (env == nullptr || session == nullptr || input.empty()) {
        return false;
    }
    retans.clear();

    float scale_ratio;
    int pad_x, pad_y;

    cv::Mat preprocessed = PreprocessResizePad(input, scale_ratio, pad_x, pad_y);
    if (preprocessed.empty()) {
        return false;
    }

    cv::Mat rgb_mat;
    cv::cvtColor(preprocessed, rgb_mat, cv::COLOR_BGR2RGB);

    rgb_mat.convertTo(rgb_mat, CV_32F, 1.0 / 255.0);

    std::vector<Ort::Value> outData;
    if (!RunOnnx(rgb_mat, outData)) {
        return false;
    }

    const float conf_thre = 0.5f;
    retans = DealOnnxOutWithExportNmsFalseFaster(outData, conf_thre, scale_ratio, scale_ratio, pad_x, pad_y);
    outData.clear();

    return true;
}