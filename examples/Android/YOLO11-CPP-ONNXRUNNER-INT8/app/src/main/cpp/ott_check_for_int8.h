//
// Created by wangh on 2025/11/4.
//

#ifndef ONNX_RUNNER_OTT_CHECK_FOR_INT8_H
#define ONNX_RUNNER_OTT_CHECK_FOR_INT8_H

#include <vector>
#include <string>
#include "struct_def.h"
#include <android/bitmap.h>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <nnapi_provider_factory.h>

class OttCheckForInt8 {
public:
    OttCheckForInt8();
    ~OttCheckForInt8();
    bool Init(const std::vector<char>& model_data, int bit_core_number);
    std::vector<std::string> get_class_names(){
        return class_names;
    }
    // 注意input格式为BGR
    bool Process(const cv::Mat& input, std::vector<OttCheckAns>& retans);
private:
    static std::string RemoveOuterBrackets(const std::string& input);
    static std::vector<std::string> Split(const std::string& input, char delimiter);
    cv::Mat PreprocessResizePad(const cv::Mat& input, float& scale_ratio, int& pad_x, int& pad_y) const;
    bool RunOnnx(cv::Mat& inputData, std::vector<Ort::Value>& outData);
    std::vector<OttCheckAns> DealOnnxOutWithExportNmsFalseFaster(std::vector<Ort::Value>& outData, const float minscore, const float x_factor, const float y_factor, const int pad_x, const int pad_y);
private:
    Ort::Env *env = nullptr;
    Ort::Session *session = nullptr;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<std::vector<int64_t>> input_node_dims; // >=1 outputs
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
    int ott_input_w = -1;
    int ott_input_h = -1;
    std::vector<std::string> class_names;
};


#endif //ONNX_RUNNER_OTT_CHECK_FOR_INT8_H
