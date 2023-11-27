#pragma once

#include "model.h"
#include "struct.h"
#include <memory>
#include <opencv2/opencv.hpp>
using namespace trtxapi;

namespace fastrt {

    class Baseline : public Model {
    public:
        Baseline(const trt::ModelConfig &modelcfg,
            const std::string input_name = "data",
            const std::string output_name = "reid_embd");
        ~Baseline() = default;
    
    private:
        void preprocessing_cpu(const cv::Mat& img, float* const data, const std::size_t stride);
        ITensor* preprocessing_gpu(INetworkDefinition* network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor* input); 
    };
}