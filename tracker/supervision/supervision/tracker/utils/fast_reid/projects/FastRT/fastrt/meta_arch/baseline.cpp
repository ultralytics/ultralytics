#include "fastrt/layers.h"
#include "fastrt/baseline.h"

namespace fastrt {

    Baseline::Baseline(const trt::ModelConfig &modelcfg, const std::string input_name, const std::string output_name) 
        : Model(modelcfg, input_name, output_name) {}

    void Baseline::preprocessing_cpu(const cv::Mat& img, float* const data, const std::size_t stride) {
        /* Normalization & BGR->RGB */
        for (std::size_t i = 0; i < stride; ++i) { 
            data[i] = img.at<cv::Vec3b>(i)[2]; 
            data[i + stride] = img.at<cv::Vec3b>(i)[1];
            data[i + (stride<<1)] = img.at<cv::Vec3b>(i)[0];
        }
    }

    ITensor* Baseline::preprocessing_gpu(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor* input) {
        /* Standardization */
        static const float mean[3] = {123.675, 116.28, 103.53};
        static const float std[3] = {58.395, 57.120000000000005, 57.375};
        return addMeanStd(network, weightMap, input, "", mean, std, false); // true for div 255
    }

}