#pragma once

#include "module.h"
#include "utils.h"
#include "holder.h"
#include "layers.h"
#include "struct.h"
#include "InferenceEngine.h"

#include <memory>
#include <vector>
#include <opencv2/opencv.hpp>
extern Logger gLogger;
using namespace trt;
using namespace trtxapi;

namespace fastrt {

    class Model {
    public:
        Model(const trt::ModelConfig &modelcfg, 
            const std::string input_name="input", 
            const std::string output_name="output");

        virtual ~Model() = default;

        /* 
         * Serialize TRT Engine
         * @engine_file: save serialized engine as engine_file
         * @modules: sequential modules(variadic length). (e.g., backbone1 + backbone2 + head, backbone + head, backbone)
         */ 
        bool serializeEngine(const std::string engine_file, 
            const std::initializer_list<std::unique_ptr<Module>>& modules);

        bool deserializeEngine(const std::string engine_file);

        /* Support batch inference */
        bool inference(std::vector<cv::Mat> &input); 

        /* 
         * Access the memory allocated by cudaMallocHost. (It's on CPU side) 
         * Use this after each inference.
         */ 
        float* getOutput(); 

        /* 
         * Output buffer size
         */ 
        int getOutputSize(); 

        /* 
         * Cuda device id
         * You may need this in multi-thread/multi-engine inference
         */ 
        int getDeviceID(); 

    private:
        TensorRTHolder<ICudaEngine> createEngine(IBuilder* builder,
            const std::initializer_list<std::unique_ptr<Module>>& modules);

        virtual void preprocessing_cpu(const cv::Mat& img, float* const data, const std::size_t stride) = 0;
        virtual ITensor* preprocessing_gpu(INetworkDefinition* network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor* input) { return nullptr; };

    private:
        DataType _dt{DataType::kFLOAT};
        trt::EngineConfig _engineCfg;
        std::unique_ptr<trt::InferenceEngine> _inferEngine{nullptr};
    };
}
