#pragma once

#include <map>
#include <math.h>
#include <assert.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
using namespace nvinfer1;

namespace trtxapi {

    IActivationLayer* addMinClamp(INetworkDefinition* network, 
        ITensor& input, 
        const float min);

    ITensor* addDiv255(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor* input,
        const std::string lname);
        
    ITensor* addMeanStd(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor* input, 
        const std::string lname,
        const float* mean, 
        const float* std, 
        const bool div255);

    IScaleLayer* addBatchNorm2d(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const std::string lname, 
        const float eps);

    IScaleLayer* addInstanceNorm2d(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const std::string lname, 
        const float eps);

    IConcatenationLayer* addIBN(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const std::string lname);

    IActivationLayer* basicBlock_ibn(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const int inch, 
        const int outch,
        const int stride, 
        const std::string lname, 
        const std::string ibn);

    IActivationLayer* bottleneck_ibn(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const int inch, 
        const int outch,
        const int stride, 
        const std::string lname, 
        const std::string ibn);

    ILayer* distill_basicBlock_ibn(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const int inch, 
        const int outch,
        const int stride, 
        const std::string lname, 
        const std::string ibn);

    ILayer* distill_bottleneck_ibn(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const int inch, 
        const int outch,
        const int stride, 
        const std::string lname, 
        const std::string ibn);

    IShuffleLayer* addShuffle2(INetworkDefinition* network, 
        ITensor& input, 
        const Dims dims, 
        const Permutation pmt, 
        const bool reshape_first);

    IElementWiseLayer* Non_local(INetworkDefinition* network, 
        std::map<std::string, Weights>& weightMap, 
        ITensor& input, 
        const std::string lname, 
        const int reduc_ratio = 2);

    IPoolingLayer* addAdaptiveAvgPool2d(INetworkDefinition* network, 
        ITensor& input, 
        const DimsHW output_dim = DimsHW{1,1});

    IScaleLayer* addGeneralizedMeanPooling(INetworkDefinition* network, 
        ITensor& input, 
        const float norm = 3.f, 
        const DimsHW output_dim = DimsHW{1,1}, 
        const float eps = 1e-6);
}