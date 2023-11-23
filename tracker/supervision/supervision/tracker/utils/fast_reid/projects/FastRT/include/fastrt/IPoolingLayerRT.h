#pragma once

#include <map>
#include "struct.h"
#include "NvInfer.h"
using namespace nvinfer1;

namespace fastrt {

    class IPoolingLayerRT {
    public:
        IPoolingLayerRT() = default;
        virtual ~IPoolingLayerRT() = default;

        virtual ILayer* addPooling(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input) = 0; 
    };

}