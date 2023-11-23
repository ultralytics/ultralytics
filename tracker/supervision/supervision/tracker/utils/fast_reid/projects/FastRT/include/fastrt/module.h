#pragma once

#include <map>
#include "struct.h"
#include "NvInfer.h"
using namespace nvinfer1;

namespace fastrt {

    class Module {
    public:
        Module() = default;
        virtual ~Module() = default;

        virtual ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input) = 0; 
    };

}