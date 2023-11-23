#include "NvInfer.h"
#include "fastrt/IPoolingLayerRT.h"
using namespace nvinfer1;

namespace fastrt {

    class MaxPool : public IPoolingLayerRT {
    public:
        MaxPool() = default;
        ~MaxPool() = default;

        ILayer* addPooling(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap,
            ITensor& input) override;
    };

    class AvgPool : public IPoolingLayerRT {
    public:
        AvgPool() = default;
        ~AvgPool() = default;

        ILayer* addPooling(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap,
            ITensor& input) override;
    };

    class GemPool : public IPoolingLayerRT {
    public:
        GemPool() = default;
        ~GemPool() = default;

        ILayer* addPooling(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap,
            ITensor& input) override;
    };

    class GemPoolP : public IPoolingLayerRT {
    public:
        GemPoolP() = default;
        ~GemPoolP() = default;

        ILayer* addPooling(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap,
            ITensor& input) override;
    };
}