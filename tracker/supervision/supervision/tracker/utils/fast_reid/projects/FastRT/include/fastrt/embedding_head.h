#pragma once

#include <map>
#include "NvInfer.h"
#include "fastrt/module.h"
#include "fastrt/struct.h"
#include "fastrt/factory.h"
using namespace nvinfer1;

namespace fastrt {

    class embedding_head : public Module {
    private:
        FastreidConfig& _modelCfg;
        std::unique_ptr<LayerFactory> _layerFactory;

    public:
        embedding_head(FastreidConfig& modelCfg);
        embedding_head(FastreidConfig& modelCfg, std::unique_ptr<LayerFactory> layerFactory);
        ~embedding_head() = default;

        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap,
            ITensor& input) override;
    };

}