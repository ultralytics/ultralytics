#pragma once

#include <map>
#include "struct.h"
#include "module.h"
#include "NvInfer.h"
using namespace nvinfer1;

namespace fastrt {
    class backbone_sbsR18_distill : public Module {
    private:
        FastreidConfig& _modelCfg;
    public:
        backbone_sbsR18_distill(FastreidConfig& modelCfg) : _modelCfg(modelCfg){}
        ~backbone_sbsR18_distill() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input) override; 
    };

    class backbone_sbsR34_distill : public Module {
    private:
        FastreidConfig& _modelCfg;
    public:
        backbone_sbsR34_distill(FastreidConfig& modelCfg) : _modelCfg(modelCfg) {}
        ~backbone_sbsR34_distill() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input) override; 
    };

    class backbone_sbsR50_distill : public Module { 
    private:
        FastreidConfig& _modelCfg;
    public:
        backbone_sbsR50_distill(FastreidConfig& modelCfg) : _modelCfg(modelCfg) {}
        ~backbone_sbsR50_distill() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input) override;
    };

    class backbone_sbsR34 : public Module {
    private:
        FastreidConfig& _modelCfg;
    public:
        backbone_sbsR34(FastreidConfig& modelCfg) : _modelCfg(modelCfg) {}
        ~backbone_sbsR34() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input) override;
    };

    class backbone_sbsR50 : public Module {
    private:
        FastreidConfig& _modelCfg;
    public:
        backbone_sbsR50(FastreidConfig& modelCfg) : _modelCfg(modelCfg) {}
        ~backbone_sbsR50() = default;
        ILayer* topology(INetworkDefinition *network, 
            std::map<std::string, Weights>& weightMap, 
            ITensor& input) override;
    };
     
}