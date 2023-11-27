#include <iostream>
#include "fastrt/utils.h"
#include "fastrt/layers.h"
#include "fastrt/embedding_head.h"

namespace fastrt {

    embedding_head::embedding_head(FastreidConfig& modelCfg) : 
        _modelCfg(modelCfg), _layerFactory(make_unique<LayerFactory>()) {}
    
    embedding_head::embedding_head(FastreidConfig& modelCfg, 
        std::unique_ptr<LayerFactory> layerFactory) : _modelCfg(modelCfg), _layerFactory(std::move(layerFactory)) {}

    ILayer* embedding_head::topology(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        /*
         * Reference: https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/heads/embedding_head.py
         */

        ILayer* pooling = _layerFactory->createPoolingLayer(_modelCfg.pooling)->addPooling(network, weightMap, input);
        TRTASSERT(pooling);

        // Hint: It's used to be "heads.bnneck.0" before Sep 10, 2020. (JDAI-CV/fast-reid)
        std::string bnneck_lname = "heads.bottleneck.0"; 
        ILayer* reduction_neck{pooling};

        if(_modelCfg.embedding_dim > 0) { 
            Weights emptywts{DataType::kFLOAT, nullptr, 0};
            reduction_neck = network->addConvolutionNd(*pooling->getOutput(0),
                _modelCfg.embedding_dim, 
                DimsHW{1, 1}, 
                weightMap["heads.bottleneck.0.weight"],             
                emptywts);
            TRTASSERT(reduction_neck); 
            bnneck_lname[bnneck_lname.size()-1] = '1';
        }
        
        IScaleLayer* bottleneck = trtxapi::addBatchNorm2d(network, weightMap, *reduction_neck->getOutput(0), bnneck_lname, 1e-5);     
        TRTASSERT(bottleneck);
        return bottleneck;
    }

}