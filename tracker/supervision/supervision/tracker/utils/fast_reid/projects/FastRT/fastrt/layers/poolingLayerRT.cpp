#include <iostream>
#include "fastrt/layers.h"
#include "poolingLayerRT.h"

namespace fastrt {

    ILayer* MaxPool::addPooling(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        ILayer* pooling = network->addPoolingNd(input, PoolingType::kMAX, DimsHW{input.getDimensions().d[1], input.getDimensions().d[2]});       
        auto p = dynamic_cast<nvinfer1::IPoolingLayer*>(pooling);
        if(p) p->setStrideNd(DimsHW{input.getDimensions().d[1], input.getDimensions().d[2]});
        else std::cout << "Downcasting failed." << std::endl; 
        return pooling;
    }
    
    ILayer* AvgPool::addPooling(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        ILayer* pooling = network->addPoolingNd(input, PoolingType::kAVERAGE, DimsHW{input.getDimensions().d[1], input.getDimensions().d[2]});
        auto p = dynamic_cast<IPoolingLayer*>(pooling);
        if(p) p->setStrideNd(DimsHW{input.getDimensions().d[1], input.getDimensions().d[2]});
        else std::cout << "Downcasting failed." << std::endl; 
        return pooling;
    }

    ILayer* GemPool::addPooling(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        return trtxapi::addGeneralizedMeanPooling(network, input); 
    }

    ILayer* GemPoolP::addPooling(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        return trtxapi::addGeneralizedMeanPooling(network, input, *(float*)weightMap["heads.pool_layer.p"].values); 
    }    

}