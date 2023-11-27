#include <vector>
#include <iostream>
#include "fastrt/utils.h"
#include "fastrt/layers.h"
#include "fastrt/sbs_resnet.h"
using namespace trtxapi;

namespace fastrt {
    ILayer* backbone_sbsR18_distill::topology(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        std::string ibn{""};
        if(_modelCfg.with_ibna) {
            ibn = "a";
        }
        std::map<std::string, std::vector<std::string>> ibn_layers{ 
            {"a", {"a","a","a","a","a","a","",""}},
            {"b", {"","","b","","","","b","","","","","","","","","",}},
            {"", {16,""}}};

        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, 64, DimsHW{7, 7}, weightMap["backbone.conv1.weight"], emptywts);
        TRTASSERT(conv1);
        conv1->setStrideNd(DimsHW{2, 2});
        conv1->setPaddingNd(DimsHW{3, 3});

        IScaleLayer* bn1{nullptr};
        if (ibn == "b") {
            bn1 = addInstanceNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        // pytorch: nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
        TRTASSERT(pool1);
        pool1->setStrideNd(DimsHW{2, 2});
        pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

        ILayer* x = distill_basicBlock_ibn(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "backbone.layer1.0.", ibn_layers[ibn][0]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 64, 1, "backbone.layer1.1.", ibn_layers[ibn][1]);

        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 128, 2, "backbone.layer2.0.", ibn_layers[ibn][2]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 128, 1, "backbone.layer2.1.", ibn_layers[ibn][3]);

        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 256, 2, "backbone.layer3.0.", ibn_layers[ibn][4]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.1.", ibn_layers[ibn][5]);
       
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 512, _modelCfg.last_stride, "backbone.layer4.0.", ibn_layers[ibn][6]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 512, 512, 1, "backbone.layer4.1.", ibn_layers[ibn][7]);

        IActivationLayer* relu2 = network->addActivation(*x->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu2);
        return relu2;
    }

    ILayer* backbone_sbsR34_distill::topology(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        std::string ibn{""};
        if(_modelCfg.with_ibna) {
            ibn = "a";
        }
        std::map<std::string, std::vector<std::string>> ibn_layers{ 
            {"a", {"a","a","a","a","a","a","a","a","a","a","a","a","a","","",""}},
            {"b", {"","","b","","","","b","","","","","","","","","",}},
            {"", {16,""}}};

        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, 64, DimsHW{7, 7}, weightMap["backbone.conv1.weight"], emptywts);
        TRTASSERT(conv1);
        conv1->setStrideNd(DimsHW{2, 2});
        conv1->setPaddingNd(DimsHW{3, 3});

        IScaleLayer* bn1{nullptr};
        if (ibn == "b") {
            bn1 = addInstanceNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        // pytorch: nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
        TRTASSERT(pool1);
        pool1->setStrideNd(DimsHW{2, 2});
        pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

        ILayer* x = distill_basicBlock_ibn(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "backbone.layer1.0.", ibn_layers[ibn][0]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 64, 1, "backbone.layer1.1.", ibn_layers[ibn][1]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 64, 1, "backbone.layer1.2.", ibn_layers[ibn][2]);

        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 128, 2, "backbone.layer2.0.", ibn_layers[ibn][3]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 128, 1, "backbone.layer2.1.", ibn_layers[ibn][4]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 128, 1, "backbone.layer2.2.", ibn_layers[ibn][5]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 128, 1, "backbone.layer2.3.", ibn_layers[ibn][6]);

        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 256, 2, "backbone.layer3.0.", ibn_layers[ibn][7]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.1.", ibn_layers[ibn][8]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.2.", ibn_layers[ibn][9]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.3.", ibn_layers[ibn][10]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.4.", ibn_layers[ibn][11]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.5.", ibn_layers[ibn][12]);
       
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 512, _modelCfg.last_stride, "backbone.layer4.0.", ibn_layers[ibn][13]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 512, 512, 1, "backbone.layer4.1.", ibn_layers[ibn][14]);
        x = distill_basicBlock_ibn(network, weightMap, *x->getOutput(0), 512, 512, 1, "backbone.layer4.2.", ibn_layers[ibn][15]);

        IActivationLayer* relu2 = network->addActivation(*x->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu2);
        return relu2;
    }

    ILayer* backbone_sbsR50_distill::topology(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        std::string ibn{""};
        if(_modelCfg.with_ibna) {
            ibn = "a";
        }
        std::map<std::string, std::vector<std::string>> ibn_layers{ 
            {"a", {"a","a","a","a","a","a","a","a","a","a","a","a","a","","",""}},
            {"b", {"","","b","","","","b","","","","","","","","","",}},
            {"", {16,""}}};

        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, 64, DimsHW{7, 7}, weightMap["backbone.conv1.weight"], emptywts);
        TRTASSERT(conv1);
        conv1->setStrideNd(DimsHW{2, 2});
        conv1->setPaddingNd(DimsHW{3, 3});

        IScaleLayer* bn1{nullptr};
        if (ibn == "b") {
            bn1 = addInstanceNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        // pytorch: nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
        TRTASSERT(pool1);
        pool1->setStrideNd(DimsHW{2, 2});
        pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

        ILayer* x = distill_bottleneck_ibn(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "backbone.layer1.0.", ibn_layers[ibn][0]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 64, 1, "backbone.layer1.1.", ibn_layers[ibn][1]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 64, 1, "backbone.layer1.2.", ibn_layers[ibn][2]);

        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 128, 2, "backbone.layer2.0.", ibn_layers[ibn][3]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 128, 1, "backbone.layer2.1.", ibn_layers[ibn][4]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 128, 1, "backbone.layer2.2.", ibn_layers[ibn][5]);
        ILayer* _layer{x};
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_2.0.");
        }
        x = distill_bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 512, 128, 1, "backbone.layer2.3.", ibn_layers[ibn][6]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_2.1.");
        }

        x = distill_bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 512, 256, 2, "backbone.layer3.0.", ibn_layers[ibn][7]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "backbone.layer3.1.", ibn_layers[ibn][8]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "backbone.layer3.2.", ibn_layers[ibn][9]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "backbone.layer3.3.", ibn_layers[ibn][10]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_3.0.");
        } 
        x = distill_bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 1024, 256, 1, "backbone.layer3.4.", ibn_layers[ibn][11]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_3.1.");
        }
        x = distill_bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 1024, 256, 1, "backbone.layer3.5.", ibn_layers[ibn][12]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_3.2.");
        }

        x = distill_bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 1024, 512, _modelCfg.last_stride, "backbone.layer4.0.", ibn_layers[ibn][13]); 
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 2048, 512, 1, "backbone.layer4.1.", ibn_layers[ibn][14]);
        x = distill_bottleneck_ibn(network, weightMap, *x->getOutput(0), 2048, 512, 1, "backbone.layer4.2.", ibn_layers[ibn][15]);
        
        IActivationLayer* relu2 = network->addActivation(*x->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu2);  
        return relu2;
    }

    ILayer* backbone_sbsR34::topology(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        std::string ibn{""};
        if(_modelCfg.with_ibna) {
            ibn = "a";
        }
        std::map<std::string, std::vector<std::string>> ibn_layers{ 
            {"a", {"a","a","a","a","a","a","a","a","a","a","a","a","a","","",""}},  /* resnet34-ibna */
            {"b", {"","","b","","","","b","","","","","","","","","",}}, /* resnet34-ibnb */
            {"", {16,""}}}; /* vanilla resnet34 */

        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, 64, DimsHW{7, 7}, weightMap["backbone.conv1.weight"], emptywts);
        TRTASSERT(conv1);
        conv1->setStrideNd(DimsHW{2, 2});
        conv1->setPaddingNd(DimsHW{3, 3});

        IScaleLayer* bn1{nullptr};
        if (ibn == "b") {
            bn1 = addInstanceNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        // pytorch: nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
        TRTASSERT(pool1);
        pool1->setStrideNd(DimsHW{2, 2});
        pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

        IActivationLayer* x = basicBlock_ibn(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "backbone.layer1.0.", ibn_layers[ibn][0]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 64, 1, "backbone.layer1.1.", ibn_layers[ibn][1]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 64, 1, "backbone.layer1.2.", ibn_layers[ibn][2]);

        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 64, 128, 2, "backbone.layer2.0.", ibn_layers[ibn][3]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 128, 1, "backbone.layer2.1.", ibn_layers[ibn][4]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 128, 1, "backbone.layer2.2.", ibn_layers[ibn][5]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 128, 1, "backbone.layer2.3.", ibn_layers[ibn][6]);

        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 128, 256, 2, "backbone.layer3.0.", ibn_layers[ibn][7]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.1.", ibn_layers[ibn][8]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.2.", ibn_layers[ibn][9]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.3.", ibn_layers[ibn][10]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.4.", ibn_layers[ibn][11]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 256, 1, "backbone.layer3.5.", ibn_layers[ibn][12]);
       
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 256, 512, _modelCfg.last_stride, "backbone.layer4.0.", ibn_layers[ibn][13]); 
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 512, 512, 1, "backbone.layer4.1.", ibn_layers[ibn][14]);
        x = basicBlock_ibn(network, weightMap, *x->getOutput(0), 512, 512, 1, "backbone.layer4.2.", ibn_layers[ibn][15]);
        return x;
    }

    ILayer* backbone_sbsR50::topology(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input) {
        /*
         * Reference: https://github.com/JDAI-CV/fast-reid/blob/master/fastreid/modeling/backbones/resnet.py
         * NL layers follow by: nl_layers_per_stage = {'50x': [0, 2, 3, 0],}[depth]
         * for nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True) => pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);
         * for nn.MaxPool2d(kernel_size=3, stride=2, padding=1) replace with => pool1->setPaddingNd(DimsHW{1, 1});
         */
        std::string ibn{""};
        if(_modelCfg.with_ibna) {
            ibn = "a";
        }
        std::map<std::string, std::vector<std::string>> ibn_layers{ 
            {"a", {"a","a","a","a","a","a","a","a","a","a","a","a","a","","",""}}, /* resnet50-ibna */
            {"b", {"","","b","","","","b","","","","","","","","","",}}, /* resnet50-ibnb(not used in fastreid) */ 
            {"", {16,""}}}; /* vanilla resnet50 */

        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, 64, DimsHW{7, 7}, weightMap["backbone.conv1.weight"], emptywts);
        TRTASSERT(conv1);
        conv1->setStrideNd(DimsHW{2, 2});
        conv1->setPaddingNd(DimsHW{3, 3});

        IScaleLayer* bn1{nullptr};
        if (ibn == "b") {
            bn1 = addInstanceNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), "backbone.bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        IPoolingLayer* pool1 = network->addPoolingNd(*relu1->getOutput(0), PoolingType::kMAX, DimsHW{3, 3});
        TRTASSERT(pool1);
        pool1->setStrideNd(DimsHW{2, 2});
        pool1->setPaddingMode(PaddingMode::kEXPLICIT_ROUND_UP);

        IActivationLayer* x = bottleneck_ibn(network, weightMap, *pool1->getOutput(0), 64, 64, 1, "backbone.layer1.0.", ibn_layers[ibn][0]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 64, 1, "backbone.layer1.1.", ibn_layers[ibn][1]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 64, 1, "backbone.layer1.2.", ibn_layers[ibn][2]);

        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 256, 128, 2, "backbone.layer2.0.", ibn_layers[ibn][3]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 128, 1, "backbone.layer2.1.", ibn_layers[ibn][4]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 512, 128, 1, "backbone.layer2.2.", ibn_layers[ibn][5]);
        ILayer* _layer{x};
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_2.0.");
        }
        x = bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 512, 128, 1, "backbone.layer2.3.", ibn_layers[ibn][6]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_2.1.");
        }

        x = bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 512, 256, 2, "backbone.layer3.0.", ibn_layers[ibn][7]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "backbone.layer3.1.", ibn_layers[ibn][8]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "backbone.layer3.2.", ibn_layers[ibn][9]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 1024, 256, 1, "backbone.layer3.3.", ibn_layers[ibn][10]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_3.0.");
        } 
        x = bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 1024, 256, 1, "backbone.layer3.4.", ibn_layers[ibn][11]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_3.1.");
        }
        x = bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 1024, 256, 1, "backbone.layer3.5.", ibn_layers[ibn][12]);
        _layer = x;
        if(_modelCfg.with_nl) {
            _layer = Non_local(network, weightMap, *x->getOutput(0), "backbone.NL_3.2.");
        }

        x = bottleneck_ibn(network, weightMap, *_layer->getOutput(0), 1024, 512, _modelCfg.last_stride, "backbone.layer4.0.", ibn_layers[ibn][13]); 
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 2048, 512, 1, "backbone.layer4.1.", ibn_layers[ibn][14]);
        x = bottleneck_ibn(network, weightMap, *x->getOutput(0), 2048, 512, 1, "backbone.layer4.2.", ibn_layers[ibn][15]);
        return x;
    }

}