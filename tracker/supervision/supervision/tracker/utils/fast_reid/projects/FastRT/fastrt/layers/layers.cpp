#include <limits>
#include <vector>
#include <iostream>
#include "fastrt/utils.h"
#include "fastrt/layers.h"

namespace trtxapi {

    IActivationLayer* addMinClamp(INetworkDefinition* network, ITensor& input, const float min) {
        IActivationLayer* clip = network->addActivation(input, ActivationType::kCLIP);
        TRTASSERT(clip);
        clip->setAlpha(min);
        clip->setBeta(std::numeric_limits<float>::max());    
        return clip;
    }

    ITensor* addDiv255(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor* input, const std::string lname) {
        Weights Div_225{ DataType::kFLOAT, nullptr, 3 };
        float *wgt = reinterpret_cast<float*>(malloc(sizeof(float) * 3));
        std::fill_n(wgt, 3, 255.0f); 
        Div_225.values = wgt;
        weightMap[lname + ".div"] = Div_225;
        IConstantLayer* d = network->addConstant(Dims3{ 3, 1, 1 }, Div_225);
        IElementWiseLayer* div255 = network->addElementWise(*input, *d->getOutput(0), ElementWiseOperation::kDIV);
        return div255->getOutput(0);
    }

    ITensor* addMeanStd(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor* input, const std::string lname, const float* mean, const float* std, const bool div255) {
        ITensor* tensor_holder{input};
        if (div255) {
            tensor_holder = addDiv255(network, weightMap, input, lname);
        }
        Weights Mean{ DataType::kFLOAT, nullptr, 3 };
        Mean.values = mean;
        IConstantLayer* m = network->addConstant(Dims3{ 3, 1, 1 }, Mean);
        IElementWiseLayer* sub_mean = network->addElementWise(*tensor_holder, *m->getOutput(0), ElementWiseOperation::kSUB);
        if (std != nullptr) {
            Weights Std{ DataType::kFLOAT, nullptr, 3 };
            Std.values = std;
            IConstantLayer* s = network->addConstant(Dims3{ 3, 1, 1 }, Std);
            IElementWiseLayer* std_mean = network->addElementWise(*sub_mean->getOutput(0), *s->getOutput(0), ElementWiseOperation::kDIV);
            return std_mean->getOutput(0);
        } else {
            return sub_mean->getOutput(0);
        }
    }

    IScaleLayer* addBatchNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string lname, const float eps) {
        float *gamma = (float*)weightMap[lname + ".weight"].values;
        float *beta = (float*)weightMap[lname + ".bias"].values;
        float *mean = (float*)weightMap[lname + ".running_mean"].values;
        float *var = (float*)weightMap[lname + ".running_var"].values;
        int len = weightMap[lname + ".running_var"].count;

        float *scval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            scval[i] = gamma[i] / sqrt(var[i] + eps);
        }
        Weights wscale{DataType::kFLOAT, scval, len};

        float *shval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            shval[i] = beta[i] - mean[i] * gamma[i] / sqrt(var[i] + eps);
        }
        Weights wshift{DataType::kFLOAT, shval, len};

        float *pval = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        for (int i = 0; i < len; i++) {
            pval[i] = 1.0;
        }
        Weights wpower{DataType::kFLOAT, pval, len};

        weightMap[lname + ".scale"] = wscale;
        weightMap[lname + ".shift"] = wshift;
        weightMap[lname + ".power"] = wpower;
        IScaleLayer* scale_1 = network->addScale(input, ScaleMode::kCHANNEL, wshift, wscale, wpower);
        TRTASSERT(scale_1);
        return scale_1;
    }

    IScaleLayer* addInstanceNorm2d(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string lname, const float eps) {
        int len = weightMap[lname + ".weight"].count;
        IReduceLayer* reduce1 = network->addReduce(input, 
            ReduceOperation::kAVG,
            6, 
            true);
        TRTASSERT(reduce1);

        IElementWiseLayer* ew1 = network->addElementWise(input, 
            *reduce1->getOutput(0),
            ElementWiseOperation::kSUB);  
        TRTASSERT(ew1);

        const static float pval1[3]{0.0, 1.0, 2.0};   
        Weights wshift1{DataType::kFLOAT, pval1, 1};
        Weights wscale1{DataType::kFLOAT, pval1+1, 1};
        Weights wpower1{DataType::kFLOAT, pval1+2, 1};

        IScaleLayer* scale1 = network->addScale(
            *ew1->getOutput(0), 
            ScaleMode::kUNIFORM,
            wshift1,  
            wscale1,  
            wpower1); 
        TRTASSERT(scale1);

        IReduceLayer* reduce2 = network->addReduce(
            *scale1->getOutput(0), 
            ReduceOperation::kAVG,
            6, 
            true);
        TRTASSERT(reduce2);

        const static float pval2[3]{eps, 1.0, 0.5}; 
        Weights wshift2{DataType::kFLOAT, pval2, 1};
        Weights wscale2{DataType::kFLOAT, pval2+1, 1};
        Weights wpower2{DataType::kFLOAT, pval2+2, 1};
        
        IScaleLayer* scale2 = network->addScale(
            *reduce2->getOutput(0), 
            ScaleMode::kUNIFORM,
            wshift2,  
            wscale2,  
            wpower2);
        TRTASSERT(scale2);

        IElementWiseLayer* ew2 = network->addElementWise(*ew1->getOutput(0), 
            *scale2->getOutput(0),
            ElementWiseOperation::kDIV); 
        TRTASSERT(ew2);

        float* pval3 = reinterpret_cast<float*>(malloc(sizeof(float) * len));
        std::fill_n(pval3, len, 1.0); 
        Weights wpower3{DataType::kFLOAT, pval3, len};
        weightMap[lname + ".power3"] = wpower3;

        IScaleLayer* scale3 = network->addScale(
            *ew2->getOutput(0), 
            ScaleMode::kCHANNEL,
            weightMap[lname + ".bias"], 
            weightMap[lname + ".weight"],  
            wpower3); 
        TRTASSERT(scale3);
        return scale3;
    }

    IConcatenationLayer* addIBN(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string lname) {
        Dims spliteDims = input.getDimensions();
        ISliceLayer *split1 = network->addSlice(input, 
            Dims3{0, 0, 0}, 
            Dims3{spliteDims.d[0]/2, spliteDims.d[1], spliteDims.d[2]}, 
            Dims3{1, 1, 1});
        TRTASSERT(split1);

        ISliceLayer *split2 = network->addSlice(input, 
            Dims3{spliteDims.d[0]/2, 0, 0}, 
            Dims3{spliteDims.d[0]/2, spliteDims.d[1], spliteDims.d[2]}, 
            Dims3{1, 1, 1});
        TRTASSERT(split2);

        auto in1 = addInstanceNorm2d(network, weightMap, *split1->getOutput(0), lname + "IN", 1e-5);
        auto bn1 = addBatchNorm2d(network, weightMap, *split2->getOutput(0), lname + "BN", 1e-5);

        ITensor* tensor1[] = {in1->getOutput(0), bn1->getOutput(0)};
        auto cat1 = network->addConcatenation(tensor1, 2);
        TRTASSERT(cat1);
        return cat1;
    }

    IActivationLayer* basicBlock_ibn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, const int inch, const int outch, const int stride, const std::string lname, const std::string ibn) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};

        IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
        TRTASSERT(conv1);
        conv1->setStrideNd(DimsHW{stride, stride});
        conv1->setPaddingNd(DimsHW{1, 1});

        ILayer* bn1{conv1};
        if (ibn == "a") {
            bn1 = addIBN(network, weightMap, *conv1->getOutput(0), lname + "bn1.");
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
        TRTASSERT(conv2);
        conv2->setPaddingNd(DimsHW{1, 1});

        IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

        IElementWiseLayer* ew1;
        if (inch != outch) {
            IConvolutionLayer* conv3 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
            TRTASSERT(conv3);
            conv3->setStrideNd(DimsHW{stride, stride});
            IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
            ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
        } else {
            ew1 = network->addElementWise(input, *bn2->getOutput(0), ElementWiseOperation::kSUM);
        }
        ILayer* in1{ew1};
        if (ibn == "b") {
            in1 = addInstanceNorm2d(network, weightMap, *ew1->getOutput(0), lname + "IN", 1e-5);
        }

        IActivationLayer* relu2 = network->addActivation(*in1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu2);
        return relu2;
    }

    IActivationLayer* bottleneck_ibn(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, const int inch, const int outch, const int stride, const std::string lname, const std::string ibn) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};
        IConvolutionLayer* conv1 = network->addConvolutionNd(input, outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
        TRTASSERT(conv1);

        ILayer* bn1{conv1};
        if (ibn == "a") {
            bn1 = addIBN(network, weightMap, *conv1->getOutput(0), lname + "bn1.");
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
        TRTASSERT(conv2);
        conv2->setStrideNd(DimsHW{stride, stride});
        conv2->setPaddingNd(DimsHW{1, 1});

        IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

        IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu2);

        IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
        TRTASSERT(conv3);

        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

        IElementWiseLayer* ew1;
        if (stride != 1 || inch != outch * 4) {
            IConvolutionLayer* conv4 = network->addConvolutionNd(input, outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
            TRTASSERT(conv4);
            conv4->setStrideNd(DimsHW{stride, stride});

            IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
            ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        } else {
            ew1 = network->addElementWise(input, *bn3->getOutput(0), ElementWiseOperation::kSUM);
        }

        ILayer* in1{ew1};
        if (ibn == "b") {
            in1 = addInstanceNorm2d(network, weightMap, *ew1->getOutput(0), lname + "IN", 1e-5);
        }
        IActivationLayer* relu3 = network->addActivation(*in1->getOutput(0), ActivationType::kRELU);

        TRTASSERT(relu3);
        return relu3;
    }

    ILayer* distill_basicBlock_ibn(INetworkDefinition *network, std::map<std::string, Weights>& weightMap, ITensor& input, const int inch, const int outch, const int stride, const std::string lname, const std::string ibn) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};

        IActivationLayer* relu_identity = network->addActivation(input, ActivationType::kRELU);
        TRTASSERT(relu_identity);

        IConvolutionLayer* conv1 = network->addConvolutionNd(*relu_identity->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv1.weight"], emptywts);
        TRTASSERT(conv1);
        conv1->setStrideNd(DimsHW{stride, stride});
        conv1->setPaddingNd(DimsHW{1, 1});

        ILayer* bn1{conv1};
        if (ibn == "a") {
            bn1 = addIBN(network, weightMap, *conv1->getOutput(0), lname + "bn1.");
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
        TRTASSERT(conv2);
        conv2->setPaddingNd(DimsHW{1, 1});

        IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

        IElementWiseLayer* ew1;
        if (inch != outch) {
            IConvolutionLayer* conv3 = network->addConvolutionNd(*relu_identity->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
            TRTASSERT(conv3);
            conv3->setStrideNd(DimsHW{stride, stride});
            IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "downsample.1", 1e-5);
            ew1 = network->addElementWise(*bn3->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
        } else {
            ew1 = network->addElementWise(*relu_identity->getOutput(0), *bn2->getOutput(0), ElementWiseOperation::kSUM);
        }
        ILayer* in1{ew1};
        if (ibn == "b") {
            in1 = addInstanceNorm2d(network, weightMap, *ew1->getOutput(0), lname + "IN", 1e-5);
        }
        return in1;
    }

    ILayer* distill_bottleneck_ibn(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, const int inch, const int outch, const int stride, const std::string lname, const std::string ibn) {
        Weights emptywts{DataType::kFLOAT, nullptr, 0};

        IActivationLayer* relu_identity = network->addActivation(input, ActivationType::kRELU);
        TRTASSERT(relu_identity);

        IConvolutionLayer* conv1 = network->addConvolutionNd(*relu_identity->getOutput(0), outch, DimsHW{1, 1}, weightMap[lname + "conv1.weight"], emptywts);
        TRTASSERT(conv1);

        ILayer* bn1{conv1};
        if (ibn == "a") {
            bn1 = addIBN(network, weightMap, *conv1->getOutput(0), lname + "bn1.");
        } else {
            bn1 = addBatchNorm2d(network, weightMap, *conv1->getOutput(0), lname + "bn1", 1e-5);
        }
        IActivationLayer* relu1 = network->addActivation(*bn1->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu1);

        IConvolutionLayer* conv2 = network->addConvolutionNd(*relu1->getOutput(0), outch, DimsHW{3, 3}, weightMap[lname + "conv2.weight"], emptywts);
        TRTASSERT(conv2);
        conv2->setStrideNd(DimsHW{stride, stride});
        conv2->setPaddingNd(DimsHW{1, 1});

        IScaleLayer* bn2 = addBatchNorm2d(network, weightMap, *conv2->getOutput(0), lname + "bn2", 1e-5);

        IActivationLayer* relu2 = network->addActivation(*bn2->getOutput(0), ActivationType::kRELU);
        TRTASSERT(relu2);

        IConvolutionLayer* conv3 = network->addConvolutionNd(*relu2->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "conv3.weight"], emptywts);
        TRTASSERT(conv3);

        IScaleLayer* bn3 = addBatchNorm2d(network, weightMap, *conv3->getOutput(0), lname + "bn3", 1e-5);

        IElementWiseLayer* ew1;
        if (stride != 1 || inch != outch * 4) {
            IConvolutionLayer* conv4 = network->addConvolutionNd(*relu_identity->getOutput(0), outch * 4, DimsHW{1, 1}, weightMap[lname + "downsample.0.weight"], emptywts);
            TRTASSERT(conv4);
            conv4->setStrideNd(DimsHW{stride, stride});

            IScaleLayer* bn4 = addBatchNorm2d(network, weightMap, *conv4->getOutput(0), lname + "downsample.1", 1e-5);
            ew1 = network->addElementWise(*bn4->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        } else {
            ew1 = network->addElementWise(*relu_identity->getOutput(0), *bn3->getOutput(0), ElementWiseOperation::kSUM);
        }

        ILayer* in1{ew1};
        if (ibn == "b") {
            in1 = addInstanceNorm2d(network, weightMap, *ew1->getOutput(0), lname + "IN", 1e-5);
        }
        return in1;
    }

    IShuffleLayer* addShuffle2(INetworkDefinition* network, ITensor& input, const Dims dims, const Permutation pmt, const bool reshape_first) {
        IShuffleLayer* shuffleLayer = network->addShuffle(input);
        TRTASSERT(shuffleLayer);
        if (reshape_first) {
            shuffleLayer->setReshapeDimensions(dims);
            shuffleLayer->setSecondTranspose(pmt); 
        } else {
            shuffleLayer->setFirstTranspose(pmt); 
            shuffleLayer->setReshapeDimensions(dims);
        }
        return shuffleLayer;
    }

    IElementWiseLayer* Non_local(INetworkDefinition* network, std::map<std::string, Weights>& weightMap, ITensor& input, const std::string lname, const int reduc_ratio) {
        int in_channel = input.getDimensions().d[0];
        /* Hint: fast-reid use "in_channel / reduc_ratio" during Sep 10, 2020 to Dec 7, 2020 */
        //int inter_channels = in_channel / reduc_ratio; 
        int inter_channels = 1; 
        std::cout << "[Non_local] inter_channels: " << inter_channels << std::endl;
        IConvolutionLayer* g = network->addConvolutionNd(input, inter_channels, DimsHW{1, 1}, weightMap[ lname + "g.weight"],  weightMap[lname + "g.bias"]);
        TRTASSERT(g); 

        auto g_permute = addShuffle2(network, *g->getOutput(0), Dims2{g->getOutput(0)->getDimensions().d[0], -1}, Permutation{1, 0}, true);
        IConvolutionLayer* theta = network->addConvolutionNd(input, inter_channels, DimsHW{1, 1}, weightMap[lname + "theta.weight"],  weightMap[lname + "theta.bias"]);
        TRTASSERT(theta); 

        auto theta_permute = addShuffle2(network, *theta->getOutput(0), Dims2{theta->getOutput(0)->getDimensions().d[0], -1}, Permutation{1, 0}, true);
        IConvolutionLayer* phi = network->addConvolutionNd(input, inter_channels, DimsHW{1, 1}, weightMap[lname + "phi.weight"],  weightMap[lname + "phi.bias"]);
        TRTASSERT(phi);  

        IShuffleLayer* phi_view = network->addShuffle(*phi->getOutput(0));
        TRTASSERT(phi_view);
        phi_view->setReshapeDimensions(Dims2{phi->getOutput(0)->getDimensions().d[0], -1});

        IMatrixMultiplyLayer *f = network->addMatrixMultiply(*theta_permute->getOutput(0), MatrixOperation::kNONE, *phi_view->getOutput(0), MatrixOperation::kNONE);
        int N = f->getOutput(0)->getDimensions().d[f->getOutput(0)->getDimensions().nbDims-1];

        float* pval =  reinterpret_cast<float*>(malloc(sizeof(float) * N * N));
        std::fill_n(pval, N*N, N); 
        Weights dem{DataType::kFLOAT, pval, N*N};
        weightMap[lname + ".dem"] = dem;

        auto dem_n = network->addConstant(Dims2(N, N), dem);
        IElementWiseLayer* f_div_C = network->addElementWise(*f->getOutput(0), 
            *dem_n->getOutput(0),
            ElementWiseOperation::kDIV);  
        TRTASSERT(f_div_C);

        IMatrixMultiplyLayer *y = network->addMatrixMultiply(*f_div_C->getOutput(0), MatrixOperation::kNONE, *g_permute->getOutput(0), MatrixOperation::kNONE);
        IShuffleLayer* y_permute = addShuffle2(network, *y->getOutput(0), Dims3{inter_channels, input.getDimensions().d[1], input.getDimensions().d[2]}, Permutation{1, 0}, false);
        TRTASSERT(y_permute);
        IConvolutionLayer* w_conv = network->addConvolutionNd(*y_permute->getOutput(0), in_channel, DimsHW{1, 1}, weightMap[lname + "W.0.weight"], weightMap[lname + "W.0.bias"]);
        TRTASSERT(w_conv);
        IScaleLayer* w_bn = addBatchNorm2d(network, weightMap, *w_conv->getOutput(0), lname + "W.1", 1e-5);
        TRTASSERT(w_bn);

        // z = W_y + x
        IElementWiseLayer* z = network->addElementWise(*w_bn->getOutput(0), 
            input,
            ElementWiseOperation::kSUM);  
        TRTASSERT(z);
        return z;
    }

    IPoolingLayer* addAdaptiveAvgPool2d(INetworkDefinition* network, ITensor& input, const DimsHW output_dim) {
        Dims input_dims = input.getDimensions();
        TRTASSERT((input_dims.nbDims == 3));
        // stride_dim = floor(input_dim/output_dim)
        DimsHW stride_dims{(int)(input_dims.d[1]/output_dim.h()), 
            (int)(input_dims.d[2]/output_dim.w())};
        // kernel_dims = input_dim -(output_dim-1)*stride_dim
        DimsHW kernel_dims{input_dims.d[1] - (output_dim.h()-1) * stride_dims.h(), 
            input_dims.d[2] - (output_dim.w()-1) * stride_dims.w()};
        IPoolingLayer* avgpool = network->addPoolingNd(input, PoolingType::kAVERAGE, kernel_dims);
        TRTASSERT(avgpool);
        avgpool->setStrideNd(stride_dims);
        return avgpool;
    }

    IScaleLayer* addGeneralizedMeanPooling(INetworkDefinition* network, ITensor& input, const float norm, const DimsHW output_dim, const float eps) {
        TRTASSERT((norm > 0.f));
        // x = x.clamp(min=eps)
        IActivationLayer* clamp1 = addMinClamp(network, input, eps);
        // (x)^norm
        const static float pval1[3]{0.0, 1.0, norm};   
        Weights wshift1{DataType::kFLOAT, pval1, 1};
        Weights wscale1{DataType::kFLOAT, pval1+1, 1};
        Weights wpower1{DataType::kFLOAT, pval1+2, 1};

        IScaleLayer* scale1 = network->addScale(
            *clamp1->getOutput(0), 
            ScaleMode::kUNIFORM,
            wshift1,
            wscale1,
            wpower1);
        TRTASSERT(scale1); 

        IPoolingLayer* ada_avg_pool = addAdaptiveAvgPool2d(network, *scale1->getOutput(0));
        TRTASSERT(ada_avg_pool);

        // (ada_avg_pool)^(1/norm)
        const static float pval2[3]{0.0, 1.0, 1.f/norm};   
        Weights wshift2{DataType::kFLOAT, pval2, 1};
        Weights wscale2{DataType::kFLOAT, pval2+1, 1};
        Weights wpower2{DataType::kFLOAT, pval2+2, 1};

        IScaleLayer* scale2 = network->addScale(
            *ada_avg_pool->getOutput(0), 
            ScaleMode::kUNIFORM,
            wshift2,  
            wscale2,   
            wpower2); 
        TRTASSERT(scale2);
        return scale2;
    }
}