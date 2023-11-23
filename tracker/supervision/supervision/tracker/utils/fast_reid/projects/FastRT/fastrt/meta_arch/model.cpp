#include "fastrt/model.h"
#include "fastrt/calibrator.h"

#ifdef BUILD_INT8
#include "fastrt/config.h"
#endif 

namespace fastrt {

    Model::Model(const trt::ModelConfig &modelcfg, const std::string input_name, const std::string output_name) {
        
        _engineCfg.weights_path = modelcfg.weights_path;
        _engineCfg.max_batch_size = modelcfg.max_batch_size;
        _engineCfg.input_h = modelcfg.input_h;
        _engineCfg.input_w = modelcfg.input_w;
        _engineCfg.output_size = modelcfg.output_size;
        _engineCfg.device_id = modelcfg.device_id;

        _engineCfg.input_name = input_name;
        _engineCfg.output_name = output_name;       
        _engineCfg.trtModelStream = nullptr;
        _engineCfg.stream_size = 0;
    };

    bool Model::serializeEngine(const std::string engine_file, const std::initializer_list<std::unique_ptr<Module>>& modules) {

        /* Create builder */  
        auto builder = make_holder(createInferBuilder(gLogger));

        /* Create model to populate the network, then set the outputs and create an engine */ 
        auto engine = createEngine(builder.get(), modules);
        TRTASSERT(engine.get());

        /* Serialize the engine */ 
        auto modelStream = make_holder(engine->serialize());
        TRTASSERT(modelStream.get());

        std::ofstream p(engine_file, std::ios::binary | std::ios::out);
        if (!p) {
            std::cerr << "could not open plan output file" << std::endl;
            return false;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        std::cout << "[Save serialized engine]: " << engine_file << std::endl;
        return true;
    }

    TensorRTHolder<ICudaEngine> Model::createEngine(IBuilder* builder, const std::initializer_list<std::unique_ptr<Module>>& modules) {

        auto network = make_holder(builder->createNetworkV2(0U));
        auto config = make_holder(builder->createBuilderConfig());
        auto data = network->addInput(_engineCfg.input_name.c_str(), _dt, Dims3{3, _engineCfg.input_h, _engineCfg.input_w});
        TRTASSERT(data);

        auto weightMap = loadWeights(_engineCfg.weights_path);

        /* Preprocessing */
        auto input = preprocessing_gpu(network.get(), weightMap, data);
        if (!input) input = data;

        /* Modeling */
        ILayer* output{nullptr};
        for(auto& sequential_module: modules) {
            output = sequential_module->topology(network.get(), weightMap, *input);
            TRTASSERT(output);
            input = output->getOutput(0);
        }

        /* Set output */
        output->getOutput(0)->setName(_engineCfg.output_name.c_str());
        network->markOutput(*output->getOutput(0));

        /* Build engine */ 
        builder->setMaxBatchSize(_engineCfg.max_batch_size);
        config->setMaxWorkspaceSize(1 << 20);
#if defined(BUILD_FP16) && defined(BUILD_INT8)
        std::cout << "Flag confilct! BUILD_FP16 and BUILD_INT8 can't be both True!" << std::endl;
        return null;
#endif 
#if defined(BUILD_FP16)
        std::cout << "[Build fp16]" << std::endl;
        config->setFlag(BuilderFlag::kFP16);
#elif defined(BUILD_INT8)
        std::cout << "[Build int8]" << std::endl;
        std::cout << "Your platform support int8: " << (builder->platformHasFastInt8() ? "true" : "false") << std::endl;
        TRTASSERT(builder->platformHasFastInt8());
        config->setFlag(BuilderFlag::kINT8);
        Int8EntropyCalibrator2* calibrator = new Int8EntropyCalibrator2(1, _engineCfg.input_w, _engineCfg.input_h, 
            INT8_CALIBRATE_DATASET_PATH.c_str(), "int8calib.table", _engineCfg.input_name.c_str());
        config->setInt8Calibrator(calibrator);
#endif 
        auto engine = make_holder(builder->buildEngineWithConfig(*network, *config));
        std::cout << "[TRT engine build out]" << std::endl;

        for (auto& mem : weightMap) {
            free((void*) (mem.second.values));
        }
        return engine;
    }

    bool Model::deserializeEngine(const std::string engine_file) {
        std::ifstream file(engine_file, std::ios::binary | std::ios::in);
        if (file.good()) {
            file.seekg(0, file.end);
            _engineCfg.stream_size = file.tellg();
            file.seekg(0, file.beg);
            _engineCfg.trtModelStream = std::shared_ptr<char>( new char[_engineCfg.stream_size], []( char* ptr ){ delete [] ptr; } );
            TRTASSERT(_engineCfg.trtModelStream.get());
            file.read(_engineCfg.trtModelStream.get(), _engineCfg.stream_size);
            file.close();
    
            _inferEngine = make_unique<trt::InferenceEngine>(_engineCfg);
            return true;
        }
        return false;
    }

    bool Model::inference(std::vector<cv::Mat> &input) {
        if (_inferEngine != nullptr) {
            const std::size_t stride = _engineCfg.input_h * _engineCfg.input_w;
            return _inferEngine.get()->doInference(input.size(), 
                [&](float* data) {
                    for(const auto &img : input) {
                        preprocessing_cpu(img, data, stride);
                        data += 3 * stride;
                    }
                }
            );
        } else {
            return false;
        }
    }

    float* Model::getOutput() { 
        if(_inferEngine != nullptr) 
            return _inferEngine.get()->getOutput(); 
        return nullptr;
    }

    int Model::getOutputSize() { 
        return _engineCfg.output_size; 
    }

    int Model::getDeviceID() { 
        return _engineCfg.device_id; 
    }
}