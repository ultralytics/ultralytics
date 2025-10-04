#include <NvInfer.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string.h>


using namespace nvinfer1;

class Logger : public ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        // suppress info-level messages
        if (severity <= Severity::kERROR) {
            std::cout << "Errors:" << msg << std::endl;
        }
        if (severity == Severity::kINTERNAL_ERROR) {
            std::cerr << "Internal Error: " << msg << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }
} logger;

inline void checkCudaErrorCode(cudaError_t code) {
    if (code != cudaSuccess) {
        std::string errMsg = "CUDA operation failed with code: " + std::to_string(code) + " (" + cudaGetErrorName(code) +
                             "), with message: " + cudaGetErrorString(code);
        throw std::runtime_error(errMsg);
    }
}

class Buffer {
    public:
        Buffer(ICudaEngine* engine, cudaStream_t &stream) {
            for (int i = 0; i < engine->getNbIOTensors(); i++) {
                buffers.push_back(nullptr);
                IOTensorNames.push_back(engine->getIOTensorName(i));
                auto tensorType = engine->getTensorIOMode(IOTensorNames[i].c_str());
                auto tensorShape = engine->getTensorShape(IOTensorNames[i].c_str());
                if (tensorType == TensorIOMode::kINPUT) {
                    inputDims.emplace_back(tensorShape.d[1], tensorShape.d[2], tensorShape.d[3]);
                    checkCudaErrorCode(cudaMallocAsync(&buffers[i], tensorShape.d[1] * tensorShape.d[2] * tensorShape.d[3] * sizeof(float), stream));
                    for (int j = 0; j < tensorShape.nbDims; j++) {
                        inputLength *= tensorShape.d[j];
                    }
                }
                else if (tensorType == TensorIOMode::kOUTPUT) {
                    outputDims.push_back(tensorShape);
                    for (int j = 0; j < tensorShape.nbDims; j++) {
                        outputLength *= tensorShape.d[j];
                    }
                    checkCudaErrorCode(cudaMallocAsync(&buffers[i], outputLength * sizeof(float), stream));
                }
                else {
                    std::cerr << "Tensor type is kNONE, program might misbehave or crash." << std::endl;
                }
            }
        }

        void getBuffers(std::vector<void*> &resbuf) {
            resbuf = buffers;
        }

        void getIOTensorNames(std::vector<std::string> &resIOTensorNames) {
            resIOTensorNames = IOTensorNames;
        }

        int32_t getOutputLength() {
            return outputLength;
        }

        int32_t getInputLength() {
            return inputLength;
        }
    private:
        std::vector<void*> buffers;
        std::vector<std::string> IOTensorNames;
        std::vector<Dims3> inputDims;
        std::vector<Dims> outputDims;
        int32_t outputLength = 1;
        int32_t inputLength = 1;
};