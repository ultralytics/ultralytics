#include <glob.h>
#include <vector>
#include "fastrt/utils.h"

namespace io {

    std::vector<std::string> fileGlob(const std::string& pattern){
        glob_t glob_result;
        glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
        std::vector<std::string> files;
        for (size_t i = 0;i < glob_result.gl_pathc; ++i){
            files.push_back(std::string(glob_result.gl_pathv[i]));
        }
        globfree(&glob_result);
        return files;
    }

}

namespace trt {

    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file) {
        std::cout << "[Loading weights]: " << file << std::endl;
        std::map<std::string, nvinfer1::Weights> weightMap;

        // Open weights file
        std::ifstream input(file);
        if(!input.is_open()) throw std::runtime_error("Unable to load weight file.");
        
        // Read number of weight blobs
        int32_t count;
        input >> count;
        if(count <= 0) throw std::runtime_error("Invalid weight map file.");
        
        while (count--) {
            nvinfer1::Weights wt{nvinfer1::DataType::kFLOAT, nullptr, 0};
            uint32_t size;

            // Read name and type of blob
            std::string name;
            input >> name >> std::dec >> size;
            wt.type = nvinfer1::DataType::kFLOAT;

            // Load blob
            uint32_t* val = reinterpret_cast<uint32_t*>(malloc(sizeof(val) * size));
            for (uint32_t x = 0, y = size; x < y; ++x) {
                input >> std::hex >> val[x];
            }
            wt.values = val;
            wt.count = size;
            weightMap[name] = wt;
        }
        return weightMap;
    }

    std::ostream& operator<<(std::ostream& os, const ModelConfig& modelCfg) {
        os << "\tweights_path: "    << modelCfg.weights_path      << "\n\t"
            << "max_batch_size: "   << modelCfg.max_batch_size    << "\n\t"
            << "input_h: "          << modelCfg.input_h           << "\n\t"
            << "input_w: "          << modelCfg.input_w           << "\n\t"
            << "output_size: "      << modelCfg.output_size       << "\n\t"
            << "device_id: "        << modelCfg.device_id         << "\n";
        return os;   
    }
    
}

namespace fastrt {

    const std::string BackboneTypetoString(FastreidBackboneType value) {
    #define X(a, b) b,
        static std::vector<std::string> table{ FASTBACKBONE_TABLE };
    #undef X
        return table[value];
    }

    const std::string HeadTypetoString(FastreidHeadType value) {
    #define X(a, b) b,
        static std::vector<std::string> table{ FASTHEAD_TABLE };
    #undef X
        return table[value];
    }

    const std::string PoolingTypetoString(FastreidPoolingType value) {
    #define X(a, b) b,
        static std::vector<std::string> table{ FASTPOOLING_TABLE };
    #undef X
        return table[value];
    }

    std::ostream& operator<<(std::ostream& os, const FastreidConfig& fastreidCfg) {
        os << "\tbackbone: "            << BackboneTypetoString(fastreidCfg.backbone) << "\n\t"
            << "head: "                 << HeadTypetoString(fastreidCfg.head)         << "\n\t"
            << "pooling: "              << PoolingTypetoString(fastreidCfg.pooling)   << "\n\t"
            << "last_stride: "          << fastreidCfg.last_stride                    << "\n\t"
            << "with_ibna: "            << fastreidCfg.with_ibna                      << "\n\t"
            << "with_nl: "              << fastreidCfg.with_nl                        << "\n\t"
            << "embedding_dim: "        << fastreidCfg.embedding_dim                  << "\n";
        return os;   
    } 

}