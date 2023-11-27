#pragma once

#include <map>
#include <chrono>
#include <memory>
#include <vector>
#include <fstream>
#include <iostream>
#include <cassert>
#include <string.h> 

#include <dirent.h>
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "fastrt/struct.h"

#define CHECK(status)                             \
    do                                            \
    {                                             \
        auto ret = (status);                      \
        if (ret != 0)                             \
        {                                         \
            std::cout << "Cuda failure: " << ret; \
            abort();                              \
        }                                         \
    } while (0)

#define TRTASSERT assert

using Time = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

template<typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

namespace io {
    std::vector<std::string> fileGlob(const std::string& pattern);
}

static inline int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {

            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}

namespace trt {
    /* 
     * Load weights from files shared with TensorRT samples.
     * TensorRT weight files have a simple space delimited format:
     * [type] [size] <data x size in hex>
     */ 
    std::map<std::string, nvinfer1::Weights> loadWeights(const std::string file);

    std::ostream& operator<<(std::ostream& os, const ModelConfig& modelCfg);
}

namespace fastrt {
    std::ostream& operator<<(std::ostream& os, const FastreidConfig& fastreidCfg);
}