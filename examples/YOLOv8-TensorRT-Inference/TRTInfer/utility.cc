#include "utility.h"
namespace utility
{

    void *safeCudaMalloc(size_t memSize)
    {
        void *deviceMem;
        cudaError_t status = cudaMalloc(&deviceMem, memSize);
        if (status != cudaSuccess)
        {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(status) << std::endl;
            return nullptr;
        }
        return deviceMem;
    }
    bool safeCudaFree(void *&ptr)
    {
        if (ptr != nullptr)
        {
            std::cerr << "Pointer is already nullptr." << std::endl;
            return false;
        }
        cudaError_t result = cudaFree(ptr);
        if (result != cudaSuccess)
        {
            std::cerr << "Failed to free CUDA memory: " << cudaGetErrorString(result) << std::endl;
            return false;
        }
        std::cout << "CUDA memory successfully freed." << std::endl;
        ptr = nullptr; // 释放后将指针设置为nullptr
        return true;
    }
    int getTypebytes(const nvinfer1::DataType &type)
    {
        switch (type)
        {
        case nvinfer1::DataType::kBF16:
            return 2;
        case nvinfer1::DataType::kBOOL:
            return 1;
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kFP8:
            return 1;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT32:
            return 4;
        case nvinfer1::DataType::kINT4:
            return 0.5;
        case nvinfer1::DataType::kINT64:
            return 8;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kUINT8:
            return 1;
        }
    }

    int getTensorbytes(const nvinfer1::Dims &dim, const nvinfer1::DataType &type)
    {
        int64_t size = 1;
        for (int i = 0; i < dim.nbDims; i++)
            size *= dim.d[i];
        return size * getTypebytes(type);
    }

    //
    nvinfer1::DataType typeCv2Rt(const int &cv_type)
    {
        switch (cv_type)
        {
        case CV_8U:
            return nvinfer1::DataType::kUINT8;
        case CV_8S:
            return nvinfer1::DataType::kINT8;
        case CV_16F:
            return nvinfer1::DataType::kHALF;
        case CV_32S:
            return nvinfer1::DataType::kINT32;
        case CV_32F:
            return nvinfer1::DataType::kFLOAT;
        }
        return nvinfer1::DataType::kFLOAT;
    }

    //
    int typeRt2Cv(const nvinfer1::DataType &rt_type)
    {
        switch (rt_type)
        {
        case nvinfer1::DataType::kFLOAT:
            return CV_32F;
        case nvinfer1::DataType::kHALF:
            return CV_16F;
        case nvinfer1::DataType::kINT32:
            return CV_32S;
        case nvinfer1::DataType::kINT8:
            return CV_8S;
        case nvinfer1::DataType::kUINT8:
            return CV_8U;
        }

        return CV_32F;
    }
}