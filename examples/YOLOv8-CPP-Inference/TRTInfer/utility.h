#ifndef UTILITY_H
#define UTILITY_H
#include <cuda_runtime_api.h>
#include <iostream>
#include <NvInfer.h>
#include <opencv2/core.hpp>
#include "config.h"

namespace utility
{
    /**
     *@brief Safely Allocate GPU Memory
     *@param memSize The required number of GPU memory bytes for
     *@return The address header of the GPU memory address block allocated by
     */
    CPP_API void *safeCudaMalloc(size_t memSize);

    /**
     *@brief Safely Release GPU Memory
     *@param ptr Address header of  GPU address block
     *@return Does  release a successful flag
     */
    CPP_API bool safeCudaFree(void *&ptr);

    /**
     *@brief Get the byte count of TensorRT basic types
     *@param type The basic types of  tensorrt
     *@return byte count
     */
    CPP_API int getTypebytes(const nvinfer1::DataType &type);

    /**
     *@brief Input the dimensions and basic types of a matrix, and output the total number of bytes in this matrix
     *@param dim Dimension of  dim matrix
     *@param type The basic data types of  TensorRT
     */
    CPP_API int getTensorbytes(const nvinfer1::Dims &dim, const nvinfer1::DataType &type);

    /**
     * @brief convert the cv type to nvidia tensorrt type
     * @param cv_type the type of opencv
     * @return nvidia type ,default float32
     */
    CPP_API nvinfer1::DataType typeCv2Rt(const int &cv_type);

    /**
     * @brief convert the nvidia tensorrt type to cv type
     * @param rt_type the type of nvidia tensorrt
     * @return cv type ,default CV_32F
     */
    CPP_API int typeRt2Cv(const nvinfer1::DataType &rt_type);
}

#endif