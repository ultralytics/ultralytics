cmake_minimum_required(VERSION 3.20)

project(Yolov8CPPInference VERSION 0.2)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# set dependencies lib path
set(TensorRT_DIR "E:/lib/Tensorrt/TensorRT-10.0.1.6")
set(CUDA_TOOLKIT_ROOT_DIR "E:/lib/cudalib-11.8/development")


# set lib type
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message("Window operating system build SHARED library")

    # set TensorRT and cuda library and include directory
    set(TensorRT_INCLUDE_DIR "${TensorRT_DIR}/include")
    file(GLOB TensorRT_LIBS "${TensorRT_DIR}/lib/*.lib")
    set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/*.lib")

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message("Linux operating system build SHARED library")
    # CUDA(old)
    # set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    # find_package(CUDA 11 REQUIRED)
    # set(CMAKE_CUDA_STANDARD 11)
    # set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # set TensorRT and cuda library and include directory
    set(TensorRT_INCLUDE_DIR "${TensorRT_DIR}/include")
    file(GLOB TensorRT_LIBS "${TensorRT_DIR}/targets/x86_64-linux-gnu/lib/stubs/*.so")
    set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib64/*.so")
endif()





# OpenCV
find_package(OpenCV REQUIRED)
include_directories(
    ${PROJECT_SOURCE_DIR}/TRTInfer
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
# !OpenCV


# generate library
add_library(trtemplate SHARED
    TRTInfer/TRTinfer.cpp
    TRTInfer/utility.cpp
)
target_link_libraries(trtemplate 
PRIVATE
    ${CUDA_LIBS}
    ${TensorRT_LIBS}
PUBLIC
    ${OpenCV_LIBS}
)
# opencv onnx example files
add_library(Inferdll SHARED
    TRTInfer/inference.h
    TRTInfer/inference.cpp
)
target_link_libraries(Inferdll 
    ${OpenCV_LIBS}
)

# opencv onnx example
add_executable(OnnxExample OnnxExample.cpp)
target_link_libraries(OnnxExample 
    Inferdll
)

# tensorrt example
add_executable(TrtExample TrtExample.cpp)
target_link_libraries(TrtExample
    trtemplate
)