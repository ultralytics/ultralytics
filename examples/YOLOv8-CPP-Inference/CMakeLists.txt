cmake_minimum_required(VERSION 3.20)

project(Yolov8CPPInference VERSION 0.2)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

option(WITH_TENSORRT "Build tensorrt example ?" OFF)

# set dependencies lib path
if(WITH_TENSORRT)
    message("Build TensorRT example!")
    set(TensorRT_DIR "/usr/local/tensorrt-10.0.1.6")
endif()
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")


# set lib type
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    message("Window operating system build SHARED library")
    if(WITH_TENSORRT)
        # set TensorRT and cuda library and include directory
        set(TensorRT_INCLUDE_DIR "${TensorRT_DIR}/include")
        file(GLOB TensorRT_LIBS "${TensorRT_DIR}/lib/*.lib")
    endif()
    set(CUDA_INCLUDE_DIR "${CUDA_TOOLKIT_ROOT_DIR}/include")
    file(GLOB CUDA_LIBS "${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/*.lib")

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message("Linux operating system build SHARED library")
    # CUDA(old)
    # set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
    # find_package(CUDA 11 REQUIRED)
    # set(CMAKE_CUDA_STANDARD 11)
    # set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(WITH_TENSORRT)
    # set TensorRT and cuda library and include directory
        set(TensorRT_INCLUDE_DIR "${TensorRT_DIR}/include")
        file(GLOB TensorRT_LIBS "${TensorRT_DIR}/targets/x86_64-linux-gnu/lib/stubs/*.so")
    endif()
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

if(WITH_TENSORRT)
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
endif()

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
if(WITH_TENSORRT)
    # tensorrt example
    add_executable(TrtExample TrtExample.cpp)
    target_link_libraries(TrtExample
        trtemplate
    )
endif()