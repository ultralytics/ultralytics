cmake_minimum_required(VERSION 3.18)
project(YOLOTensorRTInference LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fPIC")

find_package(OpenCV PATHS "/path/to/OpenCV/Install" REQUIRED)
add_library(tensorrt INTERFACE)
target_include_directories(tensorrt INTERFACE "/path/to/TensorRT-10.12.0.36/include" ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(tensorrt INTERFACE "/path/to/TensorRT-10.12.0.36/lib")
if(WIN32)
    target_link_libraries(tensorrt INTERFACE "/path/to/TensorRT-10.12.0.36/lib/nvinfer_10.lib" "/path/to/TensorRT-10.12.0.36/lib/nvinfer_plugin_10.lib" cudart)
elseif(UNIX)
    target_link_libraries(tensorrt INTERFACE nvinfer nvinfer_plugin cudart)
endif()

include_directories("/path/to/OpenCV/Install/opencv/include")

add_executable(YOLOTensorRTInference main.cu)
set_target_properties(YOLOTensorRTInference PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(YOLOTensorRTInference ${OpenCV_LIBS} tensorrt)
