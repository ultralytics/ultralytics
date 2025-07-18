cmake_minimum_required(VERSION 3.18)
project(YOLOTensorRTInference LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(OpenCV PATHS "D:/opencv-4.12.0/build" REQUIRED)
add_library(tensorrt INTERFACE)
target_include_directories(tensorrt INTERFACE "D:/TensorRT-10.12.0.36/include" ${CMAKE_CURRENT_SOURCE_DIR})
target_link_directories(tensorrt INTERFACE "D:/TensorRT-10.12.0.36/lib")
target_link_libraries(tensorrt INTERFACE "D:/TensorRT-10.12.0.36/lib/nvinfer_10.lib" "D:/TensorRT-10.12.0.36/lib/nvinfer_plugin_10.lib" cudart)
include_directories("D:/opencv-4.12.0/include")

add_executable(YOLOTensorRTInference main.cu)
set_target_properties(YOLOTensorRTInference PROPERTIES LINKER_LANGUAGE CUDA)
target_link_libraries(YOLOTensorRTInference ${OpenCV_LIBS} tensorrt)