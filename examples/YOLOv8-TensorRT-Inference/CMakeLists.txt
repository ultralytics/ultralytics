cmake_minimum_required(VERSION 3.10)

project(trttemplate)

# set lib type
set(LIB_TYPE SHARED)
if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(LIB_TYPE STATIC)
    message("Window operating system build static library")
endif()

# set dependencies lib path
set(TensorRT_DIR "E:/lib/Tensorrt/TensorRT-10.0.1.6")
set(CUDA_DIR "E:/lib/cudalib-11.8/development")

# set TensorRT and cuda library and include directory
set(TensorRT_INCLUDE_DIR "${TensorRT_DIR}/include")
file(GLOB TensorRT_LIBS "${TensorRT_DIR}/lib/*.lib")
set(CUDA_INCLUDE_DIR "${CUDA_DIR}/include")
file(GLOB CUDA_LIBS "${CUDA_DIR}/lib/x64/*.lib")

# find dependencies library
find_package(OpenCV REQUIRED)

# include
include_directories(
    ${PROJECT_SOURCE_DIR}/TRTInfer
    ${TensorRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
# generate library
add_library(trtemplate ${LIB_TYPE}
    TRTInfer/TRTinfer.cc
    TRTInfer/utility.cc
)

target_link_libraries(trtemplate 
PRIVATE
    ${CUDA_LIBS}
    ${TensorRT_LIBS}
PUBLIC
    ${OpenCV_LIBS}
)

add_executable(example example.cc)

target_link_libraries(example
    trtemplate
)
