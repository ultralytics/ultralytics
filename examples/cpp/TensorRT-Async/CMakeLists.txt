cmake_minimum_required(VERSION 3.18)

project(yolo_tensorrt_async LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core dnn highgui imgproc videoio)
find_package(Threads REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation directory")
find_path(
    TENSORRT_INCLUDE_DIR
    NvInfer.h
    HINTS
        "${TENSORRT_ROOT}/include"
        "$ENV{TENSORRT_ROOT}/include"
        "$ENV{TensorRT_ROOT}/include"
    PATHS
        /usr/include
        /usr/include/x86_64-linux-gnu
        /usr/local/include
)
find_library(
    TENSORRT_LIBRARY
    NAMES nvinfer nvinfer_10 nvinfer_9 nvinfer_8
    HINTS
        "${TENSORRT_ROOT}/lib"
        "${TENSORRT_ROOT}/lib64"
        "$ENV{TENSORRT_ROOT}/lib"
        "$ENV{TENSORRT_ROOT}/lib64"
        "$ENV{TensorRT_ROOT}/lib"
        "$ENV{TensorRT_ROOT}/lib64"
    PATHS
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/lib
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(
        FATAL_ERROR
        "TensorRT was not found. Set -DTENSORRT_ROOT to the extracted TensorRT SDK directory."
    )
endif()

add_executable(
    yolo_tensorrt_async
    src/main.cpp
    src/tensorrt_detector.cpp
)

target_include_directories(
    yolo_tensorrt_async
    PRIVATE
        include
        ../common
        "${TENSORRT_INCLUDE_DIR}"
        ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    yolo_tensorrt_async
    PRIVATE
        "${TENSORRT_LIBRARY}"
        CUDA::cudart
        Threads::Threads
        ${OpenCV_LIBS}
)
target_compile_definitions(yolo_tensorrt_async PRIVATE NOMINMAX)

if(MSVC)
    target_compile_options(yolo_tensorrt_async PRIVATE /W4 /permissive-)
else()
    target_compile_options(yolo_tensorrt_async PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(WIN32)
    get_filename_component(TENSORRT_LIBRARY_DIR "${TENSORRT_LIBRARY}" DIRECTORY)
    find_file(
        TENSORRT_RUNTIME
        NAMES nvinfer_10.dll nvinfer_9.dll nvinfer_8.dll nvinfer.dll
        HINTS
            "${TENSORRT_ROOT}/bin"
            "${TENSORRT_ROOT}/lib"
            "${TENSORRT_LIBRARY_DIR}"
    )
    if(TENSORRT_RUNTIME)
        add_custom_command(
            TARGET yolo_tensorrt_async
            POST_BUILD
            COMMAND
                ${CMAKE_COMMAND} -E copy_if_different
                "${TENSORRT_RUNTIME}"
                "$<TARGET_FILE_DIR:yolo_tensorrt_async>"
        )
    endif()
endif()
