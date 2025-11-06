cmake_minimum_required(VERSION 3.22.1)

project("onnx_ott_runner")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(PROJECT_ALL_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../..") # 最上层根目录
set(THIS_PROJECT_BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..") # 本工程源码根目录
#set(ONNXRUNTIME_BASE_DIR "${THIS_PROJECT_BASE_DIR}")
#set(ONNXRUNTIME_SO_File "${ONNXRUNTIME_BASE_DIR}/jni/${ANDROID_ABI}/libonnxruntime.so")
set(ONNXRUNTIME_SO_File "${PROJECT_ALL_BASE_DIR}/onnxruntime_sdk/jni/${ANDROID_ABI}/libonnxruntime.so")
#file(GLOB ONNXRUNTIME_STATIC_LIBS ${PROJECT_ALL_BASE_DIR}/onnxruntime_sdk/jni/${ANDROID_ABI}/*.a)
#set(ONNXRUNTIME_NEED_SHARED_LIB "${PROJECT_ALL_BASE_DIR}/onnxruntime_sdk/jni/${ANDROID_ABI}/libc.so")
set(ONNXRUNTIME_INCLUDE_DIR "${PROJECT_ALL_BASE_DIR}/onnxruntime_sdk/headers")

set(OPENCV_BASE_DIR "${PROJECT_ALL_BASE_DIR}/OpenCV-android-sdk/sdk/native")
set(OpenCV_DIR "${OPENCV_BASE_DIR}/jni/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

message("CMAKE_C_COMPILER_ID = ${CMAKE_C_COMPILER_ID}")
message("Before project command: CMAKE_CXX_COMPILER_ID = ${CMAKE_CXX_COMPILER_ID}")

message("CMAKE_C_COMPILER = ${CMAKE_C_COMPILER}")
message("CMAKE_CXX_COMPILER = ${CMAKE_CXX_COMPILER}")
message("CMAKE_CXX_COMPILER_VERSION = ${CMAKE_CXX_COMPILER_VERSION}")

set(SIMPLIFY "ON")
if(SIMPLIFY STREQUAL "ON")
    MESSAGE("catkin_make small so")
    SET(CMAKE_BUILD_TYPE "Release")
    # 删除调试符号
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -s")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -s")

    # 使用gc-section优化
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffunction-sections -fdata-sections")
    set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS}    -Wl,--gc-sections")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,--gc-sections")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--gc-sections")
    set(CMAKE_C_FLAGS     "${CMAKE_C_FLAGS}   -fvisibility=hidden -ffast-math -Wl,-z,max-page-size=16384")
    set(CMAKE_CXX_FLAGS   "${CMAKE_CXX_FLAGS} -fvisibility=hidden -ffast-math -Wl,-z,max-page-size=16384")
    set(CMAKE_C_FLAGS  "${CMAKE_C_FLAGS}   -O3 -flto -funroll-loops -march=armv8.2-a+fp16")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -O3 -flto -funroll-loops -march=armv8.2-a+fp16")
endif()



find_library( # Sets the name of the path variable.
        log-lib
        log)

find_library(
        android-lib
        android)

find_library(
        android-graphics
        jnigraphics
)


add_library(${CMAKE_PROJECT_NAME} SHARED
    # List C/C++ source files with relative paths to this CMakeLists.txt.
        ott_check_for_int8.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})


target_link_libraries(${CMAKE_PROJECT_NAME}
        PRIVATE
        -Wl,--exclude-libs,ALL
        ${ONNXRUNTIME_SO_File}
#        ${ONNXRUNTIME_NEED_SHARED_LIB}
#        ${ONNXRUNTIME_STATIC_LIBS}
        ${OpenCV_LIBS}
        ${android-graphics}
        ${android-lib}
        ${log-lib}
)