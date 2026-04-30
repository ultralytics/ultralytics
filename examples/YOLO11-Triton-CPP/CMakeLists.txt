cmake_minimum_required(VERSION 3.5)

project(YOLO11TritonCPP VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Require external Triton client directory as a parameter
if(NOT DEFINED TRITON_CLIENT_DIR)
    message(FATAL_ERROR "Please specify -DTRITON_CLIENT_DIR=/path/to/tritonclient")
endif()

# Triton-related paths
set(Protobuf_DIR "${TRITON_CLIENT_DIR}/protobuf/lib/cmake/protobuf")
set(gRPC_DIR "${TRITON_CLIENT_DIR}/grpc")
set(c-ares_DIR "${TRITON_CLIENT_DIR}/c-ares/lib/cmake/c-ares")
set(TritonClient_DIR "${TRITON_CLIENT_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "${TRITON_CLIENT_DIR}/lib/cmake/TritonCommon")

# Compiler optimizations
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c -mavx2 -O3 -ffast-math -march=native")

# OpenCV setup
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Find Triton packages
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)

# Project source files
set(PROJECT_SOURCES
    main.cpp
    inference.cpp
    inference.hpp
)

# Define executable target
add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

# Include directories
target_include_directories(${PROJECT_NAME}
    PRIVATE
    ${TRITON_CLIENT_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link directories
target_link_directories(${PROJECT_NAME}
    PRIVATE
    ${TRITON_CLIENT_DIR}/lib
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    PRIVATE
    ${OpenCV_LIBS}
    grpcclient
)
