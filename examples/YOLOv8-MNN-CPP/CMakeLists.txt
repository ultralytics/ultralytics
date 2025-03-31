cmake_minimum_required(VERSION 3.12)
project(mnn_yolo_cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

add_executable("main" "${CMAKE_CURRENT_LIST_DIR}/main.cpp")

target_link_libraries("main" MNN MNN_Express MNNOpenCV)
