cmake_minimum_required(VERSION 3.12)
project(yolo_mnn)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

include_directories(${CMAKE_CURRENT_LIST_DIR}/include/)

link_directories(${CMAKE_CURRENT_LIST_DIR}/libs)

add_executable("yolo_mnn" "${CMAKE_CURRENT_LIST_DIR}/main.cpp")
add_executable("yolo_mnn_interpreter" "${CMAKE_CURRENT_LIST_DIR}/main_interpreter.cpp")

target_link_libraries("yolo_mnn" MNN MNN_Express MNNOpenCV)
target_link_libraries("yolo_mnn_interpreter" MNN MNN_Express MNNOpenCV)
