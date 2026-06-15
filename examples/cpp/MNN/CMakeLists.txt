cmake_minimum_required(VERSION 3.12)
project(yolo_mnn)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

# Point these at your built MNN. By default they use the include/ and libs/ folders
# inside this example (see README.md for how to populate them from an MNN build).
set(MNN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include" CACHE PATH "Directory containing MNN/Interpreter.hpp")
set(MNN_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/libs" CACHE PATH "Directory containing the MNN library")

add_executable(yolo_mnn "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/inference.cc")

# Shared header-only helpers (palette, names, annotator, types, post-processing) under examples/cpp/common
target_include_directories(yolo_mnn PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../common
    ${MNN_INCLUDE_DIR}
)

target_link_directories(yolo_mnn PRIVATE ${MNN_LIB_DIR})

target_link_libraries(yolo_mnn ${OpenCV_LIBS} MNN)
