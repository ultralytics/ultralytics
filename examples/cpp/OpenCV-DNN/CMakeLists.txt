cmake_minimum_required(VERSION 3.5)

project(yolo_opencv_dnn VERSION 0.1)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# !OpenCV

set(PROJECT_SOURCES
    main.cpp

    inference.h
    inference.cpp
)

add_executable(yolo_opencv_dnn ${PROJECT_SOURCES})

# Shared header-only helpers (types, post-processing, annotator, COCO names) under examples/cpp/common
target_include_directories(yolo_opencv_dnn PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(yolo_opencv_dnn ${OpenCV_LIBS})
