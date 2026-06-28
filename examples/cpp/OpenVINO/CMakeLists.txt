cmake_minimum_required(VERSION 3.12)
project(yolo_openvino)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(yolo_openvino
	main.cc
	inference.cc
)

# Shared header-only helpers (types, post-processing, annotator, COCO names) under examples/cpp/common
target_include_directories(yolo_openvino PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../common)

target_link_libraries(yolo_openvino
	${OpenCV_LIBS}
	openvino::runtime
)
