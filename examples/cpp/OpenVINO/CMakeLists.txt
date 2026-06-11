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

target_link_libraries(yolo_openvino
	${OpenCV_LIBS}
	openvino::runtime
)
