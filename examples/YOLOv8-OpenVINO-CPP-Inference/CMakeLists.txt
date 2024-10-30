cmake_minimum_required(VERSION 3.12)
project(yolov8_openvino_example)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	/path/to/intel/openvino/runtime/include
)

add_executable(detect 
	main.cc
	inference.cc
)

target_link_libraries(detect
	${OpenCV_LIBS}
	/path/to/intel/openvino/runtime/lib/intel64/libopenvino.so
)
