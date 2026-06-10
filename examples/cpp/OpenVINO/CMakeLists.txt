cmake_minimum_required(VERSION 3.12)
project(yolo_openvino)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)

include_directories(
	${OpenCV_INCLUDE_DIRS}
	/path/to/intel/openvino/runtime/include
)

add_executable(yolo_openvino
	main.cc
	inference.cc
)

target_link_libraries(yolo_openvino
	${OpenCV_LIBS}
	/path/to/intel/openvino/runtime/lib/intel64/libopenvino.so
)
