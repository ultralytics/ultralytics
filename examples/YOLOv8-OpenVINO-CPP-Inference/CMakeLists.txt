cmake_minimum_required(VERSION 3.12)
project(yolo11_openvino_example)

find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(detect 
    main.cc
    inference.cc
)

target_link_libraries(detect 
    PRIVATE 
    openvino::runtime
    ${OpenCV_LIBS}
)
