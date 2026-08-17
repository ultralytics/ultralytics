# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
add_executable(yolo-cli ${CMAKE_CURRENT_SOURCE_DIR}/../../src/cli.cpp)
target_link_libraries(yolo-cli PRIVATE yologgml)
