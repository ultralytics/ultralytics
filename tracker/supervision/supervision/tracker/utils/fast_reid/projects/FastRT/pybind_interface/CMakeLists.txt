SET(APP_PROJECT_NAME ReID)

# pybind
find_package(pybind11)

find_package(CUDA REQUIRED)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(${SOLUTION_DIR}/include)

pybind11_add_module(${APP_PROJECT_NAME} ${PROJECT_SOURCE_DIR}/pybind_interface/reid.cpp)

# OpenCV
find_package(OpenCV)
target_include_directories(${APP_PROJECT_NAME}
PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(${APP_PROJECT_NAME}
PUBLIC
  ${OpenCV_LIBS}
)

if(BUILD_FASTRT_ENGINE AND BUILD_PYTHON_INTERFACE)
  SET(FASTRTENGINE_LIB FastRTEngine)
else()
  SET(FASTRTENGINE_LIB ${SOLUTION_DIR}/libs/FastRTEngine/libFastRTEngine.so)
endif()

target_link_libraries(${APP_PROJECT_NAME} 
PRIVATE
  ${FASTRTENGINE_LIB}
  nvinfer
)