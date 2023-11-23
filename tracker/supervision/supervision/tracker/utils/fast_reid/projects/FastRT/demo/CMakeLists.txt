SET(APP_PROJECT_NAME fastrt)

find_package(CUDA REQUIRED)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

include_directories(${SOLUTION_DIR}/include)
add_executable(${APP_PROJECT_NAME} inference.cpp)

# numpy
if(USE_CNUMPY)
  include_directories(${SOLUTION_DIR}/libs/cnpy/include)
  SET(CNPY_LIB ${SOLUTION_DIR}/libs/cnpy/lib/libcnpy.so)
else()
  SET(CNPY_LIB)
endif()

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

if(BUILD_FASTRT_ENGINE AND BUILD_DEMO)
  SET(FASTRTENGINE_LIB FastRTEngine)
else()
  SET(FASTRTENGINE_LIB ${SOLUTION_DIR}/libs/FastRTEngine/libFastRTEngine.so)
endif()

target_link_libraries(${APP_PROJECT_NAME} 
PRIVATE
  ${FASTRTENGINE_LIB}
  nvinfer
  ${CNPY_LIB}
)