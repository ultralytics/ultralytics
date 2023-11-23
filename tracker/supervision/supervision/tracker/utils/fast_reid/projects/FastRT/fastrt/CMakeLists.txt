project(FastRTEngine)

file(GLOB_RECURSE COMMON_SRC_FILES
  ${CMAKE_CURRENT_SOURCE_DIR}/common/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/common/calibrator.cpp
)

find_package(CUDA REQUIRED)
# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)
# tensorrt
include_directories(/usr/include/x86_64-linux-gnu/)
link_directories(/usr/lib/x86_64-linux-gnu/)

# build engine as library
add_library(${PROJECT_NAME} ${TARGET} ${COMMON_SRC_FILES})

target_include_directories(${PROJECT_NAME}
PUBLIC
  ../include
)

find_package(OpenCV)
target_include_directories(${PROJECT_NAME}
PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME} 
  nvinfer
  cudart
  ${OpenCV_LIBS}
)

SET_TARGET_PROPERTIES(${PROJECT_NAME} 
PROPERTIES
  SOVERSION ${LIBARARY_SOVERSION}
  VERSION ${LIBARARY_VERSION}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

install(TARGETS ${PROJECT_NAME}
  LIBRARY DESTINATION ${SOLUTION_DIR}/libs/${PROJECT_NAME})

add_subdirectory(layers)
add_subdirectory(engine)
add_subdirectory(heads)
add_subdirectory(backbones)
add_subdirectory(meta_arch)
add_subdirectory(factory)