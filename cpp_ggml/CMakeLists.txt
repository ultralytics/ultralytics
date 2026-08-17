cmake_minimum_required(VERSION 3.14)
project(yologgml LANGUAGES C CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Options
option(YOLO_GGML_BUILD_CLI   "Build the yolo-cli binary"           ON)
option(YOLO_GGML_SHARED      "Build libyologgml as a shared library" OFF)

# Backend forwarding to ggml (mirrors rf-detr.cpp pattern)
option(YOLO_GGML_CUDA    "Enable CUDA backend in ggml"    OFF)
option(YOLO_GGML_CUDNN   "Use cuDNN for direct CUDA convolutions" ON)
option(YOLO_GGML_METAL   "Enable Metal backend in ggml"   OFF)
option(YOLO_GGML_VULKAN  "Enable Vulkan backend in ggml"  OFF)
option(YOLO_GGML_HIPBLAS "Enable HIPBLAS backend in ggml" OFF)

set(_yolo_gpu_backends 0)
foreach(_backend YOLO_GGML_CUDA YOLO_GGML_METAL YOLO_GGML_VULKAN YOLO_GGML_HIPBLAS)
    if(${_backend})
        math(EXPR _yolo_gpu_backends "${_yolo_gpu_backends} + 1")
    endif()
endforeach()
if(_yolo_gpu_backends GREATER 1)
    message(FATAL_ERROR "Select at most one GPU backend per build directory")
endif()

# ggml uses CMake's `native` CUDA architecture only with CMake >=3.24. On
# older CMake, select the attached GPU instead of compiling a broad list that
# can include device intrinsics unavailable in the installed CUDA toolkit.
if(YOLO_GGML_CUDA AND NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND CMAKE_VERSION VERSION_LESS 3.24)
    find_program(NVIDIA_SMI nvidia-smi)
    if(NVIDIA_SMI)
        execute_process(
            COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE _yolo_cuda_compute_cap
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
        string(REGEX MATCH "^[0-9]+\\.[0-9]+" _yolo_cuda_compute_cap "${_yolo_cuda_compute_cap}")
        if(_yolo_cuda_compute_cap)
            string(REPLACE "." "" CMAKE_CUDA_ARCHITECTURES "${_yolo_cuda_compute_cap}")
            message(STATUS "CUDA architecture auto-detected: ${CMAKE_CUDA_ARCHITECTURES}")
        endif()
    endif()
endif()

if(YOLO_GGML_CUDA)
    set(GGML_CUDA        ON CACHE BOOL "" FORCE)
    set(GGML_CUDA_GRAPHS ON CACHE BOOL "Replay CUDA graphs" FORCE)
    set(GGML_CUDA_CUDNN  ${YOLO_GGML_CUDNN} CACHE BOOL "Use cuDNN direct convolutions" FORCE)
endif()
if(YOLO_GGML_METAL)
    set(GGML_METAL   ON CACHE BOOL "" FORCE)
endif()
if(YOLO_GGML_VULKAN)
    set(GGML_VULKAN  ON CACHE BOOL "" FORCE)
endif()
if(YOLO_GGML_HIPBLAS)
    set(GGML_HIP     ON CACHE BOOL "" FORCE)
endif()

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ggml.
#
# CPU optimization flags applied by default (override on cmake command line if
# you need a portable / non-AVX-512 / minimal build):
#
#   GGML_NATIVE=ON      -> compiles with `-march=native`, so the build adapts
#                          to the host (AVX2 on Zen2/Skylake, AVX-512 on Zen4 /
#                          Ice Lake / 9950X3D, etc.). Safe on any x86_64 host
#                          because it never sets explicit feature flags the
#                          local CPU doesn't have.
#   GGML_LLAMAFILE=ON   -> enables ggml's tinyBLAS path (an AVX-512 / AVX2
#                          tiled SGEMM for F32 x F32 mul_mat). Big win for the
#                          F32 conv hotspots via the im2col+mul_mat pipeline.
#                          tinyBLAS gracefully falls back to AVX2 / scalar
#                          if AVX-512 isn't available, so this is also safe on
#                          older hosts.
#   GGML_OPENMP=ON      -> ggml's default; relies on OpenMP for the thread pool
#                          which has lower wake-up overhead than the pthread
#                          fallback for this workload.
#
# Explicit GGML_AVX512* options are intentionally NOT set: -march=native already
# tells the compiler to emit AVX-512 intrinsics when the host supports them,
# and the explicit options would make the binary non-portable to non-AVX-512
# machines.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE      ON  CACHE BOOL "Enable tinyBLAS SGEMM kernels in ggml" FORCE)

# Apply local ggml patches before we descend into the submodule. The patches
# live in third_party/ggml-patches/ and are applied idempotently; re-running
# CMake is a no-op once they're in place. See scripts/apply_ggml_patches.sh
# for the logic; it can also be invoked standalone for debugging.
execute_process(
    COMMAND bash ${CMAKE_SOURCE_DIR}/scripts/apply_ggml_patches.sh
    RESULT_VARIABLE _ggml_patch_result
    OUTPUT_VARIABLE _ggml_patch_output
    ERROR_VARIABLE  _ggml_patch_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
)
if(NOT _ggml_patch_result EQUAL 0)
    message(FATAL_ERROR
        "Failed to apply ggml patches (exit ${_ggml_patch_result}):\n"
        "stdout:\n${_ggml_patch_output}\n"
        "stderr:\n${_ggml_patch_error}")
endif()
message(STATUS "${_ggml_patch_output}")

add_subdirectory(third_party/ggml EXCLUDE_FROM_ALL)

# stb (header-only interface target)
add_library(yolo_stb INTERFACE)
target_include_directories(yolo_stb INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/stb)

# libyologgml (cli.cpp is built as the yolo-cli executable below)
set(YOLOGGML_SOURCES
    src/common.cpp
    src/image_io.cpp
    src/gguf_loader.cpp
    src/backend.cpp
    src/yolo_graph.cpp
    src/postprocess.cpp
)

if(YOLO_GGML_SHARED)
    add_library(yologgml SHARED ${YOLOGGML_SOURCES})
else()
    add_library(yologgml STATIC ${YOLOGGML_SOURCES})
endif()

# Propagate the active GPU backend to src/backend.cpp so it knows which
# ggml_backend_*_init() to call. ggml itself is configured above via the
# GGML_* cache vars; these definitions just gate our wiring code.
if(YOLO_GGML_CUDA)
    target_compile_definitions(yologgml PRIVATE YOLO_USE_CUDA)
endif()
if(YOLO_GGML_METAL)
    target_compile_definitions(yologgml PRIVATE YOLO_USE_METAL)
endif()
if(YOLO_GGML_VULKAN)
    target_compile_definitions(yologgml PRIVATE YOLO_USE_VULKAN)
endif()
if(YOLO_GGML_HIPBLAS)
    # HIP uses the CUDA backend code path in ggml (ggml_backend_cuda_init
    # works for HIP builds). Expose the same define.
    target_compile_definitions(yologgml PRIVATE YOLO_USE_CUDA)
endif()

target_include_directories(yologgml
    PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/src
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(yologgml
    PUBLIC  ggml
    PRIVATE yolo_stb)

find_package(OpenMP QUIET)
if(OpenMP_CXX_FOUND)
    target_link_libraries(yologgml PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(yologgml PRIVATE YOLO_USE_OPENMP)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(yologgml PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()

# CLI
if(YOLO_GGML_BUILD_CLI AND EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/examples/cli/CMakeLists.txt)
    add_subdirectory(examples/cli)
endif()
