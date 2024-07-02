// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// wrapper for ORT C/C++ API headers

#if defined(__clang__)
#pragma clang diagnostic push
// ignore clang documentation-related warnings
// instead, we will rely on Doxygen warnings for the C/C++ API headers
#pragma clang diagnostic ignored "-Wdocumentation"
#endif  // defined(__clang__)

// paths are different when building the Swift Package Manager package as the headers come from the iOS pod archive
// clang-format off
#define STRINGIFY(x) #x
#ifdef SPM_BUILD
#define ORT_C_CXX_HEADER_FILE_PATH(x) STRINGIFY(onnxruntime/x)
#else
#define ORT_C_CXX_HEADER_FILE_PATH(x) STRINGIFY(x)
#endif
// clang-format on

#if __has_include(ORT_C_CXX_HEADER_FILE_PATH(onnxruntime_training_c_api.h))
#include ORT_C_CXX_HEADER_FILE_PATH(onnxruntime_training_c_api.h)
#include ORT_C_CXX_HEADER_FILE_PATH(onnxruntime_training_cxx_api.h)
#else
#include ORT_C_CXX_HEADER_FILE_PATH(onnxruntime_c_api.h)
#include ORT_C_CXX_HEADER_FILE_PATH(onnxruntime_cxx_api.h)
#endif

#if __has_include(ORT_C_CXX_HEADER_FILE_PATH(coreml_provider_factory.h))
#define ORT_OBJC_API_COREML_EP_AVAILABLE 1
#include ORT_C_CXX_HEADER_FILE_PATH(coreml_provider_factory.h)
#else
#define ORT_OBJC_API_COREML_EP_AVAILABLE 0
#endif

#if defined(__clang__)
#pragma clang diagnostic pop
#endif  // defined(__clang__)
