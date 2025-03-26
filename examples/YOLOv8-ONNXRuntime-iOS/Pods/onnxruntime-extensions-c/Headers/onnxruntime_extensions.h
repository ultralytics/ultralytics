// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

// Note: The following include path is used for building Swift Package Manager support for ORT Extensions.
// The macro is defined in cxxSettings config in Package.swift.
// The reason why we need a prefix is that when Xcode includes the package it copies it to an internally generated path with 
// the package name as a prefix. 
// And we don't have control over the include paths when that happens in the user's iOS app. 
// The only way we found to make the include path work automatically for now.
#ifdef ORT_SWIFT_PACKAGE_MANAGER_BUILD
#include "onnxruntime/onnxruntime_c_api.h"
#else
#include "onnxruntime_c_api.h"
#endif


#ifdef __cplusplus
extern "C" {
#endif

OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api);

int ORT_API_CALL GetActiveOrtAPIVersion();

#ifdef __cplusplus
}
#endif
