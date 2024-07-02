// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_enums.h"

#import "cxx_api.h"

OrtLoggingLevel PublicToCAPILoggingLevel(ORTLoggingLevel logging_level);

ORTValueType CAPIToPublicValueType(ONNXType capi_type);

ONNXTensorElementDataType PublicToCAPITensorElementType(ORTTensorElementDataType type);
ORTTensorElementDataType CAPIToPublicTensorElementType(ONNXTensorElementDataType capi_type);

size_t SizeOfCAPITensorElementType(ONNXTensorElementDataType capi_type);

GraphOptimizationLevel PublicToCAPIGraphOptimizationLevel(ORTGraphOptimizationLevel opt_level);
