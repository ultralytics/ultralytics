// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "error_utils.h"

NS_ASSUME_NONNULL_BEGIN

static NSString* const kOrtErrorDomain = @"onnxruntime";

void ORTSaveCodeAndDescriptionToError(int code, const char* descriptionCstr, NSError** error) {
  if (!error) return;

  NSString* description = [NSString stringWithCString:descriptionCstr
                                             encoding:NSASCIIStringEncoding];

  *error = [NSError errorWithDomain:kOrtErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

void ORTSaveCodeAndDescriptionToError(int code, NSString* description, NSError** error) {
  if (!error) return;

  *error = [NSError errorWithDomain:kOrtErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

void ORTSaveOrtExceptionToError(const Ort::Exception& e, NSError** error) {
  ORTSaveCodeAndDescriptionToError(e.GetOrtErrorCode(), e.what(), error);
}

void ORTSaveExceptionToError(const std::exception& e, NSError** error) {
  ORTSaveCodeAndDescriptionToError(ORT_RUNTIME_EXCEPTION, e.what(), error);
}

NS_ASSUME_NONNULL_END
