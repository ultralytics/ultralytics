// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_value.h"

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

@interface ORTValue ()

/**
 * Creates a value from an existing C++ API Ort::Value and takes ownership from it.
 * Note: Ownership is guaranteed to be transferred on success but not otherwise.
 *
 * @param existingCXXAPIOrtValue The existing C++ API Ort::Value.
 * @param externalTensorData Any external tensor data referenced by `existingCXXAPIOrtValue`.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithCXXAPIOrtValue:(Ort::Value&&)existingCXXAPIOrtValue
                             externalTensorData:(nullable NSMutableData*)externalTensorData
                                          error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (Ort::Value&)CXXAPIOrtValue;

@end

NS_ASSUME_NONNULL_END
