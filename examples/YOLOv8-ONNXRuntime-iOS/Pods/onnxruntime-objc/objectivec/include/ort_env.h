// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_enums.h"

NS_ASSUME_NONNULL_BEGIN

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gets the ORT version string in format major.minor.patch.
 *
 * Available since 1.15.
 */
NSString* _Nullable ORTVersion(void);

#ifdef __cplusplus
}
#endif

/**
 * The ORT environment.
 * It maintains shared state including the default logger.
 *
 * @note One ORTEnv should be created before and destroyed after other ORT API usage.
 */
@interface ORTEnv : NSObject

- (instancetype)init NS_UNAVAILABLE;

/**
 * Creates an ORT Environment.
 *
 * @param loggingLevel The environment logging level.
 * @param error Optional error information set if an error occurs.
 * @return The instance, or nil if an error occurs.
 */
- (nullable instancetype)initWithLoggingLevel:(ORTLoggingLevel)loggingLevel
                                        error:(NSError**)error NS_DESIGNATED_INITIALIZER;

@end

NS_ASSUME_NONNULL_END
