// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_session.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Gets whether the CoreML execution provider is available.
 */
BOOL ORTIsCoreMLExecutionProviderAvailable(void);

#ifdef __cplusplus
}
#endif

NS_ASSUME_NONNULL_BEGIN

/**
 * Options for configuring the CoreML execution provider.
 */
@interface ORTCoreMLExecutionProviderOptions : NSObject

/**
 * Whether the CoreML execution provider should run on CPU only.
 */
@property BOOL useCPUOnly;

/**
 * Whether the CoreML execution provider is enabled on subgraphs.
 */
@property BOOL enableOnSubgraphs;

/**
 * Whether the CoreML execution provider is only enabled for devices with Apple
 * Neural Engine (ANE).
 */
@property BOOL onlyEnableForDevicesWithANE;

@end

@interface ORTSessionOptions (ORTSessionOptionsCoreMLEP)

/**
 * Enables the CoreML execution provider in the session configuration options.
 * It is appended to the execution provider list which is ordered by
 * decreasing priority.
 *
 * @param options The CoreML execution provider configuration options.
 * @param error Optional error information set if an error occurs.
 * @return Whether the provider was enabled successfully.
 */
- (BOOL)appendCoreMLExecutionProviderWithOptions:(ORTCoreMLExecutionProviderOptions*)options
                                           error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
