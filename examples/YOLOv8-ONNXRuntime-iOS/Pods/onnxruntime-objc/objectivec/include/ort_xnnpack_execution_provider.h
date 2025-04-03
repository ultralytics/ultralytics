// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#import "ort_session.h"

NS_ASSUME_NONNULL_BEGIN

/**
 * Options for configuring the Xnnpack execution provider.
 */
@interface ORTXnnpackExecutionProviderOptions : NSObject

/**
 * How many threads used for the Xnnpack execution provider.
 */
@property int intra_op_num_threads;

@end

@interface ORTSessionOptions (ORTSessionOptionsXnnpackEP)

/**
 * Available since 1.14.
 * Enables the Xnnpack execution provider in the session configuration options.
 * It is appended to the execution provider list which is ordered by
 * decreasing priority.
 *
 * @param options The Xnnpack execution provider configuration options.
 * @param error Optional error information set if an error occurs.
 * @return Whether the provider was enabled successfully.
 */
- (BOOL)appendXnnpackExecutionProviderWithOptions:(ORTXnnpackExecutionProviderOptions*)options
                                            error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
