// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_xnnpack_execution_provider.h"

#import "cxx_api.h"
#import "error_utils.h"
#import "ort_session_internal.h"

NS_ASSUME_NONNULL_BEGIN

@implementation ORTXnnpackExecutionProviderOptions

@end

@implementation ORTSessionOptions (ORTSessionOptionsXnnpackEP)

- (BOOL)appendXnnpackExecutionProviderWithOptions:(ORTXnnpackExecutionProviderOptions*)options
                                            error:(NSError**)error {
  try {
    NSDictionary* provider_options = @{
      @"intra_op_num_threads" : [NSString stringWithFormat:@"%d", options.intra_op_num_threads]
    };
    return [self appendExecutionProvider:@"XNNPACK" providerOptions:provider_options error:error];
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error);
}

@end

NS_ASSUME_NONNULL_END
