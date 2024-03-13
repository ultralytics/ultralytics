// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_coreml_execution_provider.h"

#import "cxx_api.h"
#import "error_utils.h"
#import "ort_session_internal.h"

NS_ASSUME_NONNULL_BEGIN

BOOL ORTIsCoreMLExecutionProviderAvailable() {
  return ORT_OBJC_API_COREML_EP_AVAILABLE ? YES : NO;
}

@implementation ORTCoreMLExecutionProviderOptions

@end

@implementation ORTSessionOptions (ORTSessionOptionsCoreMLEP)

- (BOOL)appendCoreMLExecutionProviderWithOptions:(ORTCoreMLExecutionProviderOptions*)options
                                           error:(NSError**)error {
#if ORT_OBJC_API_COREML_EP_AVAILABLE
  try {
    const uint32_t flags =
        (options.useCPUOnly ? COREML_FLAG_USE_CPU_ONLY : 0) |
        (options.enableOnSubgraphs ? COREML_FLAG_ENABLE_ON_SUBGRAPH : 0) |
        (options.onlyEnableForDevicesWithANE ? COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE : 0);
    Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(
        [self CXXAPIOrtSessionOptions], flags));
    return YES;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error);
#else  // !ORT_OBJC_API_COREML_EP_AVAILABLE
  static_cast<void>(options);
  ORTSaveCodeAndDescriptionToError(ORT_FAIL, "CoreML execution provider is not enabled.", error);
  return NO;
#endif
}

@end

NS_ASSUME_NONNULL_END
