// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_env_internal.h"

#include <optional>

#import "cxx_api.h"

#import "error_utils.h"
#import "ort_enums_internal.h"

NS_ASSUME_NONNULL_BEGIN

NSString* _Nullable ORTVersion(void) {
  return [NSString stringWithUTF8String:OrtGetApiBase()->GetVersionString()];
}

@implementation ORTEnv {
  std::optional<Ort::Env> _env;
}

- (nullable instancetype)initWithLoggingLevel:(ORTLoggingLevel)loggingLevel
                                        error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    const auto CAPILoggingLevel = PublicToCAPILoggingLevel(loggingLevel);
    _env = Ort::Env{CAPILoggingLevel};
    return self;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (Ort::Env&)CXXAPIOrtEnv {
  return *_env;
}

@end

NS_ASSUME_NONNULL_END
