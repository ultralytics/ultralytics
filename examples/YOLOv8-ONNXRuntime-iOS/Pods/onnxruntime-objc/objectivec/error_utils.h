// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include <exception>

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN

void ORTSaveCodeAndDescriptionToError(int code, const char* description, NSError** error);
void ORTSaveCodeAndDescriptionToError(int code, NSString* description, NSError** error);
void ORTSaveOrtExceptionToError(const Ort::Exception& e, NSError** error);
void ORTSaveExceptionToError(const std::exception& e, NSError** error);

// helper macros to catch and handle C++ exceptions
#define ORT_OBJC_API_IMPL_CATCH(error, failure_return_value) \
  catch (const Ort::Exception& e) {                          \
    ORTSaveOrtExceptionToError(e, (error));                  \
    return (failure_return_value);                           \
  }                                                          \
  catch (const std::exception& e) {                          \
    ORTSaveExceptionToError(e, (error));                     \
    return (failure_return_value);                           \
  }

#define ORT_OBJC_API_IMPL_CATCH_RETURNING_BOOL(error) \
  ORT_OBJC_API_IMPL_CATCH(error, NO)

#define ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error) \
  ORT_OBJC_API_IMPL_CATCH(error, nil)

NS_ASSUME_NONNULL_END
