// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "cxx_utils.h"

#include <vector>
#include <optional>
#include <string>

#import "error_utils.h"

#import "ort_value_internal.h"

NS_ASSUME_NONNULL_BEGIN

namespace utils {

NSString* toNSString(const std::string& str) {
  NSString* nsStr = [NSString stringWithUTF8String:str.c_str()];
  if (!nsStr) {
    ORT_CXX_API_THROW("Failed to convert std::string to NSString", ORT_INVALID_ARGUMENT);
  }

  return nsStr;
}

NSString* _Nullable toNullableNSString(const std::optional<std::string>& str) {
  if (str.has_value()) {
    return toNSString(*str);
  }
  return nil;
}

std::string toStdString(NSString* str) {
  return std::string([str UTF8String]);
}

std::optional<std::string> toStdOptionalString(NSString* _Nullable str) {
  if (str) {
    return std::optional<std::string>([str UTF8String]);
  }
  return std::nullopt;
}

std::vector<std::string> toStdStringVector(NSArray<NSString*>* strs) {
  std::vector<std::string> result;
  result.reserve(strs.count);
  for (NSString* str in strs) {
    result.push_back([str UTF8String]);
  }
  return result;
}

NSArray<NSString*>* toNSStringNSArray(const std::vector<std::string>& strs) {
  NSMutableArray<NSString*>* result = [NSMutableArray arrayWithCapacity:strs.size()];
  for (const std::string& str : strs) {
    [result addObject:toNSString(str)];
  }
  return result;
}

NSArray<ORTValue*>* _Nullable wrapUnownedCAPIOrtValues(const std::vector<OrtValue*>& CAPIValues, NSError** error) {
  NSMutableArray<ORTValue*>* result = [NSMutableArray arrayWithCapacity:CAPIValues.size()];
  for (size_t i = 0; i < CAPIValues.size(); ++i) {
    // Wrap the C OrtValue in a C++ Ort::Value to automatically handle its release.
    // Then, transfer that C++ Ort::Value to a new ORTValue.
    Ort::Value CXXAPIValue{CAPIValues[i]};
    ORTValue* val = [[ORTValue alloc] initWithCXXAPIOrtValue:std::move(CXXAPIValue)
                                          externalTensorData:nil
                                                       error:error];
    if (!val) {
      // clean up remaining C OrtValues which haven't been wrapped by a C++ Ort::Value yet
      for (size_t j = i + 1; j < CAPIValues.size(); ++j) {
        Ort::GetApi().ReleaseValue(CAPIValues[j]);
      }
      return nil;
    }
    [result addObject:val];
  }
  return result;
}

std::vector<const OrtValue*> getWrappedCAPIOrtValues(NSArray<ORTValue*>* values) {
  std::vector<const OrtValue*> result;
  result.reserve(values.count);
  for (ORTValue* val in values) {
    result.push_back(static_cast<const OrtValue*>([val CXXAPIOrtValue]));
  }
  return result;
}

}  // namespace utils

NS_ASSUME_NONNULL_END
