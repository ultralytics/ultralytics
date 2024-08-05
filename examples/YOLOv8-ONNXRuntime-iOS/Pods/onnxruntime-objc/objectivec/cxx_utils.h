// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import <Foundation/Foundation.h>

#include <optional>
#include <string>
#include <variant>

#import "cxx_api.h"

NS_ASSUME_NONNULL_BEGIN
@class ORTValue;

namespace utils {

NSString* toNSString(const std::string& str);
NSString* _Nullable toNullableNSString(const std::optional<std::string>& str);

std::string toStdString(NSString* str);
std::optional<std::string> toStdOptionalString(NSString* _Nullable str);

std::vector<std::string> toStdStringVector(NSArray<NSString*>* strs);
NSArray<NSString*>* toNSStringNSArray(const std::vector<std::string>& strs);

NSArray<ORTValue*>* _Nullable wrapUnownedCAPIOrtValues(const std::vector<OrtValue*>& values, NSError** error);

std::vector<const OrtValue*> getWrappedCAPIOrtValues(NSArray<ORTValue*>* values);

}  // namespace utils

NS_ASSUME_NONNULL_END
