// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "ort_value_internal.h"

#include <optional>

#import "cxx_api.h"
#import "error_utils.h"
#import "ort_enums_internal.h"

NS_ASSUME_NONNULL_BEGIN

namespace {

ORTTensorTypeAndShapeInfo* CXXAPIToPublicTensorTypeAndShapeInfo(
    const Ort::ConstTensorTypeAndShapeInfo& CXXAPITensorTypeAndShapeInfo) {
  auto* result = [[ORTTensorTypeAndShapeInfo alloc] init];
  const auto elementType = CXXAPITensorTypeAndShapeInfo.GetElementType();
  const std::vector<int64_t> shape = CXXAPITensorTypeAndShapeInfo.GetShape();

  result.elementType = CAPIToPublicTensorElementType(elementType);
  auto* shapeArray = [[NSMutableArray alloc] initWithCapacity:shape.size()];
  for (size_t i = 0; i < shape.size(); ++i) {
    shapeArray[i] = @(shape[i]);
  }
  result.shape = shapeArray;

  return result;
}

ORTValueTypeInfo* CXXAPIToPublicValueTypeInfo(
    const Ort::TypeInfo& CXXAPITypeInfo) {
  auto* result = [[ORTValueTypeInfo alloc] init];
  const auto valueType = CXXAPITypeInfo.GetONNXType();

  result.type = CAPIToPublicValueType(valueType);

  if (valueType == ONNX_TYPE_TENSOR) {
    const auto tensorTypeAndShapeInfo = CXXAPITypeInfo.GetTensorTypeAndShapeInfo();
    result.tensorTypeAndShapeInfo = CXXAPIToPublicTensorTypeAndShapeInfo(tensorTypeAndShapeInfo);
  }

  return result;
}

// out = a * b
// returns true iff the result does not overflow
bool SafeMultiply(size_t a, size_t b, size_t& out) {
  return !__builtin_mul_overflow(a, b, &out);
}

}  // namespace

@interface ORTValue ()

// pointer to any external tensor data to keep alive for the lifetime of the ORTValue
@property(nonatomic, nullable) NSMutableData* externalTensorData;

@end

@implementation ORTValue {
  std::optional<Ort::Value> _value;
  std::optional<Ort::TypeInfo> _typeInfo;
}

#pragma mark - Public

- (nullable instancetype)initWithTensorData:(NSMutableData*)tensorData
                                elementType:(ORTTensorElementDataType)elementType
                                      shape:(NSArray<NSNumber*>*)shape
                                      error:(NSError**)error {
  try {
    if (elementType == ORTTensorElementDataTypeString) {
      ORT_CXX_API_THROW(
          "ORTTensorElementDataTypeString element type provided. "
          "Please call initWithTensorStringData:shape:error: instead to create an ORTValue with string data.",
          ORT_INVALID_ARGUMENT);
    }
    const auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    const auto ONNXElementType = PublicToCAPITensorElementType(elementType);
    const auto shapeVector = [shape]() {
      std::vector<int64_t> result{};
      result.reserve(shape.count);
      for (NSNumber* dim in shape) {
        result.push_back(dim.longLongValue);
      }
      return result;
    }();
    Ort::Value ortValue = Ort::Value::CreateTensor(
        memoryInfo, tensorData.mutableBytes, tensorData.length,
        shapeVector.data(), shapeVector.size(), ONNXElementType);

    return [self initWithCXXAPIOrtValue:std::move(ortValue)
                     externalTensorData:tensorData
                                  error:error];
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable instancetype)initWithTensorStringData:(NSArray<NSString*>*)tensorStringData
                                            shape:(NSArray<NSNumber*>*)shape
                                            error:(NSError**)error {
  try {
    Ort::AllocatorWithDefaultOptions allocator;
    size_t tensorSize = 1U;
    const auto shapeVector = [&tensorSize, shape]() {
      std::vector<int64_t> result{};
      result.reserve(shape.count);
      for (NSNumber* dim in shape) {
        const auto dimValue = dim.longLongValue;
        if (dimValue < 0 || !SafeMultiply(static_cast<size_t>(dimValue), tensorSize, tensorSize)) {
          ORT_CXX_API_THROW("Failed to compute the tensor size.", ORT_RUNTIME_EXCEPTION);
        }
        result.push_back(dimValue);
      }
      return result;
    }();

    if (tensorSize != [tensorStringData count]) {
      ORT_CXX_API_THROW(
          "Computed tensor size does not equal the length of the provided tensor string data.",
          ORT_INVALID_ARGUMENT);
    }

    Ort::Value ortValue = Ort::Value::CreateTensor(
        allocator, shapeVector.data(), shapeVector.size(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);

    size_t index = 0;
    for (NSString* stringData in tensorStringData) {
      ortValue.FillStringTensorElement([stringData UTF8String], index++);
    }

    return [self initWithCXXAPIOrtValue:std::move(ortValue)
                     externalTensorData:nil
                                  error:error];
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable ORTValueTypeInfo*)typeInfoWithError:(NSError**)error {
  try {
    return CXXAPIToPublicValueTypeInfo(*_typeInfo);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable ORTTensorTypeAndShapeInfo*)tensorTypeAndShapeInfoWithError:(NSError**)error {
  try {
    const auto tensorTypeAndShapeInfo = _typeInfo->GetTensorTypeAndShapeInfo();
    return CXXAPIToPublicTensorTypeAndShapeInfo(tensorTypeAndShapeInfo);
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSMutableData*)tensorDataWithError:(NSError**)error {
  try {
    const auto tensorTypeAndShapeInfo = _typeInfo->GetTensorTypeAndShapeInfo();
    if (tensorTypeAndShapeInfo.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
      ORT_CXX_API_THROW(
          "This ORTValue holds string data. Please call tensorStringDataWithError: "
          "instead to retrieve the string data from this ORTValue.",
          ORT_RUNTIME_EXCEPTION);
    }
    const size_t elementCount = tensorTypeAndShapeInfo.GetElementCount();
    const size_t elementSize = SizeOfCAPITensorElementType(tensorTypeAndShapeInfo.GetElementType());
    size_t rawDataLength;
    if (!SafeMultiply(elementCount, elementSize, rawDataLength)) {
      ORT_CXX_API_THROW("failed to compute tensor data length", ORT_RUNTIME_EXCEPTION);
    }

    void* rawData;
    Ort::ThrowOnError(Ort::GetApi().GetTensorMutableData(*_value, &rawData));

    return [NSMutableData dataWithBytesNoCopy:rawData
                                       length:rawDataLength
                                 freeWhenDone:NO];
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

- (nullable NSArray<NSString*>*)tensorStringDataWithError:(NSError**)error {
  try {
    const auto tensorTypeAndShapeInfo = _typeInfo->GetTensorTypeAndShapeInfo();
    const size_t elementCount = tensorTypeAndShapeInfo.GetElementCount();
    const size_t tensorStringDataLength = _value->GetStringTensorDataLength();
    std::vector<char> tensorStringData(tensorStringDataLength, '\0');
    std::vector<size_t> offsets(elementCount);
    _value->GetStringTensorContent(tensorStringData.data(), tensorStringDataLength,
                                   offsets.data(), offsets.size());

    NSMutableArray<NSString*>* result = [NSMutableArray arrayWithCapacity:elementCount];
    for (size_t idx = 0; idx < elementCount; ++idx) {
      const size_t strLength = (idx == elementCount - 1) ? tensorStringDataLength - offsets[idx]
                                                         : offsets[idx + 1] - offsets[idx];
      [result addObject:[[NSString alloc] initWithBytes:tensorStringData.data() + offsets[idx]
                                                 length:strLength
                                               encoding:NSUTF8StringEncoding]];
    }
    return result;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error)
}

#pragma mark - Internal

- (nullable instancetype)initWithCXXAPIOrtValue:(Ort::Value&&)existingCXXAPIOrtValue
                             externalTensorData:(nullable NSMutableData*)externalTensorData
                                          error:(NSError**)error {
  if ((self = [super init]) == nil) {
    return nil;
  }

  try {
    _typeInfo = existingCXXAPIOrtValue.GetTypeInfo();
    _externalTensorData = externalTensorData;

    // transfer C++ Ort::Value ownership to this instance
    _value = std::move(existingCXXAPIOrtValue);
    return self;
  }
  ORT_OBJC_API_IMPL_CATCH_RETURNING_NULLABLE(error);
}

- (Ort::Value&)CXXAPIOrtValue {
  return *_value;
}

@end

@implementation ORTValueTypeInfo
@end

@implementation ORTTensorTypeAndShapeInfo
@end

NS_ASSUME_NONNULL_END
