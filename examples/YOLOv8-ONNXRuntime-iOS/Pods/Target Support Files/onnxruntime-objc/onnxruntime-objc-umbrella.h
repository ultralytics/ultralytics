#ifdef __OBJC__
#import <UIKit/UIKit.h>
#else
#ifndef FOUNDATION_EXPORT
#if defined(__cplusplus)
#define FOUNDATION_EXPORT extern "C"
#else
#define FOUNDATION_EXPORT extern
#endif
#endif
#endif

#import "ort_coreml_execution_provider.h"
#import "ort_env.h"
#import "ort_xnnpack_execution_provider.h"
#import "ort_session.h"
#import "ort_enums.h"
#import "ort_value.h"
#import "ort_custom_op_registration.h"
#import "onnxruntime.h"

FOUNDATION_EXPORT double onnxruntime_objcVersionNumber;
FOUNDATION_EXPORT const unsigned char onnxruntime_objcVersionString[];

