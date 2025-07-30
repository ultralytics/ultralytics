#ifndef ISL_AFF_TYPE_H
#define ISL_AFF_TYPE_H

#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_subclass(isl_multi_aff) __isl_subclass(isl_pw_aff) isl_aff;
typedef struct isl_aff isl_aff;

ISL_DECLARE_LIST(aff)

struct __isl_subclass(isl_multi_pw_aff) __isl_subclass(isl_pw_multi_aff)
	__isl_subclass(isl_union_pw_aff) isl_pw_aff;
typedef struct isl_pw_aff isl_pw_aff;

ISL_DECLARE_LIST(pw_aff)

struct __isl_subclass(isl_multi_union_pw_aff)
	__isl_subclass(isl_union_pw_multi_aff) isl_union_pw_aff;
typedef struct isl_union_pw_aff isl_union_pw_aff;

ISL_DECLARE_LIST_TYPE(union_pw_aff)

struct __isl_subclass(isl_multi_pw_aff) __isl_subclass(isl_pw_multi_aff)
	isl_multi_aff;
typedef struct isl_multi_aff isl_multi_aff;

struct __isl_subclass(isl_multi_pw_aff) __isl_subclass(isl_union_pw_multi_aff)
	isl_pw_multi_aff;
typedef struct isl_pw_multi_aff isl_pw_multi_aff;

struct __isl_export isl_union_pw_multi_aff;
typedef struct isl_union_pw_multi_aff isl_union_pw_multi_aff;

ISL_DECLARE_LIST_TYPE(union_pw_multi_aff)

struct __isl_subclass(isl_multi_union_pw_aff) isl_multi_pw_aff;
typedef struct isl_multi_pw_aff isl_multi_pw_aff;

struct __isl_export isl_multi_union_pw_aff;
typedef struct isl_multi_union_pw_aff isl_multi_union_pw_aff;

#if defined(__cplusplus)
}
#endif

#endif
