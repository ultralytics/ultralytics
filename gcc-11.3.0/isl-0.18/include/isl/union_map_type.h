#ifndef ISL_UNION_MAP_TYPE_H
#define ISL_UNION_MAP_TYPE_H

#include <isl/ctx.h>
#include <isl/list.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_export isl_union_map;
typedef struct isl_union_map isl_union_map;
ISL_DECLARE_LIST_TYPE(union_map)
#ifndef isl_union_set
struct __isl_export isl_union_set;
typedef struct isl_union_set isl_union_set;
ISL_DECLARE_LIST_TYPE(union_set)
#endif

#if defined(__cplusplus)
}
#endif

#endif
