#ifndef ISL_DEPRECATED_UNION_MAP_INT_H
#define ISL_DEPRECATED_UNION_MAP_INT_H

#include <isl/deprecated/int.h>
#include <isl/union_map_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_union_map *isl_union_map_fixed_power(
	__isl_take isl_union_map *umap, isl_int exp);

#if defined(__cplusplus)
}
#endif

#endif
