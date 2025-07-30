#ifndef ISL_DEPRECATED_MAP_INT_H
#define ISL_DEPRECATED_MAP_INT_H

#include <isl/deprecated/int.h>
#include <isl/map_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

int isl_basic_map_plain_is_fixed(__isl_keep isl_basic_map *bmap,
	enum isl_dim_type type, unsigned pos, isl_int *val);

__isl_give isl_map *isl_map_fix(__isl_take isl_map *map,
	enum isl_dim_type type, unsigned pos, isl_int value);
int isl_map_plain_is_fixed(__isl_keep isl_map *map,
	enum isl_dim_type type, unsigned pos, isl_int *val);

__isl_give isl_map *isl_map_fixed_power(__isl_take isl_map *map, isl_int exp);

#if defined(__cplusplus)
}
#endif

#endif
