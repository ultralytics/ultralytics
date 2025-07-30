#ifndef ISL_MAP_TO_BASIC_SET_H
#define ISL_MAP_TO_BASIC_SET_H

#include <isl/set_type.h>
#include <isl/map_type.h>
#include <isl/maybe_basic_set.h>

#define ISL_KEY		isl_map
#define ISL_VAL		isl_basic_set
#define ISL_HMAP_SUFFIX	map_to_basic_set
#define ISL_HMAP	isl_map_to_basic_set
#include <isl/hmap.h>
#undef ISL_KEY
#undef ISL_VAL
#undef ISL_HMAP_SUFFIX
#undef ISL_HMAP

#endif
