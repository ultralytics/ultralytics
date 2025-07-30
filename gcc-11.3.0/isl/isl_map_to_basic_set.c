#include <isl/map_to_basic_set.h>
#include <isl/map.h>
#include <isl/set.h>

#define ISL_KEY		isl_map
#define ISL_VAL		isl_basic_set
#define ISL_HMAP_SUFFIX	map_to_basic_set
#define ISL_HMAP	isl_map_to_basic_set
#define ISL_KEY_IS_EQUAL	isl_map_plain_is_equal
#define ISL_VAL_IS_EQUAL	isl_basic_set_plain_is_equal
#define ISL_KEY_PRINT		isl_printer_print_map
#define ISL_VAL_PRINT		isl_printer_print_basic_set

#include <isl/hmap_templ.c>
