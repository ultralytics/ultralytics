#include <isl/id_to_pw_aff.h>
#include <isl/aff.h>

#define isl_id_is_equal(id1,id2)	id1 == id2

#define ISL_KEY		isl_id
#define ISL_VAL		isl_pw_aff
#define ISL_HMAP_SUFFIX	id_to_pw_aff
#define ISL_HMAP	isl_id_to_pw_aff
#define ISL_KEY_IS_EQUAL	isl_id_is_equal
#define ISL_VAL_IS_EQUAL	isl_pw_aff_plain_is_equal
#define ISL_KEY_PRINT		isl_printer_print_id
#define ISL_VAL_PRINT		isl_printer_print_pw_aff

#include <isl/hmap_templ.c>
