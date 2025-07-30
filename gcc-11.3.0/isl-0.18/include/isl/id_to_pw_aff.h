#ifndef ISL_ID_TO_PW_AFF_H
#define ISL_ID_TO_PW_AFF_H

#include <isl/id.h>
#include <isl/aff_type.h>
#include <isl/maybe_pw_aff.h>

#define ISL_KEY		isl_id
#define ISL_VAL		isl_pw_aff
#define ISL_HMAP_SUFFIX	id_to_pw_aff
#define ISL_HMAP	isl_id_to_pw_aff
#include <isl/hmap.h>
#undef ISL_KEY
#undef ISL_VAL
#undef ISL_HMAP_SUFFIX
#undef ISL_HMAP

#endif
