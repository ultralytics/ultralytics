#ifndef ISL_DEPRECATED_SET_INT_H
#define ISL_DEPRECATED_SET_INT_H

#include <isl/deprecated/int.h>
#include <isl/set_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_basic_set *isl_basic_set_fix(__isl_take isl_basic_set *bset,
		enum isl_dim_type type, unsigned pos, isl_int value);
__isl_give isl_set *isl_set_lower_bound(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, isl_int value);
__isl_give isl_set *isl_set_upper_bound(__isl_take isl_set *set,
	enum isl_dim_type type, unsigned pos, isl_int value);
__isl_give isl_set *isl_set_fix(__isl_take isl_set *set,
		enum isl_dim_type type, unsigned pos, isl_int value);

int isl_set_plain_is_fixed(__isl_keep isl_set *set,
	enum isl_dim_type type, unsigned pos, isl_int *val);

#if defined(__cplusplus)
}
#endif

#endif
