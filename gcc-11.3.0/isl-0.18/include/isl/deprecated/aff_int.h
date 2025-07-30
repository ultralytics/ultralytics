#ifndef ISL_DEPRECATED_AFF_INT_H
#define ISL_DEPRECATED_AFF_INT_H

#include <isl/deprecated/int.h>
#include <isl/aff_type.h>

#if defined(__cplusplus)
extern "C" {
#endif

int isl_aff_get_constant(__isl_keep isl_aff *aff, isl_int *v);
int isl_aff_get_coefficient(__isl_keep isl_aff *aff,
	enum isl_dim_type type, int pos, isl_int *v);
int isl_aff_get_denominator(__isl_keep isl_aff *aff, isl_int *v);
__isl_give isl_aff *isl_aff_set_constant(__isl_take isl_aff *aff, isl_int v);
__isl_give isl_aff *isl_aff_set_coefficient(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, isl_int v);
__isl_give isl_aff *isl_aff_set_denominator(__isl_take isl_aff *aff, isl_int v);
__isl_give isl_aff *isl_aff_add_constant(__isl_take isl_aff *aff, isl_int v);
__isl_give isl_aff *isl_aff_add_constant_num(__isl_take isl_aff *aff,
	isl_int v);
__isl_give isl_aff *isl_aff_add_coefficient(__isl_take isl_aff *aff,
	enum isl_dim_type type, int pos, isl_int v);

__isl_give isl_aff *isl_aff_mod(__isl_take isl_aff *aff, isl_int mod);

__isl_give isl_aff *isl_aff_scale(__isl_take isl_aff *aff, isl_int f);
__isl_give isl_aff *isl_aff_scale_down(__isl_take isl_aff *aff, isl_int f);

__isl_give isl_pw_aff *isl_pw_aff_mod(__isl_take isl_pw_aff *pwaff,
	isl_int mod);

__isl_give isl_pw_aff *isl_pw_aff_scale(__isl_take isl_pw_aff *pwaff,
	isl_int f);
__isl_give isl_pw_aff *isl_pw_aff_scale_down(__isl_take isl_pw_aff *pwaff,
	isl_int f);

__isl_give isl_multi_aff *isl_multi_aff_scale(__isl_take isl_multi_aff *maff,
	isl_int f);

#if defined(__cplusplus)
}
#endif

#endif
