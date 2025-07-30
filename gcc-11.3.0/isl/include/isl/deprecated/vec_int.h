#ifndef ISL_DEPRECATED_VEC_INT_H
#define ISL_DEPRECATED_VEC_INT_H

#include <isl/deprecated/int.h>
#include <isl/vec.h>

#if defined(__cplusplus)
extern "C" {
#endif

int isl_vec_get_element(__isl_keep isl_vec *vec, int pos, isl_int *v);
__isl_give isl_vec *isl_vec_set_element(__isl_take isl_vec *vec,
	int pos, isl_int v);

__isl_give isl_vec *isl_vec_set(__isl_take isl_vec *vec, isl_int v);
__isl_give isl_vec *isl_vec_fdiv_r(__isl_take isl_vec *vec, isl_int m);

#if defined(__cplusplus)
}
#endif

#endif
