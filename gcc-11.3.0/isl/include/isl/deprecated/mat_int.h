#ifndef ISL_DEPRECATED_MAT_INT_H
#define ISL_DEPRECATED_MAT_INT_H

#include <isl/deprecated/int.h>
#include <isl/mat.h>

#if defined(__cplusplus)
extern "C" {
#endif

int isl_mat_get_element(__isl_keep isl_mat *mat, int row, int col, isl_int *v);
__isl_give isl_mat *isl_mat_set_element(__isl_take isl_mat *mat,
	int row, int col, isl_int v);

#if defined(__cplusplus)
}
#endif

#endif
