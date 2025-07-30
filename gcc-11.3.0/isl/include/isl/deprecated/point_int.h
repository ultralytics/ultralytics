#ifndef ISL_DEPRECATED_POINT_INT_H
#define ISL_DEPRECATED_POINT_INT_H

#include <isl/deprecated/int.h>
#include <isl/point.h>

#if defined(__cplusplus)
extern "C" {
#endif

int isl_point_get_coordinate(__isl_keep isl_point *pnt,
	enum isl_dim_type type, int pos, isl_int *v);
__isl_give isl_point *isl_point_set_coordinate(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, isl_int v);

#if defined(__cplusplus)
}
#endif

#endif
