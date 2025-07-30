#ifndef ISL_POINT_H
#define ISL_POINT_H

#include <stdio.h>
#include <isl/space.h>
#include <isl/val.h>

#if defined(__cplusplus)
extern "C" {
#endif

struct __isl_subclass(isl_basic_set) isl_point;
typedef struct isl_point isl_point;

isl_ctx *isl_point_get_ctx(__isl_keep isl_point *pnt);
__isl_give isl_space *isl_point_get_space(__isl_keep isl_point *pnt);

__isl_give isl_point *isl_point_zero(__isl_take isl_space *dim);
__isl_give isl_point *isl_point_copy(__isl_keep isl_point *pnt);
void isl_point_free(__isl_take isl_point *pnt);

__isl_give isl_val *isl_point_get_coordinate_val(__isl_keep isl_point *pnt,
	enum isl_dim_type type, int pos);
__isl_give isl_point *isl_point_set_coordinate_val(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, __isl_take isl_val *v);

__isl_give isl_point *isl_point_add_ui(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, unsigned val);
__isl_give isl_point *isl_point_sub_ui(__isl_take isl_point *pnt,
	enum isl_dim_type type, int pos, unsigned val);

__isl_give isl_point *isl_point_void(__isl_take isl_space *dim);
isl_bool isl_point_is_void(__isl_keep isl_point *pnt);

__isl_give isl_printer *isl_printer_print_point(
	__isl_take isl_printer *printer, __isl_keep isl_point *pnt);
__isl_give char *isl_point_to_str(__isl_keep isl_point *pnt);
void isl_point_dump(__isl_keep isl_point *pnt);

#if defined(__cplusplus)
}
#endif

#endif
