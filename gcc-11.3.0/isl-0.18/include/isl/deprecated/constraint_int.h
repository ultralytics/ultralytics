#ifndef ISL_DEPRECATED_CONSTRAINT_INT_H
#define ISL_DEPRECATED_CONSTRAINT_INT_H

#include <isl/deprecated/int.h>
#include <isl/constraint.h>

#if defined(__cplusplus)
extern "C" {
#endif

void isl_constraint_get_constant(__isl_keep isl_constraint *constraint,
	isl_int *v);
void isl_constraint_get_coefficient(__isl_keep isl_constraint *constraint,
	enum isl_dim_type type, int pos, isl_int *v);
__isl_give isl_constraint *isl_constraint_set_constant(
	__isl_take isl_constraint *constraint, isl_int v);
__isl_give isl_constraint *isl_constraint_set_coefficient(
	__isl_take isl_constraint *constraint,
	enum isl_dim_type type, int pos, isl_int v);

#if defined(__cplusplus)
}
#endif

#endif
