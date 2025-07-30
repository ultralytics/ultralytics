#include <isl/constraint.h>
#include <isl/set.h>

/* This function was never documented and has been replaced by
 * isl_basic_set_add_dims.
 */
__isl_give isl_basic_set *isl_basic_set_add(__isl_take isl_basic_set *bset,
	enum isl_dim_type type, unsigned n)
{
	return isl_basic_set_add_dims(bset, type, n);
}

/* This function was replaced by isl_constraint_alloc_equality.
 */
__isl_give isl_constraint *isl_equality_alloc(__isl_take isl_local_space *ls)
{
	return isl_constraint_alloc_equality(ls);
}

/* This function was replaced by isl_constraint_alloc_inequality.
 */
__isl_give isl_constraint *isl_inequality_alloc(__isl_take isl_local_space *ls)
{
	return isl_constraint_alloc_inequality(ls);
}
