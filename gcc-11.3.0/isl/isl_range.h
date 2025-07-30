#include <isl_bound.h>

int isl_qpolynomial_bound_on_domain_range(__isl_take isl_basic_set *bset,
	__isl_take isl_qpolynomial *poly, struct isl_bound *bound);
__isl_give isl_qpolynomial *isl_qpolynomial_terms_of_sign(
	__isl_keep isl_qpolynomial *poly, int *signs, int sign);
