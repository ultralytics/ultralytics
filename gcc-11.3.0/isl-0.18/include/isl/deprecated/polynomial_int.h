#ifndef ISL_DEPRECATED_POLYNOMIAL_INT_H
#define ISL_DEPRECATED_POLYNOMIAL_INT_H

#include <isl/deprecated/int.h>
#include <isl/polynomial.h>

#if defined(__cplusplus)
extern "C" {
#endif

__isl_give isl_qpolynomial *isl_qpolynomial_rat_cst_on_domain(
	__isl_take isl_space *space, const isl_int n, const isl_int d);
int isl_qpolynomial_is_cst(__isl_keep isl_qpolynomial *qp,
	isl_int *n, isl_int *d);
__isl_give isl_qpolynomial *isl_qpolynomial_scale(
	__isl_take isl_qpolynomial *qp, isl_int v);

void isl_term_get_num(__isl_keep isl_term *term, isl_int *n);
void isl_term_get_den(__isl_keep isl_term *term, isl_int *d);

__isl_give isl_qpolynomial_fold *isl_qpolynomial_fold_scale(
	__isl_take isl_qpolynomial_fold *fold, isl_int v);

__isl_give isl_pw_qpolynomial_fold *isl_pw_qpolynomial_fold_fix_dim(
	__isl_take isl_pw_qpolynomial_fold *pwf,
	enum isl_dim_type type, unsigned n, isl_int v);

#if defined(__cplusplus)
}
#endif

#endif
