#ifndef ISL_BOUND_H
#define ISL_BOUND_H

#include <isl/polynomial.h>

struct isl_bound {
	/* input */
	int check_tight;
	int wrapping;
	enum isl_fold type;
	isl_space *dim;
	isl_basic_set *bset;
	isl_qpolynomial_fold *fold;

	/* output */
	isl_pw_qpolynomial_fold *pwf;
	isl_pw_qpolynomial_fold *pwf_tight;
};

#endif
