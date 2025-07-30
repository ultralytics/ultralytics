#ifndef ISL_ILP_PRIVATE_H
#define ISL_ILP_PRIVATE_H

#include <isl_int.h>
#include <isl/lp.h>
#include <isl/set.h>

enum isl_lp_result isl_basic_set_solve_ilp(__isl_keep isl_basic_set *bset,
	int max, isl_int *f, isl_int *opt, __isl_give isl_vec **sol_p);

#endif
